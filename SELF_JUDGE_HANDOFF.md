# Per-turn self-judge credit assignment — handoff

Branch: **`per-step-judge-reward`** (off `main`). This doc is a self-contained
summary for a fresh Claude Code instance working in this repo. Nothing here has
been pushed or launched.

## What this is

Standard GRPO assigns **one scalar advantage per rollout**, broadcast to every
completion token. For multi-turn agentic tasks that mis-attributes credit — a
failed rollout penalizes its good turns, a successful one reinforces its
mistakes. This feature lets the policy **grade each of its own turns** with a
discrete progress label (`REGRESS` / `NEUTRAL` / `PROGRESS` / `ACHIEVED`) and
**reshapes that turn's per-token advantage** accordingly, while conserving total
gradient mass (so uniform labels reproduce vanilla GRPO exactly).

This was ported from an earlier implementation in `prime-rl-hcompai`
(branch `tonywu/per-step-self-judge`, commit `07cdfb10`) and the hai-gui-env
desktop "sagent" rail, then **adapted to upstream prime-rl** and made
**hub-agnostic + text-only** so it can run against a simple env from the
[Environments Hub](https://app.primeintellect.ai/dashboard/environments).

## Design (two-completion side-channel)

- The **environment** issues a *second, short* completion each turn off the same
  prompt prefix (KV-prefix-cached), asking the policy to grade its most recent
  action. The parsed label is appended to `state["_progress_labels"]`.
- The label completion is a **pure side-channel**: never delivered downstream,
  **never recorded as a trajectory step, never trained**. It only reshapes the
  *action* tokens' advantages.
- The **orchestrator** maps `progress_labels[k]` → turn-span `k` positionally and
  computes mass-preserving per-token multipliers:
  - `w[t] = 1 + alpha * label_value[t] * sign(adv)` (floored at 0.05)
  - renormalized so `Σ_t m[t]·len[t] == total_completion_len`
  - `label_value`: REGRESS −1, NEUTRAL 0, PROGRESS +1, ACHIEVED +2
- Two anti-hacking levers on **failed** rollouts: `flip_false_achieved`
  (ACHIEVED→REGRESS, since the eval is ground truth) and `clamp_fail_dampening`
  (clamp weights ≥ 1 so optimistic labels can't dampen deserved blame).
- **Fail-open**: on a label/turn-span count mismatch the reshaping is skipped
  (scalar advantage untouched) and reported via `self_judge/n_label_span_mismatch`.

Data flow:
```
env emits state["_progress_labels"]  →  config state_columns hoists it onto the
rollout dict  →  orchestrator.attach_self_judge_advantages_to_rollout(...)  →
sample.completion_advantages (per-token)  →  trainer/batch.py packs it instead of
broadcasting the scalar  →  loss.py (already per-token) trains on it
```

## Commits on this branch

| Hash | Message |
|---|---|
| `362dceec` | feat: add per-turn self-judge credit assignment |
| `86c3904a` | feat: add self-judge-wrapper environment |
| `d814c3ff` | chore: add self-judge example configs |
| `52153ec1` | chore: add skypilot launch yaml for the self-judge wordle experiment |

## Files

### Consumer side (core mechanism)
- `src/prime_rl/orchestrator/self_judge.py` **(new)** — the weighting math:
  `LABEL_VALUE`, `SelfJudgeSpec`, `attach_self_judge_advantages_to_rollout`,
  `_find_turn_token_spans` (trainable spans from `completion_mask`),
  `_compute_multipliers`, `_build_stats` (effect-size diagnostics).
- `src/prime_rl/transport/types.py` — `TrainingSample` gains
  `completion_advantages: list[float] | None`.
- `src/prime_rl/trainer/batch.py` — `prepare_sample` uses `completion_advantages`
  as a per-token override when set (prompt tokens keep the scalar; they're
  loss-masked anyway). `loss.py` is already per-token → unchanged.
- `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py` —
  `SelfJudgeConfig(BaseConfig)` (`alpha=0.3` default, `flip_false_achieved=True`,
  `clamp_fail_dampening=True`) + `self_judge: SelfJudgeConfig | None = None` on
  `OrchestratorConfig`. NB: upstream configs subclass `BaseConfig` (from
  `pydantic_config`), not `BaseModel`.
- `src/prime_rl/orchestrator/orchestrator.py` — resolves the spec at startup,
  calls `attach_...` per trainable rollout (mutating `completion_advantages`),
  surfaces `rollout["_self_judge_multipliers"]` for trace overlays, and logs
  `self_judge/*` (effect-size, averaged over rollouts) + a
  `_progress_label_distribution` helper emitting `progress_labels/turn<k>/<label>_rate`
  with an explicit `UNPARSED` bucket so parse failures stay visible.
- `tests/unit/orchestrator/test_self_judge.py` **(new)** — 11 tests: mass
  conservation, HTML worked example, uniform→baseline, both levers, fail-open,
  multi-sample pooling, effect-size assertions.

### Env emitter (hub-agnostic, vendored)
- `environments/self_judge_wrapper/self_judge_wrapper.py` **(new)** —
  `load_environment(base_env_id, base_args, progress_label_max_tokens=8,
  progress_label_temperature=None, no_think=False)` loads **any** multi-turn
  verifiers env via `vf.load_environment` and **monkeypatches its
  `get_model_response`** to issue the label completion (calling the *pre-wrap*
  bound method so it doesn't recurse) and append to `state["_progress_labels"]`.
  Plus `_parse_progress_label`, `PROGRESS_INSTRUCTION` (text-generalized, no
  screen assumptions), and the `/no_think` lever for thinking models.
- `environments/self_judge_wrapper/{pyproject.toml,README.md}` + a label-parser
  test under `tests/`.
- `pyproject.toml` (root) — registers the package as a uv workspace member,
  `[tool.uv.sources]` entry, and in the `envs` optional-dependency group.

### Example configs
- `configs/self_judge/wordle.toml` (+ `wordle_ctrl.toml`) — **recommended first
  experiment**; wraps `primeintellect/wordle` with `no_think=true`.
- `configs/self_judge/rl.toml` (+ `rl_ctrl.toml`) — zero-install smoke test on
  `alphabet-sort` (workspace member).

### Launch
- `scripts/sky_self_judge_wordle.yaml` — single-node 8×H100 SkyPilot launch.

## How to enable it on any env

```toml
[[orchestrator.train.env]]
id = "self-judge-wrapper"
name = "wordle"                              # metrics group under this name
state_columns = ["_progress_labels"]         # REQUIRED — hoists labels to the orchestrator
args = { base_env_id = "primeintellect/wordle", no_think = true, progress_label_max_tokens = 16 }

[orchestrator.self_judge]                     # omit this block for a control arm
alpha = 0.5
```

- **`state_columns = ["_progress_labels"]` is mandatory.** Without it the labels
  never reach the orchestrator and the feature no-ops, surfacing as ~100%
  `self_judge/n_label_span_mismatch` (intentionally visible, not silent).
- The **control arm** keeps the wrapper (identical rollout cost + label logging)
  but omits `[orchestrator.self_judge]` → plain scalar GRPO. Clean A/B.

## Recommended first experiment: Wordle

Genuine per-turn progress (each guess narrows the answer), interpretable labels,
a ready SFT-warmup checkpoint (`PrimeIntellect/Qwen3-1.7B-Wordle-SFT`) so rollouts
have outcome variance from step 0, and it's cheap (~2–4 H100s). The fork already
ships `examples/wordle/`.

**Critical caveat:** the Wordle SFT model is a *thinking* Qwen3. A tiny-budget
label completion would spend its whole budget inside `<think>` and emit no label
→ all `UNPARSED` (this exact failure sank an earlier desktop run). The
`no_think=true` lever (appends `/no_think`) is why the wordle config sets it; keep
`progress_label_max_tokens` ≥ 16 as belt-and-suspenders.

## Launch (when ready — NOT yet run)

```bash
sky launch -c self-judge-wordle scripts/sky_self_judge_wordle.yaml -y
# control:
sky launch -c self-judge-wordle-ctrl scripts/sky_self_judge_wordle.yaml \
    --env CONFIG=configs/self_judge/wordle_ctrl.toml -y
```
`uv run rl @ <config>` is the unified launcher; `[deployment]` splits the 8 GPUs
(2 train / 6 infer).

**Prerequisites (verify — couldn't be validated locally):**
1. **Submodules** — `uv sync --all-extras` pulls verifiers / renderers /
   **research-environments** (private). The sky yaml inits them over HTTPS via a
   mounted `~/.netrc`; that netrc must grant access (or swap to an SSH-key mount).
2. **Secrets** — gitignored `.env` at repo root with `WANDB_API_KEY`, `HF_TOKEN`,
   `PRIME_API_KEY` (last needed for `prime env install will/wordle`).
3. **Branch** — `workdir: .` uploads this committed tree, so no push needed.

## What to check on the first run (is the feature live, not a no-op?)
- `progress_labels/turn*/unparsed_rate` **low** → labels parse (no_think working).
- `self_judge/within_rollout_adv_std > 0` → advantages actually reshaped per turn.
- `self_judge/n_label_span_mismatch ≈ 0` → label↔span alignment holds.
- Eyeball a trace: do PROGRESS/REGRESS/ACHIEVED line up with good/bad/winning guesses?

## Verified locally (macOS) vs NOT

**Done:** 11/11 `test_self_judge.py` math tests pass (run against the real modules
via sys.modules stubbing to bypass heavy package `__init__`); `ruff check` +
`ruff format --check` clean on all changed files; all TOMLs + the sky YAML parse;
`py_compile` clean.

**NOT done (needs a Linux box / synced env):**
- `uv.lock` was **not** regenerated after registering the env package — the stack
  is linux-x86_64-only and can't sync on macOS. **Run `uv sync --all-extras` on the
  cluster before launch** (the sky yaml does this in `setup`).
- Full test suite / actual env load / a real training step — not run.
- The verifiers/renderers/research-environments submodules are **not checked out**
  in this clone (empty dirs under `deps/`).

## Open questions / things to tune
- `alphabet-sort` configs use `min_turns = max_turns = 3` — guessed; verify the
  env accepts it (the upstream example used 2/2).
- `alpha = 0.5` in the example configs (the `SelfJudgeConfig` default is `0.3`).
- Whether `prime env install will/wordle` needs auth for the (public?) hub env.
- If `--all-extras` is too heavy / research-environments is inaccessible, a
  narrower sync that still includes `self-judge-wrapper` + `flash-attn` + the
  base env would be preferable, but `--extra envs` currently pulls every env.

## Provenance
- Source of truth for the math: `prime-rl-hcompai` `tonywu/per-step-self-judge`
  (commit `07cdfb10`). Spec: `~/Desktop/per_step_reward.html`.
- Env emitter generalized from hai-gui-env `hai_desktop_env/envs/sagent_env.py`
  (the two-completion desktop rail) to a text-only, hub-agnostic wrapper.
