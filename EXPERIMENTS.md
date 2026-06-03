# Per-turn self-judge experiments

Branch: `per-step-judge-reward` — PR #2 on `tonywu71/prime-rl`.

All runs are in the `self-judge-wordle` wandb project under entity `hcompai`.

---

## What the feature does

Standard GRPO broadcasts one scalar advantage to every completion token of a rollout.
For multi-turn agentic tasks that mis-attributes credit — a failed rollout penalizes its
good turns, a successful one reinforces its mistakes.

The self-judge lets the policy grade each of its own turns (REGRESS / NEUTRAL / PROGRESS /
ACHIEVED) via a short side-channel completion per turn. The orchestrator then reshapes
that turn's per-token advantage (`w[t] = 1 + alpha * label_value[t] * sign(adv)`),
mass-preserving, so uniform labels reproduce scalar GRPO exactly.

---

## Runs

### Run 1 — Wordle + Qwen3-1.7B-Wordle-SFT (thinking), label budget 16
**Config:** `configs/self_judge/wordle.toml` (first version)
**Model:** `PrimeIntellect/Qwen3-1.7B-Wordle-SFT`
**Result:** 100% UNPARSED labels — the SFT model always reasons; the 16-token budget was
spent inside `<think>` before any label word was emitted.
**Fix:** The model ignores `/no_think`; raised the budget. Also found the label parser
was reading from the pre-`</think>` enumeration of all four labels (always returning
REGRESS on the first-match), so the parser was updated to read only the answer after
`</think>`.

---

### Run 2 — Wordle + SFT, budget 512
**Config:** Same model, `progress_label_max_tokens=512`
**Result:** `within_rollout_adv_std=0`, still 100% UNPARSED.
**Root cause:** The label call raised `ModelError()` on every turn, silently swallowed by
the bare `except`. The renderer's `_to_renderer_message` rejects plain `dict` messages
(`ValueError: Unknown message type: <class 'dict'>`); the wrapper was appending raw dicts
instead of typed verifiers `AssistantMessage`/`UserMessage` objects.
**Fix:** Append the action's own `AssistantMessage` and a `UserMessage(content=instruction)`.
Also fixed: inherited the rollout's `sampling_args` (not just `max_tokens`), added
`exc_info=True` logging so future failures are immediately diagnosable.

---

### Run 3 — Wordle + SFT, budget 512, typed messages (feature verified live)
**Config:** Wordle + SFT, 2 GPUs (1 train + 1 infer)
**Result:** ✅ Feature live.
- `self_judge/within_rollout_adv_std = 0.0328`
- Label distribution: REGRESS ~2%, NEUTRAL ~5%, PROGRESS ~25%, ACHIEVED 0.6%→7.7% rising
- `unparsed_rate` ~56–59% on turns 1–4 (the thinking model's reasoning often exceeded 512
  tokens before the verdict; turn 0 worst at ~85%)
- Reward 0.69 from the SFT warm start

---

### Run 4 — Wordle + Qwen3-4B-Instruct-2507 (non-reasoning), budget 32
**Config:** Swapped to a non-reasoning 4B instruct model (no SFT warmup) to test clean
labels without the thinking-budget issue.
**Result:**
- `unparsed_rate = 0` — labels parse perfectly with a 32-token budget ✅
- But `within_rollout_adv_std = 0` and reward ~0.04
- The 4B model doesn't know the Wordle `<guess>` format (no SFT) → fails near-uniformly
  → near-zero advantage variance → the self-judge has nothing to reshape (NOT a bug)

**Lesson:** a non-SFT model gives clean labels but no reward variance. Need both.

---

### Env search — aime2025, reasoning-core-env, openmed_medmcqa, hud-text-2048, math-python
Tested to find a multi-turn env where the non-reasoning 4B gets real outcome variance:

| Env | Multi-turn? | Outcome |
|---|---|---|
| `primeintellect/aime2025` | ❌ `SingleTurnEnv` | No-op (one label → multiplier = 1) |
| `sileod/reasoning-core-env` | ❌ `SingleTurnEnv` (`max_turns=1`) | No-op |
| `maziyar/openmed_medmcqa` | ❌ explicitly single-turn | Skipped |
| `hud/hud-text-2048` | ✅ multi-turn | Needs HUD MCP/Docker backend (`hud_vf_gym` → `hud.clients.MCPClient`) — not a drop-in |
| `math-python` (workspace) | ✅ multi-turn | Needs remote Prime-managed sandbox for code execution |

**Lesson:** single-turn envs make the per-turn self-judge a mathematical no-op.

---

### Run 5 — alphabet-sort + 4B instruct (CUDA OOM, then trainer fix)
**Config:** `base_env_id = "alphabet-sort"`, Qwen3-4B-Instruct-2507, 1 train + 3 infer.
**First attempt:** Trainer CUDA OOM — a 4B with Adam states doesn't fit one 80 GB H100.
**Fix:** Sharded across 2 trainer GPUs (FSDP), dropped infer to 2, cut `seq_len` 8192→4096.

**Second attempt result:**
- Trainer healthy, steps 0+ succeed ✅
- Reward ~0.76, `n_turns=1.6`, `unparsed_rate=0` ✅
- But: **100% PROGRESS labels, `within_rollout_adv_std=0`**
- alphabet-sort is monotonic — each turn only adds correctly placed letters, never regresses.
  The grader ("did this action make progress?") correctly labels every turn PROGRESS.
  Uniform labels → mass-preservation forces multiplier=1 → no-op again.

**Lesson:** the self-judge needs tasks with genuine per-turn ups *and* downs (not just
multi-turn + variance + clean labels).

---

### Runs 6 & 7 — Wordle + SFT, budget 1024 (treatment + control)

**Treatment:** `configs/self_judge/wordle.toml`, cluster `self-judge-wordle`
**Control:** `configs/self_judge/wordle_ctrl.toml`, cluster `self-judge-wordle-ctrl`

The control uses the same wrapper (same rollout cost, same label logging) but omits
`[orchestrator.self_judge]` → plain scalar GRPO. The only difference between arms is
the advantage reshaping.

**Status:** Completed 2026-06-02.
**Treatment wandb:** https://wandb.ai/hcompai/self-judge-wordle/runs/e0d7014581084c8e8f067ac0b0a9454f
**Control wandb:** https://wandb.ai/hcompai/self-judge-wordle/runs/ef4381973ab5479399bb956f68dff420

**Step-0 metrics (treatment):**
- Reward 0.70, seq length 2562 tokens, `within_rollout_adv_std=0.0351`
- Label dist: turn0 85% UNPARSED, turns 1–4 ~52–59% UNPARSED (SFT model reasons;
  budget must cover the whole `<think>` block plus the verdict (512 left ~56% UNPARSED))
- ACHIEVED rising across turns (0.8%→13%): later guesses more often win ✅
- REGRESS ~5%, NEUTRAL ~5%, PROGRESS ~26%: genuinely differentiated labels ✅

**Note on unparsed-label yield:** Bumping budget 512→1024 did **not** reduce unparsed
rate (still ~55%). The labels aren't truncated — the grader sometimes doesn't emit a
clean verdict. Parsed labels default to NEUTRAL (weight 1), diluting but not breaking
the signal. The real lever for yield is grading-prompt engineering, not budget.

---

### Env search — AutomationBench (zapier/AutomationBench)

**Why:** After the Wordle/SFT testbed, the goal was a harder multi-turn agentic task.
See EXPERIMENTS.md §env-search for the prior single-turn dead-ends (aime2025,
reasoning-core, openmed, hud-text-2048, alphabet-sort, arc-agi).

`zapier/AutomationBench` is a `StatefulToolEnv` with **partial-credit rewards** — the
crucial property every prior env lacked. Even incomplete workflows score > 0, so
per-rollout reward variance is structural (not dependent on the model solving tasks
fully). 47 simulated SaaS tools, no sandbox/docker/remote backend.

Key config decisions reached through iteration:
- `toolset="zapier"` (meta-tool discovery via 2 tools) instead of `toolset="api"` (dumps
  all 47 schemas → huge prompts → OOM). The benchmark's own default.
- `domains="simple"` (~200 short 2–3 step tasks) so the 4B can partially solve them.
- 8-GPU recipe (`cp=2`, `ac.freq=1`, `compile`, 4-way trainer FSDP) copied from
  `configs/general_agent/rl_qwen3_4b.toml`; 4-GPU configs OOM'd at step 1.

---

### Runs 8 & 9 — AutomationBench + Qwen3-4B-Instruct-2507 (treatment + control — active)

**Treatment:** `configs/self_judge/automationbench.toml`, cluster `self-judge-ab`
**Control:** `configs/self_judge/automationbench_ctrl.toml`, cluster `self-judge-ab-ctrl`
**Model:** `Qwen/Qwen3-4B-Instruct-2507` (non-reasoning instruct)

The control runs the same wrapper (same rollout cost, same label logging) but omits
`[orchestrator.self_judge]` → plain scalar GRPO. The only difference between arms is
the advantage reshaping.

**Status:** Both running as of 2026-06-03.
**Treatment wandb:** https://wandb.ai/hcompai/self-judge-automationbench/runs/fcad75f402c1464a953b6b51951b40e8
**Control wandb:** https://wandb.ai/hcompai/self-judge-automationbench/runs/70cf96bcc87f40d08cc3c3c1d0955689

**Steps 0–50 metrics (treatment):**
- Reward rises from 0.46 → ~0.73 over 50 steps; mean 0.560 vs control 0.523.
- `within_rollout_adv_std` 0.007–0.014 — self-judge is actively reshaping. ✅
- `unparsed_rate` ~1% — non-reasoning model, tiny 16-token budget. ✅
- Label dist: PROGRESS-dominant (~80–95%/turn); REGRESS/NEUTRAL/ACHIEVED present.

**Note:** `within_rollout_adv_std` is smaller than on Wordle (~0.01 vs ~0.035) because
tool calls mostly succeed (PROGRESS-dominant) in the simple domain. Raising `alpha` or
sharpening the grading prompt to penalise partial/stalled actions would amplify reshaping.

---

## What worked: the right testbed for the self-judge

All three requirements must hold simultaneously:

1. **Clean label parsing** — the label budget must fit the model's output style (a
   non-reasoning model emits one word; a thinking model needs a large budget and the
   parser must read after `</think>`).
2. **Per-rollout reward variance** — if all rollouts score similarly, advantages ≈ 0 and
   `advantage × multiplier = 0` regardless of labels. Needs an SFT-warmed model or a task
   the instruct model can already partially solve.
3. **Per-turn label diversity within a rollout** — labels must differ across turns
   (REGRESS on a bad move, PROGRESS on a good one). Monotonic tasks (alphabet-sort) give
   uniform PROGRESS; single-turn envs give one label.

**Wordle + `Qwen3-1.7B-Wordle-SFT`** was the first testbed to satisfy all three. It
required task-specific SFT and produced ~55% UNPARSED labels (thinking model).

**AutomationBench + `Qwen3-4B-Instruct-2507`** (Runs 8/9) satisfies all three cleanly:
no task-specific SFT, ~1% UNPARSED, and partial-credit reward gives structural variance.

---

## Infrastructure fixes (all committed)

These were discovered and fixed during the runs — not visible in the feature code but
required for any run on this setup:

| Fix | Cause |
|---|---|
| Skip `configs/private` submodule | Private `PrimeIntellect-ai/research-configs` repo; netrc has no github.com creds |
| `prime env install primeintellect/wordle` (not `will/wordle`) | Stale slug in README |
| `LD_PRELOAD` venv NCCL 2.28.9 | Image ships NCCL 2.27.5; soname dedup loads it first; torch 2.11 needs 2.28+ for `ncclDevCommCreate` |
| Off-by-one: append `action.message` before grading | Label prompt lacked the just-generated action — graded the previous turn |
| Parse verdict after `</think>` | Reasoning enumerates all four labels; first-match always returned REGRESS |
| Inherit rollout `sampling_args` | Dropping `extra_body.return_token_ids` caused the renderer to reject the response |
| Append typed `AssistantMessage`/`UserMessage` | Renderer's `_to_renderer_message` rejects plain dicts with `ValueError` |
| `set +x` around `source .env` | `set -eux` echoed secrets into sky job logs (rotate `PRIME_API_KEY` + `HF_TOKEN`) |
| 2 trainer GPUs for the 4B model | Adam optimizer states + 4B weights OOM on one 80 GB H100 |
| `prime` CLI key was stale in `~/.prime/config.json` | CLI reads config.json, not `PRIME_API_KEY` env var; fixed by overwriting the key |
| `toolset="api"` OOM'd the 4B trainer | Dumps all 47 tool schemas into every prompt → huge sequences; `toolset="zapier"` uses 2 meta-tools instead |
| 4-way trainer mesh required for 4B on agentic env | 2-GPU FSDP OOM'd at step 1; `cp=2` + `ac.freq=1` + `compile` (from `general_agent/rl_qwen3_4b.toml`) fits in 79 GiB |
| Judge context redesign: hook `env_response` not `get_model_response` | Original judge saw action but not its consequence; new design appends `UserMessage(observation_{t+1} + instruction)` so the judge grades on actual outcome |
