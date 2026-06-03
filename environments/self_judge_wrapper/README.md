# self-judge-wrapper

### Overview
- **Environment ID**: `self-judge-wrapper`
- **Short description**: Wraps any multi-turn verifiers environment so the policy
  grades its own most-recent action each turn (a side-channel "self-judge" label),
  enabling prime-rl's per-turn credit assignment without modifying the base env.
- **Tags**: self-judge, multi-turn, credit-assignment, train

### How it works
After each action turn, a SECOND short completion is issued off the same prompt
prefix asking the policy to grade its most recent action with one word —
`REGRESS` / `NEUTRAL` / `PROGRESS` / `ACHIEVED` (or `UNPARSED` on failure). The
parsed label is appended to `state["_progress_labels"]`. The label completion is
never delivered downstream and its tokens are never trained; prime-rl's
`orchestrator.self_judge` consumes the list to reshape the action tokens'
per-token advantages, mass-preserving.

This is the text-only, hub-agnostic counterpart of hai-gui-env's two-completion
desktop rail. Point it at any simple multi-turn hub env.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `base_env_id` | str | `"alphabet-sort"` | Verifiers env ID of the multi-turn env to wrap. |
| `base_args` | dict | `{}` | Kwargs forwarded to the base env's `load_environment`. |
| `progress_label_max_tokens` | int | `8` | Max tokens for each label completion (one word). |
| `progress_label_temperature` | float \| None | `None` | Sampling temp for the label; `None` inherits the rollout's. |
| `no_think` | bool | `False` | Append `/no_think` to the grading prompt (Qwen3 thinking models). |

### Required config wiring
Declare `state_columns = ["_progress_labels"]` on the env config so prime-rl hoists
the labels onto the rollout record, and enable the orchestrator-side reshaping:

```toml
[[orchestrator.train.env]]
id = "self-judge-wrapper"
name = "alphabet-sort"
state_columns = ["_progress_labels"]
args = { base_env_id = "alphabet-sort", base_args = { min_turns = 3, max_turns = 5 } }

[orchestrator.self_judge]
alpha = 0.5
```

Without `state_columns`, the labels never reach the orchestrator and the self-judge
no-ops (surfaced as a ~100% `self_judge/n_label_span_mismatch`).

### Metrics
The wrapper itself emits no rewards (the base env's rubric is untouched). prime-rl
logs `self_judge/*` effect-size stats and `progress_labels/turn*/<label>_rate`
distributions when `[orchestrator.self_judge]` is set.
