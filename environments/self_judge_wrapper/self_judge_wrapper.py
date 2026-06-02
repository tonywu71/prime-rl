"""Self-judge wrapper environment: per-turn progress labels as a side-channel.

Wraps any multi-turn verifiers environment so that, after each action turn, the
policy grades its own most-recent action with a discrete progress label
(``REGRESS`` / ``NEUTRAL`` / ``PROGRESS`` / ``ACHIEVED``). The grade is produced
by a SECOND short completion off the same prompt prefix (the turn's KV cache is
shared, so it prefix-caches), parsed to one label, and appended to
``state["_progress_labels"]``.

The label is a pure side-channel: it is never delivered downstream and its tokens
are never trained (we never record it as a trajectory step). prime-rl's
``orchestrator.self_judge`` consumes the ``_progress_labels`` list to reshape the
*action* tokens' per-token advantages, mass-preserving.

This is the text-only, hub-agnostic counterpart of hai-gui-env's two-completion
desktop rail: point it at any simple multi-turn hub env (e.g. ``alphabet-sort``).

Usage (in an orchestrator config)::

    [[orchestrator.train.env]]
    id = "self-judge-wrapper"
    name = "alphabet-sort"            # display name; metrics group under this
    state_columns = ["_progress_labels"]
    args = { base_env_id = "alphabet-sort", base_args = { min_turns = 3, max_turns = 5 } }

    [orchestrator.self_judge]
    alpha = 0.5

``state_columns = ["_progress_labels"]`` is required: it tells prime-rl to hoist
the per-turn labels from rollout state onto the rollout record the orchestrator
reads. Without it the labels never reach the orchestrator and the self-judge
no-ops (surfaced as a 100% ``self_judge/n_label_span_mismatch``).
"""

import re
from typing import Any

import verifiers as vf

#: Discrete per-turn progress labels (ordered by increasing progress value).
PROGRESS_LABELS: tuple[str, ...] = ("REGRESS", "NEUTRAL", "PROGRESS", "ACHIEVED")

#: Emitted (instead of a real label) when the label completion produced no
#: parseable label or the call errored. Kept DISTINCT from NEUTRAL so silent
#: parse failures are visible downstream (prime-rl logs
#: ``progress_labels/turn*/unparsed_rate``) rather than masquerading as genuine
#: NEUTRAL grades. prime-rl treats it as zero-signal (weight 1).
UNPARSED_PROGRESS_LABEL = "UNPARSED"

#: Per-rollout list of per-turn labels (one entry per action turn) on state.
STATE_PROGRESS_LABELS = "_progress_labels"

#: Self-contained one-shot grading prompt appended to the turn's prefix. Text-only
#: (no screen/observation assumptions): the conversation above is the only context.
PROGRESS_INSTRUCTION = (
    "Before continuing, grade YOUR MOST RECENT action using the conversation above. "
    "Reply with EXACTLY ONE word, no punctuation, chosen from:\n"
    "REGRESS - the last action moved away from the goal, undid progress, or caused an error.\n"
    "NEUTRAL - no meaningful change, or this is the first action.\n"
    "PROGRESS - the last action moved measurably closer to the goal.\n"
    "ACHIEVED - the task goal is now fully satisfied.\n"
    "Answer with only the single word."
)

#: Appended to the grading instruction when ``no_think=True`` to suppress reasoning
#: for the (tiny-budget) label completion on Qwen3-style thinking models — otherwise
#: the budget is spent inside ``<think>`` and no label word is ever emitted.
_NO_THINK_SUFFIX = " /no_think"

_PROGRESS_LABEL_RE = re.compile(r"\b(REGRESS|NEUTRAL|PROGRESS|ACHIEVED)\b")


def load_environment(
    base_env_id: str = "alphabet-sort",
    base_args: dict[str, Any] | None = None,
    progress_label_max_tokens: int = 8,
    progress_label_temperature: float | None = None,
    no_think: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """Load a multi-turn hub env and wrap it with the self-judge label side-channel.

    Args:
        base_env_id: Verifiers environment ID of the multi-turn env to wrap (e.g.
            ``"alphabet-sort"``). Loaded via ``vf.load_environment``.
        base_args: Keyword arguments forwarded to the base env's ``load_environment``.
        progress_label_max_tokens: Max tokens for each label completion. Keep small
            (one word). Raise it (or set ``no_think=True``) for thinking models.
        progress_label_temperature: Sampling temperature for the label completion.
            ``None`` inherits the rollout's sampling args.
        no_think: Append ``/no_think`` to the grading prompt to suppress reasoning on
            Qwen3-style thinking models (a no-op on non-thinking models).
        **kwargs: Ignored extra args (tolerated for forward-compatible configs).

    Returns:
        The base environment, with ``get_model_response`` wrapped to append one
        progress label per action turn to ``state["_progress_labels"]``.
    """
    env = vf.load_environment(base_env_id, **(base_args or {}))
    instruction = PROGRESS_INSTRUCTION + (_NO_THINK_SUFFIX if no_think else "")
    # The pre-wrap bound method. The label completion calls THIS (not the wrapper)
    # so grading never recurses into another label.
    original_get_model_response = env.get_model_response

    async def _generate_progress_label(state: vf.State, prompt: vf.Messages) -> str:
        sampling_args: dict[str, Any] = {"max_tokens": progress_label_max_tokens}
        if progress_label_temperature is not None:
            sampling_args["temperature"] = progress_label_temperature
        label_prompt = list(prompt) + [{"role": "user", "content": instruction}]
        try:
            response = await original_get_model_response(
                state, label_prompt, tool_defs=None, sampling_args=sampling_args
            )
        except Exception:
            return UNPARSED_PROGRESS_LABEL
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return _parse_progress_label(content or "")

    async def get_model_response_with_label(state: vf.State, prompt: vf.Messages, *args: Any, **kw: Any):
        response = await original_get_model_response(state, prompt, *args, **kw)
        label = await _generate_progress_label(state, prompt)
        state.setdefault(STATE_PROGRESS_LABELS, []).append(label)
        return response

    # Instance attribute shadows the class method; verifiers calls
    # ``self.get_model_response(state, prompt)`` and finds this closure (no self).
    env.get_model_response = get_model_response_with_label
    return env


def _parse_progress_label(text: str) -> str:
    """Extract the first progress label from a label completion (case-insensitive).

    Returns :data:`UNPARSED_PROGRESS_LABEL` when the completion contains none of the
    four labels (e.g. the model spent its budget reasoning and never emitted the
    word), so the failure is observable downstream rather than silently NEUTRAL.

    Args:
        text: The raw label completion text.

    Returns:
        One of :data:`PROGRESS_LABELS`, or :data:`UNPARSED_PROGRESS_LABEL`.
    """
    if not text:
        return UNPARSED_PROGRESS_LABEL
    match = _PROGRESS_LABEL_RE.search(text.upper())
    if match is None:
        return UNPARSED_PROGRESS_LABEL
    return match.group(1)
