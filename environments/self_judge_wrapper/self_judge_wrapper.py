"""Self-judge wrapper: per-turn progress labels as a training side-channel.

Wraps any multi-turn verifiers env so that after each action the policy grades
its own action with a discrete label (REGRESS/NEUTRAL/PROGRESS/ACHIEVED), via a
second short completion off the same prefix. The label is never trained — it is
appended to ``state["_progress_labels"]``, which prime-rl's
``orchestrator.self_judge`` consumes to reshape that turn's per-token advantages.

Usage (orchestrator config)::

    [[orchestrator.train.env]]
    id = "self-judge-wrapper"
    state_columns = ["_progress_labels"]   # required: hoists labels to the orchestrator
    args = { base_env_id = "alphabet-sort" }

    [orchestrator.self_judge]
    alpha = 0.5
"""

import re
from typing import Any

import verifiers as vf

PROGRESS_LABELS: tuple[str, ...] = ("REGRESS", "NEUTRAL", "PROGRESS", "ACHIEVED")

#: Distinct from NEUTRAL so parse failures stay visible downstream
#: (progress_labels/turn*/unparsed_rate) instead of masquerading as real grades.
UNPARSED_PROGRESS_LABEL = "UNPARSED"

STATE_PROGRESS_LABELS = "_progress_labels"

PROGRESS_INSTRUCTION = (
    "Before continuing, grade YOUR MOST RECENT action using the conversation above. "
    "Reply with EXACTLY ONE word, no punctuation, chosen from:\n"
    "REGRESS - the last action moved away from the goal, undid progress, or caused an error.\n"
    "NEUTRAL - no meaningful change, or this is the first action.\n"
    "PROGRESS - the last action moved measurably closer to the goal.\n"
    "ACHIEVED - the task goal is now fully satisfied.\n"
    "Answer with only the single word."
)

#: On thinking models a tiny label budget is otherwise spent inside <think>,
#: emitting no label word.
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
    """Load a multi-turn hub env and wrap ``get_model_response`` with the label side-channel.

    Args:
        base_env_id: Verifiers env ID to wrap.
        base_args: Forwarded to the base env's ``load_environment``.
        progress_label_max_tokens: Token budget per label completion.
        progress_label_temperature: ``None`` inherits the rollout's sampling args.
        no_think: Append ``/no_think`` to suppress reasoning on thinking models.
    """
    env = vf.load_environment(base_env_id, **(base_args or {}))
    instruction = PROGRESS_INSTRUCTION + (_NO_THINK_SUFFIX if no_think else "")
    # Grading calls the pre-wrap method so it never recurses into another label.
    original_get_model_response = env.get_model_response

    async def _generate_progress_label(state: vf.State, prompt: vf.Messages, action: vf.Response) -> str:
        sampling_args: dict[str, Any] = {"max_tokens": progress_label_max_tokens}
        if progress_label_temperature is not None:
            sampling_args["temperature"] = progress_label_temperature
        # `prompt` doesn't yet contain the action just generated; append it so the
        # label grades this turn (not the previous one) and the final turn is graded.
        action_message = getattr(action, "message", None)
        action_content = getattr(action_message, "content", "") if action_message is not None else ""
        label_prompt = list(prompt) + [
            {"role": "assistant", "content": action_content or ""},
            {"role": "user", "content": instruction},
        ]
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
        label = await _generate_progress_label(state, prompt, response)
        state.setdefault(STATE_PROGRESS_LABELS, []).append(label)
        return response

    # Instance attribute shadows the class method (verifiers calls self.get_model_response).
    env.get_model_response = get_model_response_with_label
    return env


def _parse_progress_label(text: str) -> str:
    """The label from the answer, else UNPARSED (so failures stay observable).

    Thinking models enumerate all four labels inside <think>, so grade only the
    answer after </think>; an unclosed think block means the answer was truncated.
    """
    if not text:
        return UNPARSED_PROGRESS_LABEL
    answer = text.upper()
    if "</THINK>" in answer:
        answer = answer.rsplit("</THINK>", 1)[1]
    elif "<THINK>" in answer:
        return UNPARSED_PROGRESS_LABEL
    match = _PROGRESS_LABEL_RE.search(answer)
    if match is None:
        return UNPARSED_PROGRESS_LABEL
    return match.group(1)
