"""Self-judge wrapper: per-turn progress labels as a training side-channel.

Wraps any multi-turn verifiers env so that after each action the policy grades
its own action with a discrete label (REGRESS/NEUTRAL/PROGRESS/ACHIEVED), via a
short side-channel completion per turn. The label is never trained — it is
appended to ``state["_progress_labels"]``, which prime-rl's
``orchestrator.self_judge`` consumes to reshape that turn's per-token advantages.

The judge receives both the action AND its consequence so it grades based on
actual outcome rather than action semantics alone:

    judge_input = full_conversation_including_action_t
                + UserMessage(observation_{t+1} + grading_instructions)

This is implemented by patching ``env_response`` (called after the action has
been appended to the rollout) rather than ``get_model_response``.

Usage (orchestrator config)::

    [[orchestrator.train.env]]
    id = "self-judge-wrapper"
    state_columns = ["_progress_labels"]   # required: hoists labels to the orchestrator
    args = { base_env_id = "alphabet-sort" }

    [orchestrator.self_judge]
    alpha = 0.5
"""

import logging
import re
from typing import Any

import verifiers as vf
from verifiers.types import UserMessage

logger = logging.getLogger(__name__)

PROGRESS_LABELS: tuple[str, ...] = ("REGRESS", "NEUTRAL", "PROGRESS", "ACHIEVED")

#: Distinct from NEUTRAL so parse failures stay visible downstream
#: (progress_labels/turn*/unparsed_rate) instead of masquerading as real grades.
UNPARSED_PROGRESS_LABEL = "UNPARSED"

STATE_PROGRESS_LABELS = "_progress_labels"

# The observation text (observation_{t+1}) is prepended by the caller when
# available, so the judge grades based on actual outcome (tool result, env reply).
PROGRESS_INSTRUCTION = (
    "Grade the agent's MOST RECENT action (the last assistant message above). "
    "Reply with EXACTLY ONE word — no punctuation:\n"
    "REGRESS — the action moved away from the goal, caused an error, or failed.\n"
    "NEUTRAL — no meaningful change, or this is the first action.\n"
    "PROGRESS — the action moved measurably closer to the goal.\n"
    "ACHIEVED — the task goal is now fully satisfied.\n"
    "Answer:"
)

#: On thinking models a tiny label budget is otherwise spent inside <think>,
#: emitting no label word.
_NO_THINK_SUFFIX = " /no_think"

_PROGRESS_LABEL_RE = re.compile(r"\b(REGRESS|NEUTRAL|PROGRESS|ACHIEVED)\b")


def _format_observation(env_messages: vf.Messages) -> str:
    """Extract plain text from env response messages for the judge's context."""
    parts: list[str] = []
    for msg in env_messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "") or ""
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "").strip()
                    if text:
                        parts.append(text)
    return "\n".join(parts)


def load_environment(
    base_env_id: str = "alphabet-sort",
    base_args: dict[str, Any] | None = None,
    progress_label_max_tokens: int = 8,
    progress_label_temperature: float | None = None,
    no_think: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """Load a multi-turn hub env and wrap ``env_response`` with the label side-channel.

    The judge fires after the environment responds, so it sees:
        conversation_including_action_t + UserMessage(observation_{t+1} + instruction)

    Args:
        base_env_id: Verifiers env ID to wrap.
        base_args: Forwarded to the base env's ``load_environment``.
        progress_label_max_tokens: Token budget per label completion.
        progress_label_temperature: ``None`` inherits the rollout's sampling args.
        no_think: Append ``/no_think`` to suppress reasoning on thinking models.
    """
    env = vf.load_environment(base_env_id, **(base_args or {}))
    instruction = PROGRESS_INSTRUCTION + (_NO_THINK_SUFFIX if no_think else "")
    # Captured before patching so the judge call never recurses into env_response.
    original_get_model_response = env.get_model_response
    original_env_response = env.env_response

    async def _generate_progress_label(
        state: vf.State,
        messages: vf.Messages,
        env_messages: vf.Messages,
    ) -> str:
        # messages already contains action_t as its last element (appended before
        # env_response is called). Append a UserMessage that combines observation_{t+1}
        # with the grading instruction so the judge sees actual outcomes.
        observation = _format_observation(env_messages)
        grading_content = f"{observation}\n\n{instruction}" if observation else instruction
        label_prompt = list(messages) + [UserMessage(content=grading_content)]

        # Inherit the rollout's sampling args (e.g. extra_body's return_token_ids,
        # which the renderer needs to parse the response) and only override the budget.
        sampling_args: dict[str, Any] = dict(state.get("sampling_args") or {})
        sampling_args["max_tokens"] = progress_label_max_tokens
        if progress_label_temperature is not None:
            sampling_args["temperature"] = progress_label_temperature
        try:
            response = await original_get_model_response(
                state, label_prompt, tool_defs=None, sampling_args=sampling_args
            )
        except Exception:
            # Tolerate a failed label (don't kill the rollout) but never silently:
            # a systematic failure would otherwise masquerade as 100% UNPARSED.
            logger.warning("self-judge label completion failed", exc_info=True)
            return UNPARSED_PROGRESS_LABEL
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return _parse_progress_label(content or "")

    async def env_response_with_label(messages: vf.Messages, state: vf.State, **kw: Any):
        # Call the real env_response first — state is mutated in place, env_msgs is
        # observation_{t+1}. messages[-1] is action_t (already appended by the rollout).
        env_msgs = await original_env_response(messages, state, **kw)
        label = await _generate_progress_label(state, messages, env_msgs or [])
        state.setdefault(STATE_PROGRESS_LABELS, []).append(label)
        return env_msgs

    # Instance attribute shadows the class method (verifiers calls self.env_response).
    env.env_response = env_response_with_label
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
