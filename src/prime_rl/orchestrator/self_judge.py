"""Per-turn self-judge credit assignment (side-channel progress labels).

GRPO assigns one scalar advantage per rollout, broadcast to every completion
token. For multi-turn agentic tasks this mis-attributes credit: a failed
rollout penalizes its good turns, a successful one reinforces its mistakes.

Here the policy grades each of its own turns with a discrete progress label
(``REGRESS`` / ``NEUTRAL`` / ``PROGRESS`` / ``ACHIEVED``). The environment
generates these labels as a *separate, short* completion per turn that shares
the turn's KV prefix (see the ``self_judge_wrapper`` environment) and propagates
the ordered list via ``state["_progress_labels"]``. The label tokens are never
trained — this module only *reshapes* the action tokens' advantages.

The reshaping has two anti-hacking properties:

1. **Sign-aware weighting** — ``w[t] = 1 + alpha * label_value[t] * sign(adv)``.
   In successful rollouts PROGRESS turns get more positive advantage and REGRESS
   turns less; in failed rollouts REGRESS turns get more blame.
2. **Token-mass preservation** — per-token multipliers are rescaled so that
   ``sum_t m[t] * len[t] == total_completion_len``. Uniform labelling therefore
   reproduces the scalar baseline exactly; only *differential* labels across a
   rollout's turns create per-step signal.

Two failure-mode-specific levers refine failed rollouts: ``flip_false_achieved``
(ACHIEVED -> REGRESS, since the rollout eval is ground truth) and
``clamp_fail_dampening`` (clamp weights to >= 1 so optimistic PROGRESS labels
get full scalar blame rather than a dampened share).
"""

from dataclasses import dataclass

from prime_rl.transport.types import TrainingSample

# Label -> relative progress value. ACHIEVED is 2x PROGRESS to reflect both its
# rarity (one turn per successful rollout) and its semantic weight (task done,
# not just moved closer). REGRESS is symmetric to PROGRESS.
LABEL_VALUE: dict[str, float] = {
    "REGRESS": -1.0,
    "NEUTRAL": 0.0,
    "PROGRESS": 1.0,
    "ACHIEVED": 2.0,
}

# Fallback for a missing / out-of-vocabulary label. NEUTRAL contributes zero to
# the weight formula — i.e. that turn keeps the scalar GRPO advantage.
DEFAULT_LABEL = "NEUTRAL"

# Non-negative floor so a strong adverse label can't flip the advantage sign on
# its own turn (a no-op with alpha <= 0.5 and label_value in [-1, 2]).
WEIGHT_FLOOR = 0.05


@dataclass(frozen=True)
class SelfJudgeSpec:
    """Resolved self-judge weighting parameters.

    Args:
        alpha: Per-turn weight magnitude. With ``alpha=0.5`` a PROGRESS turn gets
            ``w=1.5`` and a REGRESS turn ``w=0.5`` in a successful rollout (signs
            flip in failed rollouts). ``alpha=0`` collapses to scalar GRPO.
        flip_false_achieved: In failed rollouts, re-interpret ACHIEVED as REGRESS.
            The rollout eval is ground truth; a claimed-but-rejected completion is
            a high-confidence mistake, not a near-success.
        clamp_fail_dampening: In failed rollouts, clamp per-turn weights to
            ``>= 1.0`` so optimistic PROGRESS labels get full scalar blame rather
            than a reduced (dampened) share.
    """

    alpha: float
    flip_false_achieved: bool = True
    clamp_fail_dampening: bool = True


def attach_self_judge_advantages_to_rollout(
    samples: list[TrainingSample],
    progress_labels: list[str],
    scalar_advantage: float | None,
    spec: SelfJudgeSpec,
) -> dict[str, int | float] | None:
    """Reshape per-token advantages from per-turn progress labels.

    Pools turn spans across all of a rollout's samples (a rollout may split into
    several ``TrainingSample``s when the extension property breaks), maps
    ``progress_labels[k]`` to the k-th turn span positionally, and writes
    mass-preserving per-token multipliers into each sample's
    ``completion_advantages`` in place.

    Args:
        samples: All training samples of a single rollout.
        progress_labels: One label per action turn, in turn order.
        scalar_advantage: The rollout's scalar GRPO advantage.
        spec: Resolved weighting parameters.

    Returns:
        Aggregated stats for wandb, or ``None`` when there is nothing to weight.
        Fails open (leaves scalar advantages untouched) on a label/span count
        mismatch, reporting it via ``n_label_span_mismatch``.
    """
    if not samples or scalar_advantage is None:
        return None

    # (sample_idx, (start, end_exclusive)) for every assistant turn span, in
    # rollout order (sample 0's turns, then sample 1's, ...).
    flat_spans: list[tuple[int, tuple[int, int]]] = []
    for sample_idx, sample in enumerate(samples):
        for span in _find_turn_token_spans(sample.completion_mask):
            flat_spans.append((sample_idx, span))

    if not flat_spans:
        return None

    if len(progress_labels) != len(flat_spans):
        return {"n_label_span_mismatch": 1, "n_turns": len(flat_spans), "n_labels": len(progress_labels)}

    labels = [label if label in LABEL_VALUE else DEFAULT_LABEL for label in progress_labels]
    span_lens = [end - start for _, (start, end) in flat_spans]
    total_len = sum(span_lens)
    if total_len == 0:
        return {"n_turns": len(flat_spans)}

    multipliers, n_flipped = _compute_multipliers(labels, span_lens, total_len, scalar_advantage, spec)
    if multipliers is None:
        return {"n_turns": len(flat_spans)}

    for sample in samples:
        sample.completion_advantages = [scalar_advantage] * len(sample.completion_ids)
    for (sample_idx, (start, end)), multiplier in zip(flat_spans, multipliers):
        per_token = samples[sample_idx].completion_advantages
        advantage_t = scalar_advantage * multiplier
        for pos in range(start, end):
            per_token[pos] = advantage_t

    stats = _build_stats(labels, multipliers, span_lens, scalar_advantage, n_flipped)
    # Per-turn multipliers for the orchestrator to surface (trace overlay); popped
    # before the stats dict is aggregated into scalars.
    stats["_multipliers"] = multipliers
    return stats


def _find_turn_token_spans(completion_mask: list[bool]) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) spans of contiguous trainable tokens.

    Each span is one assistant turn's generated tokens; the inter-turn gaps
    (env-response prompt re-injection) are ``mask=False`` and separate the spans.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, masked in enumerate(completion_mask):
        if masked and start is None:
            start = i
        elif not masked and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(completion_mask)))
    return spans


def _compute_multipliers(
    labels: list[str],
    span_lens: list[int],
    total_len: int,
    scalar_advantage: float,
    spec: SelfJudgeSpec,
) -> tuple[list[float] | None, int]:
    """Turn labels into mass-preserving per-turn advantage multipliers."""
    adv_sign = 1.0 if scalar_advantage >= 0 else -1.0

    n_flipped = 0
    effective_labels: list[str] = []
    for label in labels:
        if spec.flip_false_achieved and adv_sign < 0 and label == "ACHIEVED":
            effective_labels.append("REGRESS")
            n_flipped += 1
        else:
            effective_labels.append(label)

    raw_weights = [max(WEIGHT_FLOOR, 1.0 + spec.alpha * LABEL_VALUE[label] * adv_sign) for label in effective_labels]
    if spec.clamp_fail_dampening and adv_sign < 0:
        raw_weights = [max(1.0, weight) for weight in raw_weights]

    weighted_sum = sum(weight * length for weight, length in zip(raw_weights, span_lens))
    if weighted_sum <= 0:
        return None, n_flipped
    scale = total_len / weighted_sum
    return [weight * scale for weight in raw_weights], n_flipped


def _build_stats(
    labels: list[str],
    multipliers: list[float],
    span_lens: list[int],
    scalar_advantage: float,
    n_flipped: int,
) -> dict[str, int | float]:
    """Aggregate per-rollout self-judge diagnostics for wandb."""
    counts = {label: labels.count(label) for label in LABEL_VALUE}
    total_len = sum(span_lens)
    # Token-weighted variance of the multipliers. Mass preservation forces the
    # token-weighted mean to 1.0, so the within-rollout per-token advantage std is
    # exactly |scalar_advantage| * sqrt(this) — the headline "is it redistributing?"
    # signal (0 when labels are uniform, i.e. identical to flat GRPO).
    weighted_var = sum(length * (m - 1.0) ** 2 for m, length in zip(multipliers, span_lens)) / total_len
    return {
        "n_turns": len(labels),
        "n_flipped_achieved": n_flipped,
        "label_count_REGRESS": counts["REGRESS"],
        "label_count_NEUTRAL": counts["NEUTRAL"],
        "label_count_PROGRESS": counts["PROGRESS"],
        "label_count_ACHIEVED": counts["ACHIEVED"],
        "mean_multiplier": sum(multipliers) / len(multipliers),
        "max_multiplier": max(multipliers),
        "min_multiplier": min(multipliers),
        "multiplier_std": _std(multipliers),
        "multiplier_p10": _percentile(multipliers, 10.0),
        "multiplier_p90": _percentile(multipliers, 90.0),
        "first_turn_multiplier": multipliers[0],
        "last_turn_multiplier": multipliers[-1],
        "within_rollout_adv_std": abs(scalar_advantage) * (weighted_var**0.5),
    }


def _std(values: list[float]) -> float:
    """Population standard deviation (0 for empty / singleton)."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated q-th percentile (q in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
