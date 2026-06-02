"""Per-turn self-judge credit assignment (side-channel progress labels).

GRPO broadcasts one scalar advantage to every completion token, so for multi-turn
tasks a failed rollout penalizes its good turns and a successful one reinforces its
mistakes. Here the env grades each turn (REGRESS/NEUTRAL/PROGRESS/ACHIEVED) and this
module reshapes the action tokens' advantages from those labels. Two properties keep
it honest:

1. Sign-aware: ``w[t] = 1 + alpha * label_value[t] * sign(adv)`` — PROGRESS gets more
   positive advantage in wins and REGRESS more blame in losses.
2. Mass-preserving: multipliers are rescaled so ``sum_t m[t]*len[t] == total_len``,
   so uniform labels reproduce scalar GRPO exactly — only differential labels signal.

The ``flip_false_achieved`` and ``clamp_fail_dampening`` levers (see SelfJudgeSpec)
stop optimistic labels from softening deserved blame on failed rollouts.
"""

from dataclasses import dataclass

from prime_rl.transport.types import TrainingSample

#: ACHIEVED is 2x PROGRESS (rarer, and means done rather than closer).
LABEL_VALUE: dict[str, float] = {
    "REGRESS": -1.0,
    "NEUTRAL": 0.0,
    "PROGRESS": 1.0,
    "ACHIEVED": 2.0,
}

#: Fallback for missing/unknown labels: zero weight contribution (keeps the scalar).
DEFAULT_LABEL = "NEUTRAL"

#: Keeps a turn's weight positive so an adverse label can't flip the advantage sign.
WEIGHT_FLOOR = 0.05


@dataclass(frozen=True)
class SelfJudgeSpec:
    """Resolved self-judge weighting parameters.

    Args:
        alpha: Per-turn weight magnitude; ``alpha=0`` collapses to scalar GRPO.
        flip_false_achieved: In failed rollouts, treat ACHIEVED as REGRESS — the
            eval is ground truth, so a claimed-but-rejected turn is a mistake.
        clamp_fail_dampening: In failed rollouts, clamp weights to ``>= 1.0`` so
            optimistic labels can't dampen deserved blame.
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
    """Reshape per-token advantages from per-turn labels, writing into each sample.

    Pools turn spans across all of a rollout's samples (a rollout may split into
    several when extension breaks), maps ``progress_labels[k]`` to the k-th span,
    and writes mass-preserving multipliers into ``completion_advantages`` in place.

    Returns aggregated stats for wandb, or ``None`` when there is nothing to weight.
    Fails open (scalar advantages untouched) on a label/span count mismatch,
    reporting it via ``n_label_span_mismatch``.
    """
    if not samples or scalar_advantage is None:
        return None

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
    # Popped by the orchestrator for trace overlays before stats are aggregated.
    stats["_multipliers"] = multipliers
    return stats


def _find_turn_token_spans(completion_mask: list[bool]) -> list[tuple[int, int]]:
    """(start, end) spans of contiguous trainable tokens — one per assistant turn.

    Inter-turn gaps (env-response re-injection) are ``mask=False`` and split the spans.
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
    # Mass preservation pins the token-weighted mean multiplier to 1, so the
    # within-rollout per-token advantage std is |adv| * sqrt(this) — the headline
    # "is it actually redistributing?" signal (0 iff labels are uniform).
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
