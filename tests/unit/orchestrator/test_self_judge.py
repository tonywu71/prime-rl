import pytest

from prime_rl.orchestrator.self_judge import (
    SelfJudgeSpec,
    _find_turn_token_spans,
    attach_self_judge_advantages_to_rollout,
)
from prime_rl.transport.types import TrainingSample


def make_sample(turn_lens: list[int], gap: int = 1) -> TrainingSample:
    """Build one sample whose completion_mask encodes `turn_lens` trainable spans
    separated by `gap` masked (env-injection) tokens."""
    mask: list[bool] = []
    for i, n in enumerate(turn_lens):
        if i > 0:
            mask += [False] * gap
        mask += [True] * n
    length = len(mask)
    return TrainingSample(
        prompt_ids=[0],
        prompt_mask=[False],
        completion_ids=list(range(length)),
        completion_mask=mask,
        completion_logprobs=[0.0] * length,
        completion_temperatures=[1.0] * length,
        env_name="test",
    )


def trainable_advantage_mass(sample: TrainingSample) -> float:
    """Sum of per-token advantages over trainable (mask=True) completion tokens."""
    return sum(adv for adv, m in zip(sample.completion_advantages, sample.completion_mask) if m)


def test_find_turn_token_spans_splits_on_masked_gaps():
    mask = [True, True, False, True, False, True, True, True]
    assert _find_turn_token_spans(mask) == [(0, 2), (3, 4), (5, 8)]


def test_find_turn_token_spans_handles_trailing_span():
    assert _find_turn_token_spans([False, True, True]) == [(1, 3)]
    assert _find_turn_token_spans([]) == []


def test_mass_conservation_success():
    sample = make_sample([40, 40, 40, 40, 40])
    spec = SelfJudgeSpec(alpha=0.5)
    labels = ["NEUTRAL", "REGRESS", "PROGRESS", "PROGRESS", "ACHIEVED"]
    attach_self_judge_advantages_to_rollout([sample], labels, 0.35, spec)
    # Σ A^tok_t · len_t == A · L (mass preserved).
    assert trainable_advantage_mass(sample) == pytest.approx(0.35 * 200)


def test_matches_html_worked_example():
    sample = make_sample([40, 40, 40, 40, 40])
    spec = SelfJudgeSpec(alpha=0.5)
    labels = ["NEUTRAL", "REGRESS", "PROGRESS", "PROGRESS", "ACHIEVED"]
    attach_self_judge_advantages_to_rollout([sample], labels, 0.35, spec)
    # Per-token advantages from per_step_reward.html §05 (success rollout).
    expected_per_turn = [0.269, 0.135, 0.404, 0.404, 0.538]
    spans = _find_turn_token_spans(sample.completion_mask)
    for (start, _end), expected in zip(spans, expected_per_turn):
        assert sample.completion_advantages[start] == pytest.approx(expected, abs=1e-3)


def test_uniform_labels_collapse_to_scalar_baseline():
    sample = make_sample([10, 20, 30])
    spec = SelfJudgeSpec(alpha=0.5)
    attach_self_judge_advantages_to_rollout([sample], ["PROGRESS"] * 3, 0.6, spec)
    # Uniform labels → every multiplier 1.0 → identical to scalar GRPO.
    for adv, masked in zip(sample.completion_advantages, sample.completion_mask):
        if masked:
            assert adv == pytest.approx(0.6)


def test_flip_false_achieved_in_failed_rollout():
    sample = make_sample([10, 10])
    flipping = SelfJudgeSpec(alpha=0.5, flip_false_achieved=True, clamp_fail_dampening=False)
    stats = attach_self_judge_advantages_to_rollout([sample], ["NEUTRAL", "ACHIEVED"], -0.4, flipping)
    assert stats["n_flipped_achieved"] == 1
    # ACHIEVED→REGRESS in a failed rollout makes turn 2 the most blamed (most negative).
    spans = _find_turn_token_spans(sample.completion_mask)
    assert sample.completion_advantages[spans[1][0]] < sample.completion_advantages[spans[0][0]]


def test_clamp_fail_dampening_floors_optimistic_labels():
    sample = make_sample([10, 10])
    clamped = SelfJudgeSpec(alpha=0.5, flip_false_achieved=False, clamp_fail_dampening=True)
    attach_self_judge_advantages_to_rollout([sample], ["NEUTRAL", "PROGRESS"], -0.4, clamped)
    spans = _find_turn_token_spans(sample.completion_mask)
    # Equal lengths + both weights clamped to 1.0 → uniform full blame, no dampening.
    assert sample.completion_advantages[spans[0][0]] == pytest.approx(sample.completion_advantages[spans[1][0]])
    assert sample.completion_advantages[spans[1][0]] == pytest.approx(-0.4)


def test_effect_size_stats_present_and_signed():
    sample = make_sample([40, 40, 40, 40, 40])
    spec = SelfJudgeSpec(alpha=0.5)
    stats = attach_self_judge_advantages_to_rollout(
        [sample], ["NEUTRAL", "REGRESS", "PROGRESS", "PROGRESS", "ACHIEVED"], 0.35, spec
    )
    # Per-turn multipliers surfaced for the trace overlay, aligned to turns.
    assert stats["_multipliers"][0] == pytest.approx(0.769, abs=1e-3)
    assert stats["first_turn_multiplier"] == pytest.approx(0.769, abs=1e-3)
    assert stats["last_turn_multiplier"] == pytest.approx(1.538, abs=1e-3)
    assert stats["multiplier_p90"] >= stats["multiplier_p10"]
    # Differential labels => non-zero within-rollout advantage spread.
    assert stats["within_rollout_adv_std"] > 0.0


def test_uniform_labels_zero_within_rollout_spread():
    sample = make_sample([10, 10, 10])
    spec = SelfJudgeSpec(alpha=0.5)
    stats = attach_self_judge_advantages_to_rollout([sample], ["PROGRESS"] * 3, 0.6, spec)
    # Uniform labels collapse to flat GRPO => zero spread.
    assert stats["multiplier_std"] == pytest.approx(0.0, abs=1e-9)
    assert stats["within_rollout_adv_std"] == pytest.approx(0.0, abs=1e-9)


def test_count_mismatch_fails_open():
    sample = make_sample([10, 10, 10])
    spec = SelfJudgeSpec(alpha=0.5)
    stats = attach_self_judge_advantages_to_rollout([sample], ["PROGRESS", "PROGRESS"], 0.5, spec)
    assert stats["n_label_span_mismatch"] == 1
    # Advantages left untouched (scalar broadcast happens downstream in the packer).
    assert sample.completion_advantages is None


def test_pooling_spans_across_multiple_samples():
    # A rollout split into two samples; mass is preserved over the union.
    s0 = make_sample([10, 10])
    s1 = make_sample([10, 10])
    spec = SelfJudgeSpec(alpha=0.5)
    labels = ["PROGRESS", "NEUTRAL", "REGRESS", "ACHIEVED"]
    attach_self_judge_advantages_to_rollout([s0, s1], labels, 0.4, spec)
    total_mass = trainable_advantage_mass(s0) + trainable_advantage_mass(s1)
    assert total_mass == pytest.approx(0.4 * 40)
