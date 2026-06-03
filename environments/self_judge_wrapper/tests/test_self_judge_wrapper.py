"""Unit tests for the self-judge label parser.

Covers the pure ``_parse_progress_label`` helper: the label completion is free
text, so parsing must tolerate casing and surrounding words and degrade to
UNPARSED (never silently NEUTRAL) on anything unrecognized. Importing the module
pulls in ``verifiers``; skip when it is not installed (e.g. a slim dev env).
"""

import pytest

pytest.importorskip("verifiers")

from self_judge_wrapper import (  # noqa: E402 — after importorskip guard
    PROGRESS_LABELS,
    UNPARSED_PROGRESS_LABEL,
    _parse_progress_label,
)


@pytest.mark.parametrize("label", PROGRESS_LABELS)
def test_parses_each_bare_label(label: str):
    assert _parse_progress_label(label) == label


def test_case_insensitive():
    assert _parse_progress_label("progress") == "PROGRESS"
    assert _parse_progress_label("Achieved") == "ACHIEVED"


def test_extracts_from_surrounding_text():
    assert _parse_progress_label("The agent made PROGRESS this turn.") == "PROGRESS"
    assert _parse_progress_label("answer: regress\n") == "REGRESS"


def test_first_label_wins():
    assert _parse_progress_label("NEUTRAL, though arguably PROGRESS") == "NEUTRAL"


def test_unrecognized_marked_unparsed_not_neutral():
    # Failures must be visible as UNPARSED, not silently collapsed into NEUTRAL.
    assert _parse_progress_label("") == UNPARSED_PROGRESS_LABEL
    assert _parse_progress_label("no idea") == UNPARSED_PROGRESS_LABEL
    # Substring of a larger word must not match (word boundary).
    assert _parse_progress_label("PROGRESSION") == UNPARSED_PROGRESS_LABEL
    # A genuine NEUTRAL is still distinct from a parse failure.
    assert _parse_progress_label("NEUTRAL") == "NEUTRAL"
