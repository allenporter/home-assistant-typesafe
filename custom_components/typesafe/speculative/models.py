"""Canonical question and answer data models for System One decision evaluation.

System One models (such as Jev) make units of AI intelligence usable as programming
primitives: small, focused judgments that return typed answers and calibrated probabilities
rather than open-ended text generation.

The primitives are structured as paired Question -> Answer types:
1. Choice: Selects one option from a defined set of candidate criteria.
2. Noul: Evaluates the calibrated probability that a condition holds (boolean yes/no).
3. Score: Evaluates a position along an ordered rubric of discrete level descriptions.

Each question operates over a shared `state` payload (such as user utterance, context,
or entity registry metadata), and returns structured answers that application code can
combine and branch upon deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# 1. Choice Primitive: Categorical Selection
# ============================================================================


@dataclass(slots=True, frozen=True)
class ChoiceQuestion:
    """Categorical choice question with candidate criteria.

    Choice questions evaluate the input state against a set of mutually exclusive
    options and select the single best matching candidate.
    """

    instructions: str | dict[str, Any] | list[Any]
    """The core question or prompt directing the model's judgment.

    - `str`: Standard single-sentence prompt (e.g. "Which device is targeted?").
    - `dict[str, Any]`: Structured instructions separating task, definitions,
      and exclusion rules (e.g. {"task": "Route intent", "rules": "..."}).
    - `list[Any]`: Ordered checklist or sequential criteria for the model.
    Use a string by default; choose a dict or list when complex disambiguation,
    boundary constraints, or specific exclusions must be preserved semantically.
    """

    criteria: dict[str, str | None]
    """A mapping from candidate option identifier to its semantic description.

    - Key: The candidate identifier used in code (e.g. "HassTurnOn", "light.kitchen").
    - Value: A natural-language description describing what that option covers
      and when it should apply. The model evaluates and compares these competing
      descriptions against the input state to assign probabilities.
    """


@dataclass(slots=True, frozen=True)
class ChoiceAnswer:
    """Answer returned for a categorical choice question."""

    choice: str
    """The identifier of the selected option from `criteria` keys."""

    confidence: float
    """Summary confidence in the selected choice in [0.0, 1.0], derived from probability concentration across options."""

    probabilities: dict[str, float] = field(default_factory=dict)
    """Probability distribution mapping each candidate option key to its evaluated probability (sums to ~1.0)."""

    action: dict[str, Any] = field(default_factory=dict)
    """Optional structured tool call or action payload associated with the choice."""


# ============================================================================
# 2. Noul Primitive: Calibrated Boolean Judgment
# ============================================================================


@dataclass(slots=True, frozen=True)
class NoulQuestion:
    """Calibrated boolean (Yes/No) decision question.

    Noul evaluates whether a specific condition holds on the state, returning the
    calibrated probability P(true). It is ideal for gating, boundary verification,
    and compound command detection.
    """

    instructions: str | dict[str, Any] | list[Any]
    """The condition or assertion to evaluate (e.g. "Does the request contain multiple distinct commands or conjunctions?").

    - `str`: Simple condition statement.
    - `dict[str, Any]` or `list[Any]`: Structured definitions with positive and negative edge cases.
    """


@dataclass(slots=True, frozen=True)
class NoulAnswer:
    """Answer returned for a calibrated boolean (Yes/No) noul question."""

    noul: float  # P(true) in [0.0, 1.0]
    """Calibrated probability P(true) that the condition holds in [0.0, 1.0].

    A value near 0.5 indicates high uncertainty between yes and no, whereas
    values near 1.0 or 0.0 indicate high certainty.
    """

    confidence: float = 0.0
    """Summary confidence (defaults to 0.0 as `noul` probability is itself the direct calibrated measure)."""

    action: dict[str, Any] = field(default_factory=dict)
    """Optional structured payload."""


# ============================================================================
# 3. Score Primitive: Ordinal Rubric Evaluation
# ============================================================================


@dataclass(slots=True, frozen=True)
class ScoreQuestion:
    """Ordinal score question on a defined rubric spectrum.

    Score rates an input along a continuous dimension described by an ordered
    progression of discrete levels.
    """

    instructions: str | dict[str, Any] | list[Any]
    """The quality, severity, or dimension being rated (e.g. "How urgent is this request?")."""

    criteria: list[str]
    """An ordered list of level descriptions: `list[str]`.

    - The list represents discrete anchor points along an ordered spectrum,
      from the low end (index 0) to the high end (index N-1).
    - Each entry describes a concrete scenario or benchmark for that level.
    - Note: The model does NOT produce a separate score per criteria item. Instead,
      it evaluates probabilities across all levels and computes a single expected
      score along the spectrum: score = sum(i * P(level_i)).
    """


@dataclass(slots=True, frozen=True)
class ScoreAnswer:
    """Answer returned for an ordinal score question."""

    score: float
    """Continuous position along the rubric levels spectrum in [0.0, N-1].

    Calculated as the probability-weighted sum of level indices:
    score = sum(i * probability_i). Can land between discrete levels.
    """

    confidence: float = 0.0
    """Confidence in the score based on probability dispersion across levels."""

    probabilities: dict[str, float] = field(default_factory=dict)
    """Probability distribution assigned to each discrete level index "0", "1", ..."""

    legend: dict[str, str] = field(default_factory=dict)
    """Mapping of level index strings to their criteria descriptions."""

    action: dict[str, Any] = field(default_factory=dict)
    """Optional structured payload."""


# ============================================================================
# Canonical Type Unions
# ============================================================================

type Question = ChoiceQuestion | NoulQuestion | ScoreQuestion
type Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
]
