"""Deterministic, configurable in-memory decision engine test double."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from ..scoring.engine import DecisionEngine, PredictionResult


class FakeDecisionEngine(DecisionEngine):
    """Deterministic, configurable in-memory decision engine."""

    def __init__(
        self,
        default_answers: dict[str, Answer] | None = None,
        custom_handler: Callable[[Any, Mapping[str, Question]], PredictionResult]
        | None = None,
    ) -> None:
        """Initialize FakeDecisionEngine."""
        self._default_answers = default_answers or {}
        self._custom_handler = custom_handler
        self._queued_results: list[PredictionResult] = []
        self.calls: list[dict[str, Any]] = []

    def queue_result(self, result: PredictionResult) -> None:
        """Queue a prediction result to be returned on the next call."""
        self._queued_results.append(result)

    def set_default_answers(self, answers: dict[str, Answer]) -> None:
        """Set default answers to return when no queued results exist."""
        self._default_answers = answers

    async def async_predict(
        self,
        state: dict[str, Any] | str,
        questions: Mapping[str, Question],
    ) -> PredictionResult:
        """Evaluate questions deterministically."""
        self.calls.append({"state": state, "questions": questions})

        if self._queued_results:
            return self._queued_results.pop(0)

        if self._custom_handler:
            return self._custom_handler(state, questions)

        answers: dict[str, Answer] = dict(self._default_answers)
        for qid, qdef in questions.items():
            if qid in answers:
                continue

            if isinstance(qdef, ChoiceQuestion):
                options = list(qdef.criteria.keys())
                first_opt = options[0] if options else "none"
                answers[qid] = ChoiceAnswer(choice=first_opt, confidence=0.9)
            elif isinstance(qdef, NoulQuestion):
                answers[qid] = NoulAnswer(noul=0.1, confidence=0.9)
            elif isinstance(qdef, ScoreQuestion):
                answers[qid] = ScoreAnswer(score=1.0, confidence=0.9)

        return PredictionResult(
            answers=answers,
            model="fake-decision-engine",
            usage={"input_tokens": 10, "output_tokens": 0},
        )
