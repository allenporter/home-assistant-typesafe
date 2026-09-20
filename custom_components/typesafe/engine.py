"""TypeSafe API DecisionEngine adapter."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from typesafe_sdk import (
    Choice,
    Noul,
    Score,
)

from .client import TypeSafeClient
from .speculative.scoring.engine import DecisionEngine, PredictionResult
from .speculative.models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

_LOGGER = logging.getLogger(__name__)


class TypeSafeDecisionEngine(DecisionEngine):
    """Adapter connecting TypeSafeClient to the speculative DecisionEngine protocol."""

    def __init__(self, client: TypeSafeClient) -> None:
        """Initialize TypeSafeDecisionEngine."""
        self._client = client

    @property
    def client(self) -> TypeSafeClient:
        """Return the underlying TypeSafeClient."""
        return self._client

    async def async_predict(
        self,
        state: dict[str, Any] | str,
        questions: Mapping[str, Question | Choice | Noul | Score],
    ) -> PredictionResult:
        """Translate canonical questions, evaluate via TypeSafeClient, and return PredictionResult."""
        sdk_questions: dict[str, Choice | Noul | Score] = {}

        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                sdk_questions[qid] = Choice(
                    instructions=q.instructions,
                    criteria=q.criteria,
                )
            elif isinstance(q, NoulQuestion):
                sdk_questions[qid] = Noul(
                    instructions=q.instructions,
                )
            elif isinstance(q, ScoreQuestion):
                sdk_questions[qid] = Score(
                    instructions=q.instructions,
                    criteria=q.criteria,
                )
            elif isinstance(q, (Choice, Noul, Score)):
                sdk_questions[qid] = q

        raw_response = await self._client.async_system_one(
            state=state, questions=sdk_questions
        )

        answers: dict[str, Answer] = {}

        for qid, c_ans in raw_response.choices.items():
            choice_val = "" if c_ans.choice is None else str(c_ans.choice)
            answers[qid] = ChoiceAnswer(
                choice=choice_val,
                confidence=float(c_ans.confidence),
                probabilities={
                    str(k): float(v) for k, v in (c_ans.probabilities or {}).items()
                },
                action={},
            )
        for qid, n_ans in raw_response.nouls.items():
            answers[qid] = NoulAnswer(
                noul=float(n_ans.noul),
                confidence=0.0,
                action={},
            )
        for qid, s_ans in raw_response.scores.items():
            answers[qid] = ScoreAnswer(
                score=float(s_ans.score),
                confidence=float(s_ans.confidence),
                probabilities={
                    str(k): float(v) for k, v in (s_ans.probabilities or {}).items()
                },
                legend={str(k): str(v) for k, v in (s_ans.legend or {}).items()},
                action={},
            )

        usage = (
            raw_response.usage.model_dump() if raw_response.usage is not None else {}
        )
        return PredictionResult(
            answers=answers,
            model=raw_response.model,
            usage=usage,
        )
