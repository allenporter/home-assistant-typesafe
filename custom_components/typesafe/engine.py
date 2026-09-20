"""TypeSafe API DecisionEngine adapter."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from .client import TypeSafeClient
from .speculative.engine import DecisionEngine, PredictionResult
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
        questions: Mapping[str, Question | dict[str, Any]],
    ) -> PredictionResult:
        """Translate questions, evaluate via TypeSafeClient, and return PredictionResult."""
        serialized_questions: dict[str, dict[str, Any]] = {}
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                serialized_questions[qid] = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            elif isinstance(q, NoulQuestion):
                serialized_questions[qid] = {
                    "type": "noul",
                    "instructions": q.instructions,
                }
            elif isinstance(q, ScoreQuestion):
                serialized_questions[qid] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            elif isinstance(q, dict):
                serialized_questions[qid] = q

        raw_response = await self._client.async_evaluate(
            state=state, questions=serialized_questions
        )

        raw_answers = raw_response.get("answers", {})
        answers: dict[str, Answer] = {}

        for qid, ans_data in raw_answers.items():
            if not isinstance(ans_data, dict):
                continue
            qtype = ans_data.get("type")
            if not qtype:
                if "choice" in ans_data:
                    qtype = "choice"
                elif "noul" in ans_data:
                    qtype = "noul"
                elif "score" in ans_data:
                    qtype = "score"
                elif qid in questions:
                    q = questions[qid]
                    if isinstance(q, ChoiceQuestion):
                        qtype = "choice"
                    elif isinstance(q, NoulQuestion):
                        qtype = "noul"
                    elif isinstance(q, ScoreQuestion):
                        qtype = "score"

            conf = float(ans_data.get("confidence", 0.0))
            action = ans_data.get("action", {})

            if qtype == "choice":
                raw_choice = ans_data.get("choice")
                choice_val = "" if raw_choice is None else str(raw_choice)
                answers[qid] = ChoiceAnswer(
                    choice=choice_val,
                    confidence=conf,
                    probabilities={
                        str(k): float(v)
                        for k, v in ans_data.get("probabilities", {}).items()
                    },
                    action=action if isinstance(action, dict) else {},
                )
            elif qtype == "noul":
                answers[qid] = NoulAnswer(
                    noul=float(ans_data.get("noul", 0.0)),
                    confidence=conf,
                    action=action if isinstance(action, dict) else {},
                )
            elif qtype == "score":
                answers[qid] = ScoreAnswer(
                    score=float(ans_data.get("score", 0.0)),
                    confidence=conf,
                    probabilities={
                        str(k): float(v)
                        for k, v in ans_data.get("probabilities", {}).items()
                    },
                    legend={
                        str(k): str(v) for k, v in ans_data.get("legend", {}).items()
                    },
                    action=action if isinstance(action, dict) else {},
                )

        return PredictionResult(
            answers=answers,
            model=raw_response.get("model"),
            usage=raw_response.get("usage", {}),
        )
