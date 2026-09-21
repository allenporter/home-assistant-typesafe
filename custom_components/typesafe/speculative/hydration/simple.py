"""Simple pass-through candidate hydrator for Stage 3."""

from __future__ import annotations

from ..models import ChoiceQuestion, Question
from ..retrieval.models import RetrievedCandidates
from .base import CandidateHydrator


class SimpleCandidateHydrator(CandidateHydrator):
    """Simple candidate hydrator constructing a single intent classification question."""

    def hydrate(
        self,
        candidates: RetrievedCandidates,
    ) -> dict[str, Question]:
        """Hydrate candidate intents into an intent question."""
        intent_criteria: dict[str, str | None] = {}
        for c in candidates.intents:
            intent_criteria[c.intent_type] = c.description or f"Handle {c.intent_type}"

        return {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,
            )
        }
