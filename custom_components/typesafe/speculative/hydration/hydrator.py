"""Candidate hydrator implementations for Stage 3."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
)
from ..retrieval.models import RetrievedCandidates


class CandidateHydrator(ABC):
    """Abstract interface for candidate hydration and question construction."""

    @abstractmethod
    def hydrate(
        self,
        candidates: RetrievedCandidates,
    ) -> dict[str, Question]:
        """Hydrate candidate metadata and construct System One questions."""


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


class HierarchicalCandidateHydrator(CandidateHydrator):
    """Candidate hydrator constructing a hierarchical multi-question schema."""

    def hydrate(
        self,
        candidates: RetrievedCandidates,
    ) -> dict[str, Question]:
        """Hydrate candidates into canonical questions."""
        intent_criteria: dict[str, str | None] = {}
        for c in candidates.intents:
            intent_criteria[c.intent_type] = c.description or f"Handle {c.intent_type}"

        entity_criteria: dict[str, str | None] = {}
        for e in candidates.entities:
            desc = f"{e.friendly_name} ({e.domain})"
            if e.area_name:
                desc += f" in {e.area_name}"
            entity_criteria[e.entity_id] = desc

        area_criteria: dict[str, str | None] = {}
        for a in candidates.areas:
            area_criteria[a.area_id] = f"{a.area_name} area"

        questions: dict[str, Question] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,
            ),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ),
        }

        has_entity_choices = bool(entity_criteria)
        has_area_choices = bool(area_criteria)

        if has_entity_choices:
            questions["target_entity"] = ChoiceQuestion(
                instructions="Which entity is the user referring to?",
                criteria=entity_criteria,
            )

        if has_area_choices:
            questions["target_area"] = ChoiceQuestion(
                instructions="Which area or room is the user referring to?",
                criteria=area_criteria,
            )

        if has_entity_choices and has_area_choices:
            target_type_crit: dict[str, str | None] = {
                "entity": "A specific individual device or appliance",
                "area": "An entire room or area",
            }
            questions["target_type"] = ChoiceQuestion(
                instructions="Is the user targeting an individual device or an entire area?",
                criteria=target_type_crit,
            )

        return questions
