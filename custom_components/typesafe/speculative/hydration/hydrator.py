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


class DefaultCandidateHydrator(CandidateHydrator):
    """Default candidate hydrator constructing canonical System One questions."""

    def __init__(
        self,
        top_n_intents: int = 5,
        top_n_entities: int = 20,
        top_n_areas: int = 10,
    ) -> None:
        """Initialize DefaultCandidateHydrator."""
        self._top_n_intents = top_n_intents
        self._top_n_entities = top_n_entities
        self._top_n_areas = top_n_areas

    def hydrate(
        self,
        candidates: RetrievedCandidates,
    ) -> dict[str, Question]:
        """Hydrate candidates into canonical questions."""
        intent_criteria: dict[str, str | None] = {}
        for c in candidates.intents[: self._top_n_intents]:
            intent_criteria[c.intent_type] = c.description or f"Handle {c.intent_type}"
        intent_criteria["unmatched"] = (
            "Not a home control request or unsupported intent"
        )

        entity_criteria: dict[str, str | None] = {}
        for e in candidates.entities[: self._top_n_entities]:
            desc = f"{e.friendly_name} ({e.domain})"
            if e.area_name:
                desc += f" in {e.area_name}"
            entity_criteria[e.entity_id] = desc
        if entity_criteria:
            entity_criteria["none"] = "None of the listed devices"

        area_criteria: dict[str, str | None] = {}
        for a in candidates.areas[: self._top_n_areas]:
            area_criteria[a.area_id] = f"{a.area_name} area"
        if area_criteria:
            area_criteria["none"] = "None of the listed areas"

        questions: dict[str, Question] = {
            "intent": ChoiceQuestion(
                instructions="Determine the primary Home Assistant action",
                criteria=intent_criteria,
            ),
            "is_compound": NoulQuestion(
                instructions="Does the request contain multiple distinct commands or conjunctions?"
            ),
        }

        has_entity_choices = len(entity_criteria) > 1
        has_area_choices = len(area_criteria) > 1

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
