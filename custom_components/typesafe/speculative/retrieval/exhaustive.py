"""Exhaustive candidate retrieval implementation."""

from __future__ import annotations

from homeassistant.helpers import intent

from ..context import DecisionContext
from ..request.models import ParsedRequest
from .base import CandidateRetriever
from .heuristics import (
    CANONICAL_INTENT_DESCRIPTIONS,
    CONTROLLABLE_DOMAINS,
    can_fulfill_intent,
)
from .models import (
    AreaCandidate,
    EntityCandidate,
    IntentCandidate,
    RetrievedCandidates,
)


class ExhaustiveCandidateRetriever(CandidateRetriever):
    """Retriever returning all entities and supported intents without filtering."""

    def __init__(self, controllable_only: bool = True) -> None:
        """Initialize ExhaustiveCandidateRetriever."""
        self._controllable_only = controllable_only

    @property
    def controllable_only(self) -> bool:
        """Whether retrieval is restricted to controllable domains."""
        return self._controllable_only

    def retrieve(
        self,
        request: ParsedRequest,
        context: DecisionContext,
    ) -> RetrievedCandidates:
        """Retrieve all entities, areas, and fulfillable intents."""
        # All fulfillable intents
        intents: list[IntentCandidate] = []
        registered_handlers: list[intent.IntentHandler] = []
        if context.hass:
            try:
                registered_handlers = list(intent.async_get(context.hass))
            except (KeyError, AttributeError):
                registered_handlers = []

        if registered_handlers:
            for handler in registered_handlers:
                itype = getattr(handler, "intent_type", None)
                if itype and can_fulfill_intent(handler):
                    desc = CANONICAL_INTENT_DESCRIPTIONS.get(
                        itype, getattr(handler, "description", None)
                    )
                    intents.append(
                        IntentCandidate(intent_type=itype, score=1.0, description=desc)
                    )
        else:
            for itype, desc in CANONICAL_INTENT_DESCRIPTIONS.items():
                intents.append(
                    IntentCandidate(intent_type=itype, score=1.0, description=desc)
                )

        # All areas
        areas: list[AreaCandidate] = []
        if context.area_registry:
            for area in context.area_registry.async_list_areas():
                if area and area.name:
                    area_key = getattr(area, "id", None) or area.name
                    areas.append(
                        AreaCandidate(area_id=area_key, area_name=area.name, score=1.0)
                    )

        # All entities
        entities: list[EntityCandidate] = []
        states = context.states if context.states is not None else []
        for state in states:
            domain = getattr(state, "domain", None)
            if not domain:
                continue
            if self._controllable_only and domain not in CONTROLLABLE_DOMAINS:
                continue

            entity_id = state.entity_id
            friendly_name = (
                state.attributes.get("friendly_name")
                if hasattr(state, "attributes") and isinstance(state.attributes, dict)
                else getattr(state, "name", None)
            ) or entity_id

            area_name: str | None = None
            area_id: str | None = None
            if context.entity_registry and hasattr(
                context.entity_registry, "async_get"
            ):
                entry = context.entity_registry.async_get(entity_id)
                if entry and entry.area_id:
                    area_id = entry.area_id
                    if context.area_registry and hasattr(
                        context.area_registry, "async_get_area"
                    ):
                        area_entry = context.area_registry.async_get_area(entry.area_id)
                        if area_entry:
                            area_name = area_entry.name

            entities.append(
                EntityCandidate(
                    entity_id=entity_id,
                    domain=domain,
                    friendly_name=friendly_name,
                    area_id=area_id,
                    area_name=area_name,
                    score=1.0,
                )
            )

        return RetrievedCandidates(
            intents=intents,
            areas=areas,
            entities=entities,
        )
