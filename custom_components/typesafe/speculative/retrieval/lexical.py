"""Lexical and heuristic candidate retrieval implementation."""

from __future__ import annotations

from typing import Literal

from homeassistant.helpers import intent

from ..context import DecisionContext
from ..request.models import ParsedRequest
from .base import CandidateRetriever
from .heuristics import (
    CONTROLLABLE_DOMAINS,
    INFORMATIONAL_INTENTS,
    can_fulfill_intent,
    get_allowed_domains_for_intents,
    lexical_score,
)
from .models import (
    AreaCandidate,
    EntityCandidate,
    IntentCandidate,
    RetrievedCandidates,
)


class LexicalCandidateRetriever(CandidateRetriever):
    """Candidate retriever using lexical token matching and room/area boosting."""

    def __init__(
        self,
        max_intents: int = 5,
        max_areas: int = 10,
        max_entities: int = 20,
        domain_filter_mode: Literal["none", "strict", "boost"] = "none",
    ) -> None:
        """Initialize LexicalCandidateRetriever."""
        self._max_intents = max_intents
        self._max_areas = max_areas
        self._max_entities = max_entities
        self._domain_filter_mode = domain_filter_mode

    @property
    def max_intents(self) -> int:
        """Maximum number of intents to retrieve."""
        return self._max_intents

    @property
    def max_areas(self) -> int:
        """Maximum number of areas to retrieve."""
        return self._max_areas

    @property
    def max_entities(self) -> int:
        """Maximum number of entities to retrieve."""
        return self._max_entities

    @property
    def domain_filter_mode(self) -> Literal["none", "strict", "boost"]:
        """Domain filtering mode."""
        return self._domain_filter_mode

    def _retrieve_intents(
        self, request: ParsedRequest, context: DecisionContext
    ) -> list[IntentCandidate]:
        """Discover and rank candidate intent schemas."""
        query_tokens = request.tokens
        full_query = request.normalized_text

        scored_candidates: list[tuple[float, str, str]] = []
        registered_handlers: list[intent.IntentHandler] = []
        if context.hass:
            try:
                registered_handlers = list(intent.async_get(context.hass))
            except (KeyError, AttributeError):
                registered_handlers = []

        for handler in registered_handlers:
            intent_type = getattr(handler, "intent_type", None)
            if (
                not intent_type
                or intent_type in INFORMATIONAL_INTENTS
                or not can_fulfill_intent(handler)
            ):
                continue

            raw_desc = (
                getattr(handler, "description", None)
                or getattr(handler, "__doc__", None)
                or f"Handle {intent_type.replace('Hass', '').strip()}"
            )
            desc = raw_desc.strip()

            desc_score = lexical_score(query_tokens, desc, full_query)
            name_score = lexical_score(
                query_tokens, intent_type.replace("Hass", " "), full_query
            )
            score = desc_score + name_score
            scored_candidates.append((score, intent_type, desc))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        positive_candidates = [c for c in scored_candidates if c[0] > 0]
        selected = (
            positive_candidates[: self._max_intents]
            if positive_candidates
            else scored_candidates[: self._max_intents]
        )

        candidates: list[IntentCandidate] = [
            IntentCandidate(intent_type=name, score=score, description=desc)
            for score, name, desc in selected
        ]

        if not positive_candidates:
            handlers_by_type = {
                getattr(h, "intent_type", None): h for h in registered_handlers
            }
            existing_names = {c.intent_type for c in candidates}
            for itype in ("HassTurnOn", "HassTurnOff"):
                if itype not in existing_names and len(candidates) < self._max_intents:
                    h = handlers_by_type.get(itype)
                    fallback_desc = (
                        getattr(h, "description", None) or getattr(h, "__doc__", None)
                        if h
                        else None
                    ) or f"Handle {itype.replace('Hass', '')}"
                    candidates.append(
                        IntentCandidate(
                            intent_type=itype,
                            score=0.0,
                            description=fallback_desc.strip(),
                        )
                    )

        return candidates

    def _retrieve_areas(
        self, request: ParsedRequest, context: DecisionContext
    ) -> list[AreaCandidate]:
        """Discover and rank candidate areas with satellite room prior."""
        areas = (
            context.area_registry.async_list_areas() if context.area_registry else []
        )
        query_tokens = request.tokens
        full_query = request.normalized_text

        scored_areas: list[tuple[float, str, str]] = []
        for area in areas:
            if not area or not area.name:
                continue
            score = lexical_score(query_tokens, area.name, full_query)
            area_key = getattr(area, "id", None) or getattr(area, "name", "")
            id_score = lexical_score(
                query_tokens, area_key.replace("_", " "), full_query
            )
            total_score = max(score, id_score)

            if (
                request.originating_area_id
                and getattr(area, "id", None) == request.originating_area_id
            ):
                total_score += 2.0

            scored_areas.append((total_score, area_key, area.name))

        scored_areas.sort(key=lambda x: x[0], reverse=True)

        positive = [a for a in scored_areas if a[0] > 0]
        selected = (
            positive[: self._max_areas] if positive else scored_areas[: self._max_areas]
        )

        return [
            AreaCandidate(area_id=key, area_name=name, score=score)
            for score, key, name in selected
        ]

    def _retrieve_entities(
        self,
        request: ParsedRequest,
        context: DecisionContext,
        active_areas: set[str],
        allowed_domains: set[str] | None = None,
        boosted_domains: set[str] | None = None,
    ) -> list[EntityCandidate]:
        """Discover and rank candidate entities with area boosting."""
        query_tokens = request.tokens
        full_query = request.normalized_text

        target_domains = (
            allowed_domains if allowed_domains is not None else CONTROLLABLE_DOMAINS
        )
        scored_entities: list[tuple[float, str, str, str, str | None, str | None]] = []
        states = context.states if context.states is not None else []

        for state in states:
            domain = getattr(state, "domain", None)
            if not domain or domain not in target_domains:
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
                        area = context.area_registry.async_get_area(entry.area_id)
                        if area and area.name:
                            area_name = area.name

            name_score = lexical_score(query_tokens, friendly_name, full_query)
            id_score = lexical_score(
                query_tokens, entity_id.replace("_", " "), full_query
            )
            score = max(name_score, id_score)

            if domain in query_tokens:
                score += 0.5

            if area_name and lexical_score(query_tokens, area_name, full_query) > 0:
                score += 1.0
            elif (
                active_areas
                and (
                    (area_id and area_id in active_areas)
                    or (area_name and area_name in active_areas)
                )
            ) or (
                request.originating_area_id
                and area_id
                and area_id == request.originating_area_id
            ):
                score += 1.0

            if boosted_domains and domain in boosted_domains:
                score += 1.5

            scored_entities.append(
                (score, entity_id, domain, friendly_name, area_id, area_name)
            )

        scored_entities.sort(key=lambda x: x[0], reverse=True)

        selected = scored_entities[: self._max_entities]
        return [
            EntityCandidate(
                entity_id=eid,
                domain=dom,
                friendly_name=name,
                area_id=aid,
                area_name=aname,
                score=score,
            )
            for score, eid, dom, name, aid, aname in selected
        ]

    def retrieve(
        self,
        request: ParsedRequest,
        context: DecisionContext,
    ) -> RetrievedCandidates:
        """Retrieve candidates across intents, areas, and entities."""
        intents = self._retrieve_intents(request, context)
        areas = self._retrieve_areas(request, context)

        active_area_keys = {a.area_id for a in areas if a.score > 0} | {
            a.area_name for a in areas if a.score > 0
        }

        allowed_domains: set[str] | None = None
        boosted_domains: set[str] | None = None

        if self._domain_filter_mode in ("strict", "boost") and intents:
            cand_intent_names = {i.intent_type for i in intents}
            handlers_map: dict[str, intent.IntentHandler] = {}
            if context.hass:
                try:
                    handlers_map = {
                        getattr(h, "intent_type", ""): h
                        for h in intent.async_get(context.hass)
                        if hasattr(h, "intent_type")
                    }
                except (KeyError, AttributeError):
                    handlers_map = {}

            intent_domains = get_allowed_domains_for_intents(
                cand_intent_names, handlers_map
            )
            if intent_domains:
                if self._domain_filter_mode == "strict":
                    allowed_domains = intent_domains
                elif self._domain_filter_mode == "boost":
                    boosted_domains = intent_domains

        entities = self._retrieve_entities(
            request,
            context,
            active_areas=active_area_keys,
            allowed_domains=allowed_domains,
            boosted_domains=boosted_domains,
        )

        return RetrievedCandidates(
            intents=intents,
            areas=areas,
            entities=entities,
        )
