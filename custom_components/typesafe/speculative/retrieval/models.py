"""Data models for candidate retrieval in Stage 2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class IntentCandidate:
    """A candidate Home Assistant intent action."""

    intent_type: str
    score: float
    description: str | None = None


@dataclass(slots=True, frozen=True)
class AreaCandidate:
    """A candidate Home Assistant area or room."""

    area_id: str
    area_name: str
    score: float


@dataclass(slots=True, frozen=True)
class EntityCandidate:
    """A candidate Home Assistant smart device entity."""

    entity_id: str
    domain: str
    friendly_name: str
    area_id: str | None = None
    area_name: str | None = None
    score: float = 0.0


@dataclass(slots=True, frozen=True)
class RetrievedCandidates:
    """Ranked collection of candidate intents, areas, and entities."""

    intents: list[IntentCandidate] = field(default_factory=list)
    areas: list[AreaCandidate] = field(default_factory=list)
    entities: list[EntityCandidate] = field(default_factory=list)
