"""Base interfaces and data structures for decision strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, entity_registry as er

from ..engine import DecisionEngine
from ..models import Answer


@dataclass(slots=True)
class StrategyContext:
    """Home Assistant context passed to decision strategies.

    device_id: Identifier of the originating satellite/voice device, used for room-aware context.
    """

    hass: HomeAssistant
    area_registry: ar.AreaRegistry
    entity_registry: er.EntityRegistry
    states: list[State] = field(default_factory=list)
    home_name: str = "Home"
    language: str = "en"
    device_id: str | None = None


@dataclass(slots=True)
class Decision:
    """Outcome of a decision strategy evaluation."""

    intent_name: str | None
    entity_id: str | None = None
    area_name: str | None = None
    domain: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    is_compound: bool = False
    should_escalate: bool = False
    escalation_reason: str | None = None
    active_keys: set[str] = field(default_factory=set)
    raw_answers: dict[str, Answer] = field(default_factory=dict)


class DecisionStrategy(ABC):
    """Abstract base class for intent resolution and question routing strategies."""

    @abstractmethod
    async def async_decide(
        self,
        engine: DecisionEngine,
        text: str,
        context: StrategyContext,
    ) -> Decision:
        """Evaluate text utterance against engine and produce an actionable Decision."""
