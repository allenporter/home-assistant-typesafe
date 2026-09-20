"""Execution context for the speculative decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, entity_registry as er


@dataclass(slots=True)
class DecisionContext:
    """Home Assistant context passed through the decision pipeline.

    device_id: Identifier of the originating satellite/voice device, used for room-aware context.
    """

    hass: HomeAssistant
    area_registry: ar.AreaRegistry
    entity_registry: er.EntityRegistry
    states: list[State] = field(default_factory=list)
    language: str = "en"
    device_id: str | None = None
    originating_area_id: str | None = None
