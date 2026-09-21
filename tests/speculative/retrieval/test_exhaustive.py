"""Unit tests for ExhaustiveCandidateRetriever ("return everything" mode)."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
)

from custom_components.typesafe.speculative.request.models import ParsedRequest
from custom_components.typesafe.speculative.retrieval.exhaustive import (
    ExhaustiveCandidateRetriever,
)
from custom_components.typesafe.speculative.context import DecisionContext


@pytest.fixture(name="retriever")
def retriever_fixture() -> ExhaustiveCandidateRetriever:
    """Fixture providing ExhaustiveCandidateRetriever."""
    return ExhaustiveCandidateRetriever()


@pytest.fixture(name="context")
def context_fixture(hass: HomeAssistant) -> DecisionContext:
    """Fixture providing a real DecisionContext."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)

    area_reg.async_create("Area 1")
    area_reg.async_create("Area 2")

    hass.states.async_set("light.light_1", "off", {"friendly_name": "Light 1"})
    hass.states.async_set("switch.switch_1", "off", {"friendly_name": "Switch 1"})
    hass.states.async_set("sensor.sensor_1", "10", {"friendly_name": "Sensor 1"})

    return DecisionContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=hass.states.async_all(),
    )


def test_exhaustive_returns_all_controllable_entities(
    retriever: ExhaustiveCandidateRetriever, context: DecisionContext
) -> None:
    """Test exhaustive retriever returns all controllable entities and areas regardless of lexical tokens."""
    request = ParsedRequest(
        raw_text="Random text unrelated to devices",
        normalized_text="random text unrelated to devices",
        tokens={"random", "text"},
    )

    candidates = retriever.retrieve(request, context)
    entity_ids = [e.entity_id for e in candidates.entities]
    assert "light.light_1" in entity_ids
    assert "switch.switch_1" in entity_ids
    assert "sensor.sensor_1" not in entity_ids
    assert len(candidates.areas) == 2


def test_exhaustive_returns_all_entities_when_not_controllable_only(
    context: DecisionContext,
) -> None:
    """Test exhaustive retriever returns all entities when controllable_only=False."""
    retriever = ExhaustiveCandidateRetriever(controllable_only=False)
    request = ParsedRequest(
        raw_text="Random text unrelated to devices",
        normalized_text="random text unrelated to devices",
        tokens={"random", "text"},
    )

    candidates = retriever.retrieve(request, context)
    entity_ids = [e.entity_id for e in candidates.entities]
    assert "light.light_1" in entity_ids
    assert "switch.switch_1" in entity_ids
    assert "sensor.sensor_1" in entity_ids
    assert len(candidates.areas) == 2
