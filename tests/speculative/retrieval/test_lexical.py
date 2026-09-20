"""Unit tests for LexicalCandidateRetriever in Stage 2."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.speculative.context import DecisionContext
from custom_components.typesafe.speculative.request.models import ParsedRequest
from custom_components.typesafe.speculative.request.processor import (
    DefaultRequestProcessor,
)
from custom_components.typesafe.speculative.retrieval.lexical import (
    LexicalCandidateRetriever,
)
from tests.common.fixture_loader import load_synthetic_home_fixtures


@pytest.fixture(name="retriever")
def retriever_fixture() -> LexicalCandidateRetriever:
    """Fixture providing a default LexicalCandidateRetriever."""
    return LexicalCandidateRetriever()


@pytest.fixture(name="context")
def context_fixture(hass: HomeAssistant) -> DecisionContext:
    """Fixture providing a real DecisionContext with areas and entities in Home Assistant."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)

    area_kitchen = area_reg.async_create("Kitchen")
    area_bedroom = area_reg.async_create("Bedroom")

    config_entry = MockConfigEntry(domain="test")
    config_entry.add_to_hass(hass)

    # Register entities in registry and state machine
    entry_k = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="kitchen_lights",
        suggested_object_id="kitchen_lights",
    )
    entity_reg.async_update_entity(entry_k.entity_id, area_id=area_kitchen.id)
    hass.states.async_set(
        "light.kitchen_lights", "off", {"friendly_name": "Kitchen Lights"}
    )

    entry_b = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="bedroom_lights",
        suggested_object_id="bedroom_lights",
    )
    entity_reg.async_update_entity(entry_b.entity_id, area_id=area_bedroom.id)
    hass.states.async_set(
        "light.bedroom_lights", "off", {"friendly_name": "Bedroom Lights"}
    )

    entity_reg.async_get_or_create(
        domain="switch",
        platform="test",
        unique_id="attic_fan",
        suggested_object_id="attic_fan",
    )
    hass.states.async_set("switch.attic_fan", "off", {"friendly_name": "Attic Fan"})

    # Non-controllable sensor
    hass.states.async_set(
        "sensor.kitchen_temperature", "72", {"friendly_name": "Kitchen Temperature"}
    )

    return DecisionContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=hass.states.async_all(),
    )


def test_retrieve_entities_and_areas(
    retriever: LexicalCandidateRetriever, context: DecisionContext
) -> None:
    """Test candidate retrieval correctly scores and sorts kitchen light above bedroom light."""
    request = ParsedRequest(
        raw_text="Turn on kitchen light",
        normalized_text="turn on kitchen light",
        tokens={"turn", "on", "kitchen", "light"},
    )

    candidates = retriever.retrieve(request, context)

    # Sensor should be filtered out (not controllable)
    entity_ids = [e.entity_id for e in candidates.entities]
    assert "sensor.kitchen_temperature" not in entity_ids

    # Kitchen light should be top-ranked due to name + area match
    assert len(candidates.entities) >= 2
    assert candidates.entities[0].entity_id == "light.kitchen_lights"
    assert candidates.entities[0].score > candidates.entities[1].score

    # Area candidates
    area_names = [a.area_name for a in candidates.areas]
    assert "Kitchen" in area_names
    assert candidates.areas[0].area_name == "Kitchen"


def test_retrieve_satellite_room_prior_boost(
    retriever: LexicalCandidateRetriever, context: DecisionContext, hass: HomeAssistant
) -> None:
    """Test that originating_area_id provides a location prior boost."""
    area_reg = ar.async_get(hass)
    kitchen_area = area_reg.async_get_area_by_name("Kitchen")
    assert kitchen_area is not None

    request_no_prior = ParsedRequest(
        raw_text="Turn on the lights",
        normalized_text="turn on the lights",
        tokens={"turn", "on", "the", "lights"},
        originating_area_id=None,
    )
    res_no_prior = retriever.retrieve(request_no_prior, context)
    k_cand_no_prior = next(
        e for e in res_no_prior.entities if e.entity_id == "light.kitchen_lights"
    )
    b_cand_no_prior = next(
        e for e in res_no_prior.entities if e.entity_id == "light.bedroom_lights"
    )
    assert k_cand_no_prior.score == b_cand_no_prior.score

    request_with_prior = ParsedRequest(
        raw_text="Turn on the lights",
        normalized_text="turn on the lights",
        tokens={"turn", "on", "the", "lights"},
        originating_area_id=kitchen_area.id,
    )
    res_with_prior = retriever.retrieve(request_with_prior, context)

    # In res_with_prior, kitchen lights should be boosted over bedroom lights
    k_cand_prior = next(
        e for e in res_with_prior.entities if e.entity_id == "light.kitchen_lights"
    )
    b_cand_prior = next(
        e for e in res_with_prior.entities if e.entity_id == "light.bedroom_lights"
    )
    assert k_cand_prior.score > b_cand_prior.score


@pytest.fixture(name="farmhouse_context")
def farmhouse_context_fixture(hass: HomeAssistant) -> DecisionContext:
    """Load the full family farmhouse fixture context."""
    return load_synthetic_home_fixtures(hass)


def test_farmhouse_candidate_entity_ranking(
    farmhouse_context: DecisionContext,
    retriever: LexicalCandidateRetriever,
) -> None:
    """Test that candidate entity ranking surfaces the correct entity out of 28 entities."""
    processor = DefaultRequestProcessor()
    request = processor.process("Turn on the Kitchen Light")
    candidates = retriever.retrieve(request, farmhouse_context)

    entity_ids = [e.entity_id for e in candidates.entities]
    assert "light.kitchen_light" in entity_ids

    area_ids = [a.area_id for a in candidates.areas]
    assert "kitchen" in area_ids


def test_farmhouse_porch_light_ranking(
    farmhouse_context: DecisionContext,
    retriever: LexicalCandidateRetriever,
) -> None:
    """Test candidate ranking for wrap-around porch devices."""
    processor = DefaultRequestProcessor()
    request = processor.process("Turn off the Porch Light")
    candidates = retriever.retrieve(request, farmhouse_context)

    entity_ids = [e.entity_id for e in candidates.entities]
    assert "light.porch_light" in entity_ids

    area_ids = [a.area_id for a in candidates.areas]
    assert "wrap_around_porch" in area_ids


def test_farmhouse_thermostat_intent_discovery(
    farmhouse_context: DecisionContext,
    retriever: LexicalCandidateRetriever,
) -> None:
    """Test that supported climate actions are dynamically discovered when thermostat exists."""
    processor = DefaultRequestProcessor()
    request = processor.process("Turn off the thermostat")
    candidates = retriever.retrieve(request, farmhouse_context)

    intent_types = [i.intent_type for i in candidates.intents]
    assert "HassTurnOff" in intent_types

    entity_ids = [e.entity_id for e in candidates.entities]
    assert "climate.thermostat" in entity_ids


def test_farmhouse_hard_disambiguation_recall(
    farmhouse_context: DecisionContext,
    retriever: LexicalCandidateRetriever,
) -> None:
    """Test disambiguation across identical device names using area tokens."""
    processor = DefaultRequestProcessor()

    # Family room speaker
    req = processor.process("Pause the music in the family room")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "family_room" in [a.area_id for a in candidates.areas]
    assert "media_player.smart_speaker" in [e.entity_id for e in candidates.entities]

    # Master bedroom speaker
    req = processor.process("Turn up the master bedroom speaker")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "master_bedroom" in [a.area_id for a in candidates.areas]

    # Porch speaker
    req = processor.process("Resume music on the porch")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "wrap_around_porch" in [a.area_id for a in candidates.areas]


def test_farmhouse_valve_and_cover_candidate_recall(
    farmhouse_context: DecisionContext,
    retriever: LexicalCandidateRetriever,
) -> None:
    """Test candidate retrieval for valve and cover domains."""
    processor = DefaultRequestProcessor()

    # Valve: Backyard smart sprinkler system
    req = processor.process("Turn on the backyard sprinklers")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "valve.smart_sprinkler_system" in [e.entity_id for e in candidates.entities]
    assert "HassTurnOn" in [i.intent_type for i in candidates.intents]

    # Cover: Open barn garage door
    req = processor.process("Open the barn garage door")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "cover.barn_garage_door" in [e.entity_id for e in candidates.entities]
    assert "HassOpenCover" in [i.intent_type for i in candidates.intents]

    # Cover: Close barn garage door
    req = processor.process("Shut the barn garage door")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "cover.barn_garage_door" in [e.entity_id for e in candidates.entities]
    assert "HassCloseCover" in [i.intent_type for i in candidates.intents]

    # Cover: Stop moving
    req = processor.process("Stop moving the garage door")
    candidates = retriever.retrieve(req, farmhouse_context)
    assert "cover.barn_garage_door" in [e.entity_id for e in candidates.entities]
    assert "HassStopMoving" in [i.intent_type for i in candidates.intents]


def test_candidate_ranking_filters_by_intent_domain(
    farmhouse_context: DecisionContext,
) -> None:
    """Verify that strict domain filtering prunes non-media entities for 'pause' while none retains them."""
    processor = DefaultRequestProcessor()
    req = processor.process("Pause the kitchen")

    # Unpruned retriever includes kitchen light due to area matching
    standard_retriever = LexicalCandidateRetriever(domain_filter_mode="none")
    standard_candidates = [
        e.entity_id
        for e in standard_retriever.retrieve(req, farmhouse_context).entities
    ]
    assert "light.kitchen_light" in standard_candidates

    # Pruned retriever restricts candidates to media_player domain, pruning kitchen light
    pruned_retriever = LexicalCandidateRetriever(domain_filter_mode="strict")
    pruned_candidates = [
        e.entity_id for e in pruned_retriever.retrieve(req, farmhouse_context).entities
    ]
    assert "media_player.smart_speaker" in pruned_candidates
    assert "light.kitchen_light" not in pruned_candidates
