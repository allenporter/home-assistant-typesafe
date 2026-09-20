"""Tests for candidate ranking, discovery, and retrieval recall using synthetic home fixtures."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant

from custom_components.typesafe.speculative.inmemory.engine import FakeDecisionEngine
from custom_components.typesafe.speculative.models import ChoiceAnswer, NoulAnswer
from custom_components.typesafe.speculative.strategy.base import (
    DecisionStrategy,
)
from custom_components.typesafe.speculative.strategy.speculative import (
    DomainBoostedFanOutStrategy,
    IntentPrunedFanOutStrategy,
    SpeculativeFanOutStrategy,
    StandardFanOutStrategy,
)
from tests.common.fixture_loader import (
    load_device_action_cases,
    load_synthetic_home_fixtures,
)


@pytest.fixture(name="farmhouse_context")
def farmhouse_context_fixture(hass: HomeAssistant):
    """Load the full family farmhouse fixture context."""
    return load_synthetic_home_fixtures(hass)


@pytest.fixture(name="strategy")
def strategy_fixture() -> SpeculativeFanOutStrategy:
    """Fixture providing a default DecisionStrategy."""
    return SpeculativeFanOutStrategy()


@pytest.fixture(name="engine")
def engine_fixture() -> FakeDecisionEngine:
    """Fixture providing a configurable FakeDecisionEngine."""
    return FakeDecisionEngine()


async def test_farmhouse_context_loaded(farmhouse_context) -> None:
    """Verify that the farmhouse context has all 12 areas and 29 entity states loaded."""
    assert len(farmhouse_context.area_registry.areas) == 12
    assert len(farmhouse_context.states) == 29
    assert farmhouse_context.home_name == "Family Farmhouse"


async def test_farmhouse_candidate_entity_ranking(
    farmhouse_context,
    strategy: DecisionStrategy,
    engine: FakeDecisionEngine,
) -> None:
    """Test that candidate entity ranking surfaces the correct entity out of 28 entities."""
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.05),
            "target_entity": ChoiceAnswer(
                choice="light.kitchen_light", confidence=0.96
            ),
        }
    )

    await strategy.async_decide(engine, "Turn on the Kitchen Light", farmhouse_context)
    assert len(engine.calls) == 1
    questions = engine.calls[0]["questions"]

    assert "target_entity" in questions
    assert "light.kitchen_light" in questions["target_entity"].criteria

    assert "target_area" in questions
    assert "kitchen" in questions["target_area"].criteria


async def test_farmhouse_porch_light_ranking(
    farmhouse_context,
    strategy: DecisionStrategy,
    engine: FakeDecisionEngine,
) -> None:
    """Test candidate ranking for wrap-around porch devices."""
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.05),
            "target_entity": ChoiceAnswer(choice="light.porch_light", confidence=0.96),
        }
    )

    await strategy.async_decide(engine, "Turn off the Porch Light", farmhouse_context)
    questions = engine.calls[0]["questions"]

    assert "light.porch_light" in questions["target_entity"].criteria
    assert "wrap_around_porch" in questions["target_area"].criteria


async def test_farmhouse_thermostat_intent_discovery(
    farmhouse_context,
    strategy: DecisionStrategy,
    engine: FakeDecisionEngine,
) -> None:
    """Test that supported climate actions are dynamically discovered when thermostat exists."""
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOff", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.05),
            "target_entity": ChoiceAnswer(choice="climate.thermostat", confidence=0.96),
        }
    )

    await strategy.async_decide(engine, "Turn off the thermostat", farmhouse_context)
    questions = engine.calls[0]["questions"]

    assert "HassTurnOff" in questions["intent"].criteria
    assert "climate.thermostat" in questions["target_entity"].criteria


async def test_farmhouse_decision_routing(
    farmhouse_context,
    strategy: DecisionStrategy,
    engine: FakeDecisionEngine,
) -> None:
    """Test end-to-end decision routing with FakeDecisionEngine on a farmhouse utterance."""
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.96),
            "is_compound": NoulAnswer(noul=0.01),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.98),
            "target_entity": ChoiceAnswer(
                choice="light.kitchen_light", confidence=0.98
            ),
        }
    )

    decision = await strategy.async_decide(
        engine, "Turn on the Kitchen Light", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.96
    assert decision.slots == {"entity_id": "light.kitchen_light"}


async def test_farmhouse_labeled_cases_loading() -> None:
    """Test that all supported labeled action cases from family-farmhouse-us are parsed."""
    cases = load_device_action_cases()
    assert len(cases) == 173

    sentences = {c.sentence: c for c in cases}
    assert "Please turn on the Kitchen Light" in sentences
    assert sentences["Please turn on the Kitchen Light"].expected_intent == "HassTurnOn"

    assert "Set the Kitchen Light to 50% brightness" in sentences
    assert (
        sentences["Set the Kitchen Light to 50% brightness"].action == "Set brightness"
    )

    assert "Open the garage door" in sentences
    assert sentences["Open the garage door"].expected_intent == "HassOpenCover"
    assert (
        sentences["Set the Kitchen Light to 50% brightness"].expected_intent
        == "HassLightSet"
    )


async def test_farmhouse_hard_disambiguation_recall(
    farmhouse_context,
    strategy: DecisionStrategy,
    engine: FakeDecisionEngine,
) -> None:
    """Test disambiguation across identical device names using area tokens."""
    # Family room speaker
    await strategy.async_decide(
        engine, "Pause the music in the family room", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "target_area" in questions
    assert "family_room" in questions["target_area"].criteria
    assert "target_entity" in questions
    assert "media_player.smart_speaker" in questions["target_entity"].criteria

    # Master bedroom speaker
    await strategy.async_decide(
        engine, "Turn up the master bedroom speaker", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "target_area" in questions
    assert "master_bedroom" in questions["target_area"].criteria

    # Porch speaker
    await strategy.async_decide(engine, "Resume music on the porch", farmhouse_context)
    questions = engine.calls[-1]["questions"]
    assert "target_area" in questions
    assert "wrap_around_porch" in questions["target_area"].criteria


async def test_farmhouse_valve_and_cover_candidate_recall(
    farmhouse_context,
    strategy: DecisionStrategy,
    engine: FakeDecisionEngine,
) -> None:
    """Test candidate retrieval for valve and cover domains."""
    # Valve: Backyard smart sprinkler system
    await strategy.async_decide(
        engine, "Turn on the backyard sprinklers", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "valve.smart_sprinkler_system" in questions["target_entity"].criteria
    assert "HassTurnOn" in questions["intent"].criteria

    # Cover: Open barn garage door
    await strategy.async_decide(engine, "Open the barn garage door", farmhouse_context)
    questions = engine.calls[-1]["questions"]
    assert "cover.barn_garage_door" in questions["target_entity"].criteria
    assert "HassOpenCover" in questions["intent"].criteria

    # Cover: Close barn garage door
    await strategy.async_decide(engine, "Shut the barn garage door", farmhouse_context)
    questions = engine.calls[-1]["questions"]
    assert "cover.barn_garage_door" in questions["target_entity"].criteria
    assert "HassCloseCover" in questions["intent"].criteria

    # Cover: Stop moving
    await strategy.async_decide(
        engine, "Stop moving the garage door", farmhouse_context
    )
    questions = engine.calls[-1]["questions"]
    assert "cover.barn_garage_door" in questions["target_entity"].criteria
    assert "HassStopMoving" in questions["intent"].criteria


async def test_candidate_ranking_filters_by_intent_domain(
    farmhouse_context,
    engine: FakeDecisionEngine,
) -> None:
    """Verify that IntentPrunedFanOutStrategy prunes non-media entities for 'pause' while StandardFanOutStrategy retains them."""
    # Unpruned strategy includes kitchen light due to area matching
    standard_strategy = StandardFanOutStrategy()
    await standard_strategy.async_decide(engine, "Pause the kitchen", farmhouse_context)
    standard_candidates = engine.calls[-1]["questions"]["target_entity"].criteria
    assert "light.kitchen_light" in standard_candidates

    # Pruned strategy restricts candidates to media_player domain, pruning kitchen light
    pruned_strategy = IntentPrunedFanOutStrategy()
    await pruned_strategy.async_decide(engine, "Pause the kitchen", farmhouse_context)
    pruned_candidates = engine.calls[-1]["questions"]["target_entity"].criteria
    assert "media_player.smart_speaker" in pruned_candidates
    assert "light.kitchen_light" not in pruned_candidates


@pytest.mark.parametrize(
    "strategy_cls",
    [
        StandardFanOutStrategy,
        IntentPrunedFanOutStrategy,
        DomainBoostedFanOutStrategy,
    ],
)
async def test_farmhouse_batch_candidate_recall(
    farmhouse_context,
    strategy_cls: type[DecisionStrategy],
    engine: FakeDecisionEngine,
) -> None:
    """Test candidate intent and entity retrieval recall across all labeled utterances for each strategy."""
    strategy = strategy_cls()
    engine.set_default_answers(
        {
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.05),
        }
    )
    cases = load_device_action_cases()

    intent_hits = 0
    entity_hits = 0

    for c in cases:
        await strategy.async_decide(engine, c.sentence, farmhouse_context)
        questions = engine.calls[-1]["questions"]
        if c.expected_intent in questions["intent"].criteria:
            intent_hits += 1

        if "target_entity" in questions:
            matched_entity = any(
                c.device_name.lower() in desc.lower()
                for desc in questions["target_entity"].criteria.values()
            )
            if matched_entity:
                entity_hits += 1

    intent_recall = intent_hits / len(cases)
    entity_recall = entity_hits / len(cases)
    print(
        f"\n[{strategy_cls.__name__}] Recall - Intent: {intent_recall:.1%}, Entity: {entity_recall:.1%}"
    )

    # Over 90% of utterances should correctly retrieve expected intent in candidate choices
    assert intent_recall > 0.90
    # Over 85% of utterances should correctly retrieve targeted device entity in candidate choices
    assert entity_recall > 0.85
