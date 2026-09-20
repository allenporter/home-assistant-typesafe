"""Live model evaluation tests on the family-farmhouse-us synthetic home dataset."""

from __future__ import annotations

import pytest

from custom_components.typesafe.engine import TypeSafeDecisionEngine
from custom_components.typesafe.speculative.strategy import (
    SpeculativeFanOutStrategy as DecisionStrategy,
    StrategyContext,
)

pytestmark = pytest.mark.slow


async def test_live_farmhouse_turn_on_kitchen_light(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: turn on light command routes to kitchen light entity or area."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine, "Turn on the Kitchen Light", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence >= 0.30

    is_kitchen_area_light = (
        decision.slots.get("area") or ""
    ).lower() == "kitchen" and decision.slots.get("domain") == "light"
    is_kitchen_entity = decision.slots.get("entity_id") == "light.kitchen_light"
    assert (
        is_kitchen_area_light or is_kitchen_entity
    ), f"Unexpected slots: {decision.slots}"


async def test_live_farmhouse_turn_off_porch_light(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: turn off light command routes to porch light."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine, "Turn off the Porch Light", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOff"
    assert decision.confidence >= 0.30

    is_porch_entity = decision.slots.get("entity_id") == "light.porch_light"
    is_porch_area = (decision.slots.get("area") or "").lower() in (
        "porch",
        "wrap_around_porch",
        "wrap-around porch",
    )
    assert is_porch_entity or is_porch_area, f"Unexpected slots: {decision.slots}"


async def test_live_farmhouse_dim_kitchen_light(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: brightness percentage slot extracted alongside device targeting."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine,
        "Set the Kitchen Light to 50% brightness",
        farmhouse_context,
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.slots.get("brightness") == 50

    is_kitchen_entity = decision.slots.get("entity_id") == "light.kitchen_light"
    is_kitchen_area = (decision.slots.get("area") or "").lower() == "kitchen"
    assert is_kitchen_entity or is_kitchen_area, f"Unexpected slots: {decision.slots}"


async def test_live_farmhouse_compound_escalation(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: multi-action compound command triggers escalation."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine,
        "Turn on the kitchen light and turn off the porch light",
        farmhouse_context,
    )

    assert decision.should_escalate
    assert decision.is_compound or decision.escalation_reason is not None


async def test_live_farmhouse_out_of_domain_escalation(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: general conversational query not matching home control escalates."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine, "What is the capital of France?", farmhouse_context
    )

    assert decision.should_escalate


async def test_live_farmhouse_valve_turn_on(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: turn on sprinklers command routes to backyard sprinkler valve."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine, "Turn on the backyard sprinklers", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence >= 0.30

    is_sprinkler_entity = (
        decision.slots.get("entity_id") == "valve.smart_sprinkler_system"
    )
    is_backyard_area = (decision.slots.get("area") or "").lower() == "backyard"
    assert (
        is_sprinkler_entity or is_backyard_area
    ), f"Unexpected slots: {decision.slots}"


async def test_live_farmhouse_media_player_pause(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: pause command routes to family room speaker."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine, "Pause the family room speaker", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassMediaPause"
    assert decision.confidence >= 0.30

    is_family_room = (decision.slots.get("area") or "").lower() in (
        "family room",
        "family_room",
    )
    is_speaker_entity = decision.slots.get("entity_id") == "media_player.smart_speaker"
    assert is_family_room or is_speaker_entity, f"Unexpected slots: {decision.slots}"


async def test_live_farmhouse_cover_open(
    farmhouse_context: StrategyContext,
    live_typesafe_engine: TypeSafeDecisionEngine,
    typesafe_strategy: DecisionStrategy,
) -> None:
    """Live inference test: open garage door command routes to barn garage door cover."""
    decision = await typesafe_strategy.async_decide(
        live_typesafe_engine, "Open the barn garage door", farmhouse_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name in ("HassOpenCover", "HassTurnOn")
    assert decision.confidence >= 0.30

    is_garage_entity = decision.slots.get("entity_id") == "cover.barn_garage_door"
    is_barn_area = (decision.slots.get("area") or "").lower() == "barn"
    assert is_garage_entity or is_barn_area, f"Unexpected slots: {decision.slots}"
