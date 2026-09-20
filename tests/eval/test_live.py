"""Live end-to-end evaluation tests against real TypeSafe API."""

from __future__ import annotations

import os

import aiohttp
import pytest
from homeassistant.core import HomeAssistant

from custom_components.typesafe.client import TypeSafeClient
from custom_components.typesafe.engine import TypeSafeDecisionEngine
from custom_components.typesafe.strategy import DecisionStrategy
from tests.common.fixture_loader import load_synthetic_home_fixtures


@pytest.fixture(name="farmhouse_context")
def farmhouse_context_fixture(hass: HomeAssistant):
    """Load the full family farmhouse fixture context."""
    return load_synthetic_home_fixtures(hass)


@pytest.fixture(name="strategy")
def strategy_fixture() -> DecisionStrategy:
    """Fixture providing a default DecisionStrategy."""
    return DecisionStrategy()


@pytest.mark.slow
async def test_live_farmhouse_turn_on_kitchen_light(
    farmhouse_context, strategy: DecisionStrategy
) -> None:
    """Live inference test on farmhouse fixture with real Jev model via TypeSafe API."""
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        pytest.fail("TYPESAFE_API_KEY environment variable is not set")
    assert api_key is not None

    async with aiohttp.ClientSession() as session:
        client = TypeSafeClient(session=session, api_key=api_key, model="jev-latest")
        engine = TypeSafeDecisionEngine(client)
        decision = await strategy.async_decide(
            engine, "Turn on the Kitchen Light", farmhouse_context
        )
        assert not decision.should_escalate
        assert not decision.is_compound
        assert decision.intent_name == "HassTurnOn"
        assert decision.confidence >= 0.50
        is_kitchen_area_light = (
            decision.slots.get("area") == "Kitchen"
            and decision.slots.get("domain") == "light"
        )
        is_kitchen_entity = decision.slots.get("entity_id") == "light.kitchen_light"
        assert is_kitchen_area_light or is_kitchen_entity
