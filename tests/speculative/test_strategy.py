"""Unit tests for speculative intent decision strategy pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.typesafe.speculative.inmemory.engine import FakeDecisionEngine
from custom_components.typesafe.speculative.models import (
    ChoiceAnswer,
    NoulAnswer,
)
from custom_components.typesafe.speculative.strategy.base import StrategyContext
from custom_components.typesafe.speculative.strategy.speculative import (
    DomainBoostedFanOutStrategy,
    IntentPrunedFanOutStrategy,
    SpeculativeFanOutStrategy,
    StandardFanOutStrategy,
)


@pytest.fixture(name="strategy")
def strategy_fixture() -> SpeculativeFanOutStrategy:
    """Fixture providing a default SpeculativeFanOutStrategy."""
    return SpeculativeFanOutStrategy(confidence_threshold=0.7)


@pytest.fixture(name="empty_context")
def empty_context_fixture() -> MagicMock:
    """Fixture providing an empty StrategyContext mock."""
    context = MagicMock(spec=StrategyContext)
    context.home_name = "My Home"
    context.area_registry.async_list_areas.return_value = []
    context.states = []
    return context


def test_strategy_variations_domain_filter_modes() -> None:
    """Test that strategy variations initialize with their respective domain filter modes."""
    standard = StandardFanOutStrategy()
    assert standard.domain_filter_mode == "none"

    pruned = IntentPrunedFanOutStrategy()
    assert pruned.domain_filter_mode == "strict"

    boosted = DomainBoostedFanOutStrategy()
    assert boosted.domain_filter_mode == "boost"


async def test_speculative_fan_out_strategy_decision(
    strategy: SpeculativeFanOutStrategy,
    empty_context: MagicMock,
) -> None:
    """Test end-to-end decision evaluation via SpeculativeFanOutStrategy with FakeDecisionEngine."""
    fake_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_entity": ChoiceAnswer(choice="light.kitchen", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = await strategy.async_decide(
        fake_engine, "Turn on the kitchen light to 50%", empty_context
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.95
    assert decision.slots.get("entity_id") == "light.kitchen"
    assert decision.slots.get("brightness") == 50


async def test_speculative_fan_out_compound_and_low_confidence(
    empty_context: MagicMock,
) -> None:
    """Test compound command escalation and low confidence handling."""
    strategy = SpeculativeFanOutStrategy(
        confidence_threshold=0.8, compound_threshold=0.5
    )

    # Compound escalation
    compound_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.8),
        }
    )

    compound_dec = await strategy.async_decide(
        compound_engine, "Turn on light and play music", empty_context
    )
    assert compound_dec.should_escalate
    assert compound_dec.is_compound
    assert compound_dec.escalation_reason == "Compound command detected"

    # Low confidence escalation
    low_conf_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.6),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )
    low_conf_dec = await strategy.async_decide(
        low_conf_engine, "Turn on light", empty_context
    )
    assert low_conf_dec.should_escalate
    assert low_conf_dec.escalation_reason == "Unhandled intent or low confidence"


async def test_strategy_continuous_slots_extraction(
    strategy: SpeculativeFanOutStrategy,
    empty_context: MagicMock,
) -> None:
    """Test regex extraction of brightness and temperature continuous slots."""
    fake_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassClimateSetTemperature", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )
    decision = await strategy.async_decide(
        fake_engine, "Set thermostat to 72.5 deg and lights to 80%", empty_context
    )
    assert decision.slots.get("temperature") == 72.5
    assert decision.slots.get("brightness") == 80


async def test_strategy_engine_prediction_exception(
    strategy: SpeculativeFanOutStrategy,
    empty_context: MagicMock,
) -> None:
    """Test that an unhandled engine exception escalates gracefully."""
    failing_engine = MagicMock()
    failing_engine.async_predict.side_effect = RuntimeError("API connection timeout")

    decision = await strategy.async_decide(
        failing_engine, "Turn on the light", empty_context
    )
    assert decision.should_escalate
    assert "API connection timeout" in (decision.escalation_reason or "")
