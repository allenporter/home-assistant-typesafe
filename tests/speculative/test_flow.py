"""Unit tests for the end-to-end DecisionFlow pipeline."""

from __future__ import annotations

from typing import cast

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_RETRIEVER_TYPE,
)
from custom_components.typesafe.speculative.context import DecisionContext
from custom_components.typesafe.speculative.flow import (
    DecisionFlow,
    create_decision_flow,
    create_exhaustive_flow,
    create_flow_from_options,
    create_simple_flow,
)
from custom_components.typesafe.speculative.hydration.hydrator import (
    HierarchicalCandidateHydrator,
)
from custom_components.typesafe.speculative.models import (
    ChoiceAnswer,
    NoulAnswer,
)
from custom_components.typesafe.speculative.request.processor import (
    TokenizingRequestProcessor,
)
from custom_components.typesafe.speculative.resolution.resolver import (
    TargetBindingDecisionResolver,
)
from custom_components.typesafe.speculative.retrieval.exhaustive import (
    ExhaustiveCandidateRetriever,
)
from custom_components.typesafe.speculative.retrieval.lexical import (
    LexicalCandidateRetriever,
)
from custom_components.typesafe.speculative.scoring.scorer import EngineScorer
from custom_components.typesafe.speculative.testing.engine import FakeDecisionEngine
from tests.common.fixture_loader import load_synthetic_home_fixtures


class FailingDecisionEngine(FakeDecisionEngine):
    """Fake decision engine that raises an error during inference."""

    async def async_predict(self, state, questions):
        raise RuntimeError("API connection timeout")


@pytest.fixture(name="context")
def context_fixture(hass: HomeAssistant) -> DecisionContext:
    """Fixture providing a DecisionContext with real Home Assistant registries and entities."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)

    kitchen = area_reg.async_create("Kitchen")
    config_entry = MockConfigEntry(domain="test")
    config_entry.add_to_hass(hass)

    entry = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="kitchen_light",
        suggested_object_id="kitchen_light",
    )
    entity_reg.async_update_entity(entry.entity_id, area_id=kitchen.id)
    hass.states.async_set(
        "light.kitchen_light", "off", {"friendly_name": "Kitchen Light"}
    )

    return DecisionContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=hass.states.async_all(),
    )


@pytest.fixture(name="engine")
def engine_fixture() -> FakeDecisionEngine:
    """Fixture providing a FakeDecisionEngine with configured responses."""
    return FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.9),
            "target_entity": ChoiceAnswer(choice="light.kitchen_light", confidence=0.9),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )


async def test_flow_runs_all_five_stages(
    context: DecisionContext, engine: FakeDecisionEngine
) -> None:
    """Test standard DecisionFlow pipeline end-to-end."""
    flow = create_decision_flow()

    decision = await flow.async_run(
        text="Turn on the kitchen light to 50%",
        context=context,
        engine=engine,
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.slots["entity_id"] == "light.kitchen_light"
    assert decision.slots["brightness"] == 50


async def test_flow_custom_stage_injection(
    context: DecisionContext, engine: FakeDecisionEngine
) -> None:
    """Test swapping a stage (e.g. using ExhaustiveCandidateRetriever)."""
    exhaustive_retriever = ExhaustiveCandidateRetriever()
    flow = DecisionFlow(
        processor=TokenizingRequestProcessor(),
        retriever=exhaustive_retriever,
        hydrator=HierarchicalCandidateHydrator(),
        scorer=EngineScorer(),
        resolver=TargetBindingDecisionResolver(),
    )

    decision = await flow.async_run(
        text="Turn on the kitchen light",
        context=context,
        engine=engine,
    )

    assert not decision.should_escalate
    assert decision.intent_name == "HassTurnOn"
    assert decision.slots["entity_id"] == "light.kitchen_light"


def test_flow_factory_configurations() -> None:
    """Test creating decision flows with various configurations."""
    standard_flow = create_decision_flow(domain_filter_mode="none")
    assert (
        cast(LexicalCandidateRetriever, standard_flow.retriever).domain_filter_mode
        == "none"
    )

    pruned_flow = create_decision_flow(domain_filter_mode="strict")
    assert (
        cast(LexicalCandidateRetriever, pruned_flow.retriever).domain_filter_mode
        == "strict"
    )

    boosted_flow = create_decision_flow(domain_filter_mode="boost")
    assert (
        cast(LexicalCandidateRetriever, boosted_flow.retriever).domain_filter_mode
        == "boost"
    )

    exhaustive_flow = create_exhaustive_flow()
    assert type(exhaustive_flow.retriever) is ExhaustiveCandidateRetriever


def test_create_flow_from_options_default() -> None:
    """Test create_flow_from_options uses default stages and thresholds."""
    flow = create_flow_from_options({})
    resolver = cast(TargetBindingDecisionResolver, flow.resolver)
    retriever = cast(LexicalCandidateRetriever, flow.retriever)
    assert resolver.confidence_threshold == 0.7
    assert resolver.compound_threshold == 0.5
    assert retriever.domain_filter_mode == "none"


def test_create_flow_from_options_custom_parameters() -> None:
    """Test create_flow_from_options passes custom filter mode and thresholds."""
    flow = create_flow_from_options(
        {
            CONF_CONFIDENCE_THRESHOLD: 0.85,
            CONF_COMPOUND_THRESHOLD: 0.45,
            CONF_DOMAIN_FILTER_MODE: "boost",
        }
    )
    resolver = cast(TargetBindingDecisionResolver, flow.resolver)
    retriever = cast(LexicalCandidateRetriever, flow.retriever)
    assert resolver.confidence_threshold == 0.85
    assert resolver.compound_threshold == 0.45
    assert retriever.domain_filter_mode == "boost"


def test_create_flow_from_options_exhaustive() -> None:
    """Test create_flow_from_options selects exhaustive candidate retriever."""
    flow = create_flow_from_options(
        {
            CONF_RETRIEVER_TYPE: "exhaustive",
            CONF_CONFIDENCE_THRESHOLD: 0.9,
            CONF_COMPOUND_THRESHOLD: 0.2,
        }
    )
    resolver = cast(TargetBindingDecisionResolver, flow.resolver)
    assert resolver.confidence_threshold == 0.9
    assert resolver.compound_threshold == 0.2
    assert flow.retriever.__class__.__name__ == "ExhaustiveCandidateRetriever"


async def test_flow_compound_and_low_confidence_escalation(
    context: DecisionContext,
) -> None:
    """Test compound command escalation and low confidence handling in flow."""
    flow = create_decision_flow(confidence_threshold=0.8, compound_threshold=0.5)

    compound_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.8),
        }
    )

    compound_dec = await flow.async_run(
        text="Turn on light and play music",
        context=context,
        engine=compound_engine,
    )
    assert compound_dec.should_escalate
    assert compound_dec.is_compound
    assert compound_dec.escalation_reason == "Compound command detected"

    low_conf_engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.6),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )
    low_conf_dec = await flow.async_run(
        text="Turn on light",
        context=context,
        engine=low_conf_engine,
    )
    assert low_conf_dec.should_escalate
    assert low_conf_dec.escalation_reason == "Unhandled intent or low confidence"


async def test_flow_continuous_slots_extraction(
    context: DecisionContext,
) -> None:
    """Test continuous slots extraction for brightness and temperature."""
    flow = create_decision_flow()
    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassClimateSetTemperature", confidence=0.95),
            "is_compound": NoulAnswer(noul=0.01),
        }
    )

    decision = await flow.async_run(
        text="Set thermostat to 72.5 deg and lights to 80%",
        context=context,
        engine=engine,
    )
    assert decision.slots.get("temperature") == 72.5
    assert decision.slots.get("brightness") == 80


async def test_flow_engine_prediction_exception(
    context: DecisionContext,
) -> None:
    """Test that an unhandled engine exception escalates gracefully."""
    flow = create_decision_flow()
    failing_engine = FailingDecisionEngine()

    decision = await flow.async_run(
        text="Turn on the light",
        context=context,
        engine=failing_engine,
    )
    assert decision.should_escalate
    assert "API connection timeout" in (decision.escalation_reason or "")


async def test_farmhouse_decision_routing(hass: HomeAssistant) -> None:
    """Test end-to-end decision routing with FakeDecisionEngine on a farmhouse utterance."""
    farmhouse_context = load_synthetic_home_fixtures(hass)
    flow = create_decision_flow()

    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.96),
            "is_compound": NoulAnswer(noul=0.01),
            "target_type": ChoiceAnswer(choice="entity", confidence=0.98),
            "target_entity": ChoiceAnswer(
                choice="light.kitchen_light", confidence=0.98
            ),
        }
    )

    decision = await flow.async_run(
        text="Turn on the Kitchen Light",
        context=farmhouse_context,
        engine=engine,
    )

    assert not decision.should_escalate
    assert not decision.is_compound
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.96
    assert decision.slots == {"entity_id": "light.kitchen_light"}


async def test_simple_flow_end_to_end(
    context: DecisionContext,
) -> None:
    """Test create_simple_flow runs simple pass-through stages end-to-end."""
    flow = create_simple_flow(confidence_threshold=0.8)
    engine = FakeDecisionEngine(
        default_answers={
            "intent": ChoiceAnswer(choice="HassTurnOn", confidence=0.92),
        }
    )

    decision = await flow.async_run(
        text="Turn on something",
        context=context,
        engine=engine,
    )

    assert not decision.should_escalate
    assert decision.intent_name == "HassTurnOn"
    assert decision.confidence == 0.92
    assert decision.slots == {}
