"""Offline retrieval recall evaluation on the family-farmhouse-us synthetic home dataset."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant

from custom_components.typesafe.speculative.context import DecisionContext
from custom_components.typesafe.speculative.request.processor import (
    TokenizingRequestProcessor,
)

from custom_components.typesafe.speculative.retrieval.lexical import (
    LexicalCandidateRetriever,
)
from tests.common.fixture_loader import (
    load_device_action_cases,
    load_synthetic_home_fixtures,
)


@pytest.fixture(name="farmhouse_context")
def farmhouse_context_fixture(hass: HomeAssistant) -> DecisionContext:
    """Load the full family farmhouse fixture context."""
    return load_synthetic_home_fixtures(hass)


async def test_farmhouse_context_loaded(
    farmhouse_context: DecisionContext,
) -> None:
    """Verify that the farmhouse context has all 12 areas and 29 entity states loaded."""
    assert len(farmhouse_context.area_registry.areas) == 12
    assert len(farmhouse_context.states) == 29


def test_farmhouse_labeled_cases_loading() -> None:
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


@pytest.mark.parametrize(
    "domain_filter_mode",
    [
        "none",
        "strict",
        "boost",
    ],
)
async def test_farmhouse_batch_candidate_recall(
    farmhouse_context: DecisionContext,
    domain_filter_mode: str,
) -> None:
    """Test candidate intent and entity retrieval recall across all labeled utterances for each filter mode."""
    processor = TokenizingRequestProcessor()

    retriever = LexicalCandidateRetriever(domain_filter_mode=domain_filter_mode)  # type: ignore[arg-type]
    cases = load_device_action_cases()

    intent_hits = 0
    entity_hits = 0

    for c in cases:
        request = processor.process(c.sentence)
        candidates = retriever.retrieve(request, farmhouse_context)

        candidate_intent_types = {i.intent_type for i in candidates.intents}
        if c.expected_intent in candidate_intent_types:
            intent_hits += 1

        matched_entity = any(
            c.device_name.lower() in (e.friendly_name or "").lower()
            for e in candidates.entities
        )
        if matched_entity:
            entity_hits += 1

    intent_recall = intent_hits / len(cases)
    entity_recall = entity_hits / len(cases)

    # Over 90% of utterances should correctly retrieve expected intent in candidate choices
    assert intent_recall > 0.90
    # Over 85% of utterances should correctly retrieve targeted device entity in candidate choices
    assert entity_recall > 0.85
