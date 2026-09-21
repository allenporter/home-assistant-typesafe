"""Hermetic unit tests for TypeSafe component lifecycle and setup."""

from __future__ import annotations

from typing import cast

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe import create_flow_from_options
from custom_components.typesafe.const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_RETRIEVER_TYPE,
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from custom_components.typesafe.speculative.resolution.target_binding import (
    TargetBindingDecisionResolver,
)
from custom_components.typesafe.speculative.retrieval.lexical import (
    LexicalCandidateRetriever,
)


async def test_setup_and_unload_entry(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Test successful setup and unload of TypeSafe config entry."""
    assert config_entry.state is ConfigEntryState.LOADED
    assert config_entry.runtime_data is not None
    assert config_entry.runtime_data.client is not None
    assert config_entry.runtime_data.flow is not None

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.NOT_LOADED


async def test_reload_on_options_update(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Test config entry dynamically reloads on options update."""
    assert config_entry.state is ConfigEntryState.LOADED
    assert config_entry.runtime_data.flow.resolver.confidence_threshold == 0.40
    assert config_entry.runtime_data.flow.resolver.compound_threshold == 0.70
    assert config_entry.runtime_data.flow.retriever.domain_filter_mode == "boost"

    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.85,
            CONF_COMPOUND_THRESHOLD: 0.45,
            CONF_DOMAIN_FILTER_MODE: "boost",
        },
    )
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.LOADED
    assert config_entry.runtime_data.flow.resolver.confidence_threshold == 0.85
    assert config_entry.runtime_data.flow.resolver.compound_threshold == 0.45
    assert config_entry.runtime_data.flow.retriever.domain_filter_mode == "boost"

    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_RETRIEVER_TYPE: "exhaustive",
        },
    )
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.LOADED
    assert (
        config_entry.runtime_data.flow.retriever.__class__.__name__
        == "ExhaustiveCandidateRetriever"
    )


def test_create_flow_from_options_default() -> None:
    """Test create_flow_from_options uses default stages and thresholds."""
    flow = create_flow_from_options({})
    resolver = cast(TargetBindingDecisionResolver, flow.resolver)
    retriever = cast(LexicalCandidateRetriever, flow.retriever)
    assert resolver.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert resolver.compound_threshold == DEFAULT_COMPOUND_THRESHOLD
    assert retriever.domain_filter_mode == "boost"


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
