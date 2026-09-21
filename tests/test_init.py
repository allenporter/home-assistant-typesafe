"""Hermetic unit tests for TypeSafe component lifecycle and setup."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_RETRIEVER_TYPE,
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
    assert config_entry.runtime_data.flow.resolver.confidence_threshold == 0.7
    assert config_entry.runtime_data.flow.resolver.compound_threshold == 0.5
    assert config_entry.runtime_data.flow.retriever.domain_filter_mode == "none"

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
