"""Hermetic unit tests for TypeSafe component lifecycle and setup."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.const import CONF_CONFIDENCE_THRESHOLD


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

    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.85,
        },
    )
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.LOADED
    assert config_entry.runtime_data.flow.resolver.confidence_threshold == 0.85
