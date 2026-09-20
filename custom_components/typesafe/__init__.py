"""TypeSafe custom component."""

from __future__ import annotations

import logging

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import httpx_client

from .client import TypeSafeClient
from .const import (
    CONF_API_KEY,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
)
from .engine import TypeSafeDecisionEngine
from .models import TypeSafeConfigEntry, TypeSafeData
from .speculative.flow import create_decision_flow

_LOGGER = logging.getLogger(__name__)

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION,)


async def async_setup_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Set up a config entry."""
    api_key = entry.data[CONF_API_KEY]
    model = entry.data.get(CONF_MODEL, DEFAULT_MODEL)
    confidence_threshold = float(
        entry.options.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
    )

    http_client = httpx_client.get_async_client(hass)
    client = TypeSafeClient(api_key=api_key, http_client=http_client, model=model)
    engine = TypeSafeDecisionEngine(client=client)
    flow = create_decision_flow(confidence_threshold=confidence_threshold)

    entry.runtime_data = TypeSafeData(client=client, engine=engine, flow=flow)

    await hass.config_entries.async_forward_entry_setups(
        entry,
        platforms=PLATFORMS,
    )

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(
        entry,
        PLATFORMS,
    )


async def async_reload_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> None:
    """Reload config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
