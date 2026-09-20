"""Hermetic unit tests for TypeSafeClient HTTP implementation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.typesafe.client import (
    TypeSafeAuthError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeRateLimitError,
)


async def test_client_validate_key_success(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test validate_key succeeds on HTTP 200."""
    aioclient_mock.get(
        "https://api.typesafe.ai/v1/models",
        status=200,
        json={"models": [{"name": "jev-latest"}]},
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="valid-key")
    result = await client.async_validate_key()
    assert result is True


async def test_client_validate_key_unauthorized(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test validate_key raises TypeSafeAuthError on HTTP 401."""
    aioclient_mock.get(
        "https://api.typesafe.ai/v1/models",
        status=401,
        json={"error": "Unauthorized"},
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="bad-key")
    with pytest.raises(TypeSafeAuthError, match="Invalid API key"):
        await client.async_validate_key()


async def test_client_validate_key_server_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test validate_key raises TypeSafeError on HTTP 500."""
    aioclient_mock.get(
        "https://api.typesafe.ai/v1/models",
        status=500,
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="valid-key")
    with pytest.raises(TypeSafeError, match="Validation failed with status 500"):
        await client.async_validate_key()


async def test_client_evaluate_success(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate succeeds on HTTP 200."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=200,
        json={
            "model": "jev-latest",
            "answers": {
                "intent": {"type": "choice", "choice": "HassTurnOn", "confidence": 0.95}
            },
        },
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="valid-key")
    res = await client.async_evaluate(
        state="turn on kitchen light",
        questions={"intent": {"type": "choice", "instructions": "Intent"}},
    )
    assert res["answers"]["intent"]["choice"] == "HassTurnOn"


async def test_client_evaluate_auth_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate raises TypeSafeAuthError on HTTP 401."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=401,
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="bad-key")
    with pytest.raises(TypeSafeAuthError, match="Invalid API key"):
        await client.async_evaluate(
            state="turn on light",
            questions={},
        )


async def test_client_evaluate_rate_limit(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate raises TypeSafeRateLimitError on HTTP 429."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=429,
        headers={"Retry-After": "30"},
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="valid-key")
    with pytest.raises(TypeSafeRateLimitError, match="Rate limited"):
        await client.async_evaluate(
            state="turn on light",
            questions={},
        )


async def test_client_evaluate_server_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Test evaluate raises TypeSafeError on HTTP 500."""
    aioclient_mock.post(
        "https://api.typesafe.ai/v1/systemone",
        status=500,
        text="Internal Server Error",
    )
    session = aiohttp_client.async_get_clientsession(hass)
    client = TypeSafeClient(session=session, api_key="valid-key")
    with pytest.raises(TypeSafeError, match="failed"):
        await client.async_evaluate(
            state="turn on light",
            questions={},
        )


async def test_client_evaluate_timeout() -> None:
    """Test evaluate handles connection / timeout errors."""
    session = MagicMock()
    session.post.side_effect = asyncio.TimeoutError("Timed out")
    client = TypeSafeClient(session=session, api_key="valid-key")
    with pytest.raises(TypeSafeError, match="Request failed"):
        await client.async_evaluate(
            state="turn on light",
            questions={},
        )
