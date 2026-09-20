"""Hermetic unit tests for TypeSafeClient HTTP implementation."""

from __future__ import annotations


import httpx
import pytest
from respx import MockRouter

from homeassistant.core import HomeAssistant
from homeassistant.helpers import httpx_client
from typesafe_sdk import Choice

from custom_components.typesafe.client import (
    TypeSafeAuthError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafeRateLimitError,
)


@pytest.fixture(name="client")
def client_fixture(hass: HomeAssistant) -> TypeSafeClient:
    """Fixture providing a TypeSafeClient initialized with hass httpx client."""
    http_client = httpx_client.get_async_client(hass)
    return TypeSafeClient(api_key="valid-key", http_client=http_client)


async def test_client_validate_key_success(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test validate_key succeeds on HTTP 200."""
    respx_mock.get("https://api.typesafe.ai/v1/models").respond(
        status_code=200,
        json={
            "models": [
                {
                    "name": "jev-latest",
                    "description": "Jev model",
                    "release_date": "2025-01-01",
                }
            ]
        },
    )
    result = await client.async_validate_key()
    assert result is True


async def test_client_validate_key_unauthorized(
    hass: HomeAssistant,
    respx_mock: MockRouter,
) -> None:
    """Test validate_key raises TypeSafeAuthError on HTTP 401."""
    respx_mock.get("https://api.typesafe.ai/v1/models").respond(
        status_code=401,
        json={"error": "Unauthorized"},
    )
    http_client = httpx_client.get_async_client(hass)
    client = TypeSafeClient(api_key="bad-key", http_client=http_client)
    with pytest.raises(TypeSafeAuthError, match="Invalid API key"):
        await client.async_validate_key()


async def test_client_validate_key_server_error(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test validate_key raises TypeSafeError on HTTP 500."""
    respx_mock.get("https://api.typesafe.ai/v1/models").respond(
        status_code=500,
    )
    with pytest.raises(TypeSafeError, match="Validation failed with status 500"):
        await client.async_validate_key()


async def test_client_evaluate_success(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate succeeds on HTTP 200."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=200,
        json={
            "model": "jev-latest",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": "HassTurnOn",
                    "confidence": 0.95,
                    "probabilities": {"HassTurnOn": 0.95},
                }
            },
        },
    )
    res = await client.async_evaluate(
        state="turn on kitchen light",
        questions={"intent": {"type": "choice", "instructions": "Intent"}},
    )
    assert res["answers"]["intent"]["choice"] == "HassTurnOn"


async def test_client_evaluate_auth_error(
    hass: HomeAssistant,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate raises TypeSafeAuthError on HTTP 401."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=401,
    )
    http_client = httpx_client.get_async_client(hass)
    client = TypeSafeClient(api_key="bad-key", http_client=http_client)
    with pytest.raises(TypeSafeAuthError, match="Invalid API key"):
        await client.async_evaluate(
            state="turn on light",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_evaluate_rate_limit(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate raises TypeSafeRateLimitError on HTTP 429."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=429,
        headers={"Retry-After": "30"},
    )
    with pytest.raises(TypeSafeRateLimitError, match="Rate limited"):
        await client.async_evaluate(
            state="turn on light",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_evaluate_server_error(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate raises TypeSafeError on HTTP 500."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=500,
        text="Internal Server Error",
    )
    with pytest.raises(TypeSafeError, match="failed"):
        await client.async_evaluate(
            state="turn on light",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_evaluate_timeout(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate handles connection / timeout errors."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").mock(
        side_effect=httpx.ConnectTimeout("Timed out")
    )
    with pytest.raises(TypeSafeError, match=r"Request failed|failed"):
        await client.async_evaluate(
            state="turn on light",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_evaluate_422_unprocessable_entity(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate raises TypeSafeError on HTTP 422 Unprocessable Entity."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=422,
        text='{"error": "criteria must have at least 2 choices"}',
    )
    with pytest.raises(TypeSafeError, match="422"):
        await client.async_evaluate(
            state="turn on lights",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_evaluate_529_overloaded(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate raises TypeSafeError on HTTP 529 Overloaded."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=529,
        text='{"error": "Site is temporarily overloaded"}',
    )
    with pytest.raises(TypeSafeError, match="529"):
        await client.async_evaluate(
            state="turn on lights",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_validate_key_422(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test validate_key raises TypeSafeError on HTTP 422."""
    respx_mock.get("https://api.typesafe.ai/v1/models").respond(
        status_code=422,
    )
    with pytest.raises(TypeSafeError, match="422"):
        await client.async_validate_key()


async def test_client_validate_key_529(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test validate_key raises TypeSafeError on HTTP 529."""
    respx_mock.get("https://api.typesafe.ai/v1/models").respond(
        status_code=529,
    )
    with pytest.raises(TypeSafeError, match="529"):
        await client.async_validate_key()


async def test_client_evaluate_rate_limit_without_retry_header(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate handles HTTP 429 when Retry-After header is omitted."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=429,
    )
    with pytest.raises(TypeSafeRateLimitError, match="None"):
        await client.async_evaluate(
            state="turn on lights",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_evaluate_malformed_json_syntax(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test evaluate raises TypeSafeError when response body is not valid JSON."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=200,
        text="<html>Internal Gateway Timeout</html>",
    )
    with pytest.raises(TypeSafeError, match=r"Request failed|failed"):
        await client.async_evaluate(
            state="turn on lights",
            questions={"q": {"type": "choice", "instructions": "test"}},
        )


async def test_client_system_one_success(
    client: TypeSafeClient,
    respx_mock: MockRouter,
) -> None:
    """Test system_one succeeds and returns SystemOneResponse."""
    respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(
        status_code=200,
        json={
            "model": "jev-latest",
            "usage": {"input_tokens": 12, "output_tokens": 4},
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": "HassTurnOn",
                    "confidence": 0.98,
                    "probabilities": {"HassTurnOn": 0.98},
                }
            },
        },
    )
    resp = await client.async_system_one(
        state="turn on light",
        questions={
            "intent": Choice(instructions="Intent", criteria={"HassTurnOn": "Turn on"})
        },
    )
    assert resp.model == "jev-latest"
    assert resp.choices["intent"].choice == "HassTurnOn"
    assert resp.choices["intent"].confidence == 0.98
