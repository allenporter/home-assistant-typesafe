"""TypeSafe API client using Home Assistant's aiohttp session."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .const import API_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT


class TypeSafeError(HomeAssistantError):
    """Base exception for TypeSafe API errors."""


class TypeSafeAuthError(TypeSafeError):
    """Authentication failed (401)."""


class TypeSafeRateLimitError(TypeSafeError):
    """Rate limit exceeded (429)."""


class TypeSafeClient:
    """Asynchronous client for TypeSafe System One API."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = API_BASE_URL,
    ) -> None:
        """Initialize the client."""
        self._session = session
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    @property
    def model(self) -> str:
        """Return the configured model."""
        return self._model

    @property
    def api_key(self) -> str:
        """Return the API key."""
        return self._api_key

    async def async_validate_key(self) -> bool:
        """Validate the API key by querying GET /v1/models."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        url = f"{self._base_url}/v1/models"
        try:
            async with self._session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status == 401:
                    raise TypeSafeAuthError("Invalid API key")
                if response.status != 200:
                    raise TypeSafeError(
                        f"Validation failed with status {response.status}"
                    )
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            if isinstance(err, TypeSafeError):
                raise
            raise TypeSafeError(f"Connection failed: {err}") from err

    async def async_evaluate(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Send evaluation request to System One (POST /v1/systemone)."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "state": state,
            "model": model or self._model,
            "questions": questions,
        }
        url = f"{self._base_url}/v1/systemone"
        try:
            async with self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status == 401:
                    raise TypeSafeAuthError("Invalid API key")
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise TypeSafeRateLimitError(
                        f"Rate limited. Retry after: {retry_after}"
                    )
                if response.status != 200:
                    text = await response.text()
                    raise TypeSafeError(
                        f"System One evaluation failed ({response.status}): {text}"
                    )
                data = await response.json()
                if not isinstance(data, dict):
                    raise TypeSafeError(
                        f"Invalid response from TypeSafe: expected dict, got {type(data).__name__}"
                    )
                return data
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            if isinstance(err, TypeSafeError):
                raise
            raise TypeSafeError(f"Request failed: {err}") from err
