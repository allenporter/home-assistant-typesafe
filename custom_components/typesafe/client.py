"""TypeSafe API client using official typesafe-sdk backed by Home Assistant's httpx client."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import httpx
from homeassistant.exceptions import HomeAssistantError
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceModel,
    JSONContent,
    Noul,
    NoulModel,
    Score,
    ScoreModel,
    SystemOneResponse,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeError as SDKTypeSafeError,
    TypeSafeRateLimitError as SDKTypeSafeRateLimitError,
)

from .const import API_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT


class TypeSafeError(HomeAssistantError):
    """Base exception for TypeSafe API errors."""


class TypeSafeAuthError(TypeSafeError):
    """Authentication failed (401)."""


class TypeSafeRateLimitError(TypeSafeError):
    """Rate limit exceeded (429)."""


class TypeSafeClient:
    """Asynchronous client for TypeSafe System One API wrapping typesafe-sdk."""

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = API_BASE_URL,
    ) -> None:
        """Initialize the client."""
        self._model = model
        self._timeout = float(timeout)
        self._sdk_client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            timeout=self._timeout,
            http_client=cast(Any, http_client),
            base_url=base_url.rstrip("/"),
        )

    async def async_validate_key(self) -> bool:
        """Validate the API key by querying GET /v1/models."""
        try:
            await self._sdk_client.models.list()
            return True
        except TypeSafeAuthenticationError as err:
            raise TypeSafeAuthError(f"Invalid API key: {err}") from err
        except SDKTypeSafeRateLimitError as err:
            retry_after = err.headers.get("retry-after")
            raise TypeSafeRateLimitError(
                f"Rate limited. Retry after: {retry_after}"
            ) from err
        except TypeSafeAPIError as err:
            raise TypeSafeError(
                f"Validation failed with status {err.status}: {err}"
            ) from err
        except (SDKTypeSafeError, httpx.HTTPError, asyncio.TimeoutError) as err:
            raise TypeSafeError(f"Validation failed: {err}") from err

    async def async_system_one(
        self,
        state: JSONContent,
        questions: Mapping[
            str, Noul | Choice | Score | NoulModel | ChoiceModel | ScoreModel
        ],
        model: str | None = None,
    ) -> SystemOneResponse:
        """Send evaluation request to System One."""
        try:
            return await self._sdk_client.system_one(
                state=state,
                questions=questions,
                model=model or self._model,
            )
        except TypeSafeAuthenticationError as err:
            raise TypeSafeAuthError(f"Invalid API key: {err}") from err
        except SDKTypeSafeRateLimitError as err:
            retry_after = err.headers.get("retry-after")
            raise TypeSafeRateLimitError(
                f"Rate limited. Retry after: {retry_after}"
            ) from err
        except TypeSafeAPIError as err:
            raise TypeSafeError(
                f"System One evaluation failed ({err.status}): {err}"
            ) from err
        except (
            SDKTypeSafeError,
            httpx.HTTPError,
            asyncio.TimeoutError,
            ValueError,
        ) as err:
            raise TypeSafeError(f"Request failed: {err}") from err

    async def async_evaluate(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Any] | Mapping[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Send evaluation request to System One and return raw dictionary format."""
        sdk_questions: dict[
            str, Noul | Choice | Score | NoulModel | ChoiceModel | ScoreModel
        ] = {}
        for qid, q in questions.items():
            if isinstance(q, (Choice, Noul, Score)):
                sdk_questions[qid] = q
            elif isinstance(q, dict):
                qtype = q.get("type")
                if qtype == "choice":
                    sdk_questions[qid] = Choice(
                        instructions=q.get("instructions", ""),
                        criteria=q.get("criteria", {}),
                    )
                elif qtype == "noul":
                    sdk_questions[qid] = Noul(
                        instructions=q.get("instructions", ""),
                        criteria=q.get("criteria"),
                    )
                elif qtype == "score":
                    sdk_questions[qid] = Score(
                        instructions=q.get("instructions", ""),
                        criteria=q.get("criteria", []),
                    )
                else:
                    sdk_questions[qid] = Choice(
                        instructions=q.get("instructions", ""),
                        criteria=q.get("criteria", {}),
                    )
            else:
                sdk_questions[qid] = q

        resp = await self.async_system_one(
            state=state, questions=sdk_questions, model=model
        )

        answers: dict[str, Any] = {}
        for qid, c_ans in resp.choices.items():
            answers[qid] = {
                "type": "choice",
                "choice": c_ans.choice,
                "confidence": c_ans.confidence,
                "probabilities": (
                    dict(c_ans.probabilities) if c_ans.probabilities else {}
                ),
            }
        for qid, n_ans in resp.nouls.items():
            answers[qid] = {
                "type": "noul",
                "noul": n_ans.noul,
            }
        for qid, s_ans in resp.scores.items():
            answers[qid] = {
                "type": "score",
                "score": s_ans.score,
                "confidence": s_ans.confidence,
                "legend": dict(s_ans.legend) if s_ans.legend else {},
                "probabilities": (
                    dict(s_ans.probabilities) if s_ans.probabilities else {}
                ),
            }

        usage = resp.usage.model_dump() if resp.usage is not None else {}
        return {
            "model": resp.model,
            "answers": answers,
            "usage": usage,
        }
