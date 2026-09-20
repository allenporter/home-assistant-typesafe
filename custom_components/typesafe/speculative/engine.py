"""Abstract base interfaces for System One decision engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .models import Answer, Question


@dataclass(slots=True)
class PredictionResult:
    """Result of an engine prediction over a batch of questions."""

    answers: dict[str, Answer]
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class DecisionEngine(ABC):
    """Abstract interface for a System One decision engine.

    Evaluates a batch of typed questions over state in a single, parallel forward pass.
    """

    @property
    def loaded(self) -> bool:
        """Return whether the engine model is currently ready/loaded."""
        return True

    @abstractmethod
    async def async_predict(
        self,
        state: dict[str, Any] | str,
        questions: Mapping[str, Question],
    ) -> PredictionResult:
        """Evaluate a batch of questions over state in a single forward pass."""

    async def async_load(self) -> None:
        """Eagerly load model or initialize resources."""

    async def async_unload(self) -> None:
        """Release any held resources/models."""
