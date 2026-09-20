"""Data models for candidate hydration and question construction in Stage 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Question


@dataclass(slots=True, frozen=True)
class HydratedPayload:
    """State payload and canonical questions prepared for decision engine scoring."""

    questions: dict[str, Question] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
