"""Outcome models produced by Stage 5 resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Answer


@dataclass(slots=True)
class Decision:
    """Outcome of a decision resolution evaluation."""

    intent_name: str | None
    entity_id: str | None = None
    area_name: str | None = None
    domain: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    is_compound: bool = False
    should_escalate: bool = False
    escalation_reason: str | None = None
    active_keys: set[str] = field(default_factory=set)
    raw_answers: dict[str, Answer] = field(default_factory=dict)
