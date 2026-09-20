"""Simple, deterministic strategy test double."""

from __future__ import annotations

from typing import Any

from ..engine import DecisionEngine
from ..strategy.base import Decision, DecisionStrategy, StrategyContext


class SimpleStrategy(DecisionStrategy):
    """Configurable deterministic strategy for conversation agent and lifecycle testing."""

    def __init__(
        self,
        intent_name: str | None = "HassTurnOn",
        slots: dict[str, Any] | None = None,
        confidence: float = 0.95,
        is_compound: bool = False,
        should_escalate: bool = False,
        escalation_reason: str | None = None,
    ) -> None:
        """Initialize SimpleStrategy with preset outcome."""
        self.intent_name = intent_name
        self.slots = slots or {}
        self.confidence = confidence
        self.is_compound = is_compound
        self.should_escalate = should_escalate
        self.escalation_reason = escalation_reason
        self.invocations: list[dict[str, Any]] = []

    async def async_decide(
        self,
        engine: DecisionEngine,
        text: str,
        context: StrategyContext,
    ) -> Decision:
        """Return preset decision."""
        self.invocations.append({"text": text, "context": context, "engine": engine})
        return Decision(
            intent_name=self.intent_name,
            slots=dict(self.slots),
            confidence=self.confidence,
            is_compound=self.is_compound,
            should_escalate=self.should_escalate,
            escalation_reason=self.escalation_reason,
            raw_answers={},
        )
