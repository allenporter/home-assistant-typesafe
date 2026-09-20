"""DecisionFlow pipeline orchestrator coordinating the 5 decision stages."""

from __future__ import annotations

import logging
from typing import Literal

from .context import DecisionContext
from .hydration.hydrator import CandidateHydrator, DefaultCandidateHydrator
from .hydration.models import HydratedPayload
from .request.processor import DefaultRequestProcessor, RequestProcessor
from .resolution.models import Decision
from .resolution.resolver import (
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DecisionResolver,
    DefaultDecisionResolver,
)
from .retrieval.base import CandidateRetriever
from .retrieval.exhaustive import ExhaustiveCandidateRetriever
from .retrieval.lexical import LexicalCandidateRetriever
from .scoring.engine import DecisionEngine
from .scoring.scorer import DecisionScorer, EngineScorer

_LOGGER = logging.getLogger(__name__)


class DecisionFlow:
    """Orchestrates the 5-stage decision pipeline."""

    def __init__(
        self,
        processor: RequestProcessor | None = None,
        retriever: CandidateRetriever | None = None,
        hydrator: CandidateHydrator | None = None,
        scorer: DecisionScorer | None = None,
        resolver: DecisionResolver | None = None,
    ) -> None:
        """Initialize DecisionFlow with stage implementations."""
        self.processor = processor or DefaultRequestProcessor()
        self.retriever = retriever or LexicalCandidateRetriever()
        self.hydrator = hydrator or DefaultCandidateHydrator()
        self.scorer = scorer or EngineScorer()
        self.resolver = resolver or DefaultDecisionResolver()

    async def async_run(
        self,
        text: str,
        context: DecisionContext,
        engine: DecisionEngine,
    ) -> Decision:
        """Execute the complete decision pipeline across all 5 stages."""
        parsed_request = self.processor.process(
            text, originating_area_id=context.originating_area_id
        )
        candidates = self.retriever.retrieve(parsed_request, context)
        questions = self.hydrator.hydrate(candidates)
        payload = HydratedPayload(
            questions=questions,
            state={"utterance": text},
        )

        try:
            prediction = await self.scorer.score(payload, engine)
        except Exception as err:
            _LOGGER.warning("Decision engine inference failed: %s", err)
            return Decision(
                intent_name=None,
                confidence=0.0,
                should_escalate=True,
                escalation_reason=f"Engine prediction error: {err}",
            )

        return self.resolver.resolve(prediction, parsed_request, candidates)


def create_decision_flow(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
    domain_filter_mode: Literal["none", "strict", "boost"] = "none",
) -> DecisionFlow:
    """Create a configured DecisionFlow."""
    retriever = LexicalCandidateRetriever(domain_filter_mode=domain_filter_mode)
    resolver = DefaultDecisionResolver(
        confidence_threshold=confidence_threshold,
        compound_threshold=compound_threshold,
    )
    return DecisionFlow(
        retriever=retriever,
        resolver=resolver,
    )


def create_exhaustive_flow(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
) -> DecisionFlow:
    """Create a DecisionFlow that evaluates all controllable candidates without filtering."""
    retriever = ExhaustiveCandidateRetriever()
    resolver = DefaultDecisionResolver(
        confidence_threshold=confidence_threshold,
        compound_threshold=compound_threshold,
    )
    return DecisionFlow(
        retriever=retriever,
        resolver=resolver,
    )
