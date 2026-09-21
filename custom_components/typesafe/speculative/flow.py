"""DecisionFlow pipeline orchestrator coordinating the 5 decision stages."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

from ..const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_RETRIEVER_TYPE,
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DOMAIN_FILTER_MODE,
    DEFAULT_RETRIEVER_TYPE,
)
from .context import DecisionContext
from .hydration.hydrator import (
    CandidateHydrator,
    HierarchicalCandidateHydrator,
    SimpleCandidateHydrator,
)
from .hydration.models import HydratedPayload
from .request.processor import (
    RequestProcessor,
    SimpleRequestProcessor,
    TokenizingRequestProcessor,
)
from .resolution.models import Decision
from .resolution.resolver import (
    DecisionResolver,
    SimpleDecisionResolver,
    TargetBindingDecisionResolver,
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
        processor: RequestProcessor,
        retriever: CandidateRetriever,
        hydrator: CandidateHydrator,
        scorer: DecisionScorer,
        resolver: DecisionResolver,
    ) -> None:
        """Initialize DecisionFlow with explicit stage implementations."""
        self.processor = processor
        self.retriever = retriever
        self.hydrator = hydrator
        self.scorer = scorer
        self.resolver = resolver

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
    """Create a standard DecisionFlow."""
    return DecisionFlow(
        processor=TokenizingRequestProcessor(),
        retriever=LexicalCandidateRetriever(domain_filter_mode=domain_filter_mode),
        hydrator=HierarchicalCandidateHydrator(),
        scorer=EngineScorer(),
        resolver=TargetBindingDecisionResolver(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
        ),
    )


def create_exhaustive_flow(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    compound_threshold: float = DEFAULT_COMPOUND_THRESHOLD,
) -> DecisionFlow:
    """Create a DecisionFlow that evaluates all controllable candidates without filtering."""
    return DecisionFlow(
        processor=TokenizingRequestProcessor(),
        retriever=ExhaustiveCandidateRetriever(),
        hydrator=HierarchicalCandidateHydrator(),
        scorer=EngineScorer(),
        resolver=TargetBindingDecisionResolver(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
        ),
    )


def create_simple_flow(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> DecisionFlow:
    """Create a minimal, unconstrained pass-through decision flow."""
    return DecisionFlow(
        processor=SimpleRequestProcessor(),
        retriever=ExhaustiveCandidateRetriever(controllable_only=False),
        hydrator=SimpleCandidateHydrator(),
        scorer=EngineScorer(),
        resolver=SimpleDecisionResolver(confidence_threshold=confidence_threshold),
    )


def create_flow_from_options(options: Mapping[str, Any]) -> DecisionFlow:
    """Create a DecisionFlow configured from config entry options."""
    retriever_type = options.get(CONF_RETRIEVER_TYPE, DEFAULT_RETRIEVER_TYPE)
    confidence_threshold = float(
        options.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
    )
    compound_threshold = float(
        options.get(CONF_COMPOUND_THRESHOLD, DEFAULT_COMPOUND_THRESHOLD)
    )

    retriever: CandidateRetriever
    if retriever_type == "exhaustive":
        retriever = ExhaustiveCandidateRetriever()
    else:
        domain_filter_mode = options.get(
            CONF_DOMAIN_FILTER_MODE, DEFAULT_DOMAIN_FILTER_MODE
        )
        retriever = LexicalCandidateRetriever(domain_filter_mode=domain_filter_mode)

    return DecisionFlow(
        processor=TokenizingRequestProcessor(),
        retriever=retriever,
        hydrator=HierarchicalCandidateHydrator(),
        scorer=EngineScorer(),
        resolver=TargetBindingDecisionResolver(
            confidence_threshold=confidence_threshold,
            compound_threshold=compound_threshold,
        ),
    )
