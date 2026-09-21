"""DecisionFlow pipeline orchestrator coordinating the 5 decision stages."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

from .context import DecisionContext
from .hydration.base import CandidateHydrator
from .hydration.hierarchical import HierarchicalCandidateHydrator
from .hydration.models import HydratedPayload
from .request.base import RequestProcessor
from .request.tokenizing import TokenizingRequestProcessor
from .resolution.base import DecisionResolver
from .resolution.models import Decision
from .resolution.target_binding import TargetBindingDecisionResolver

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


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Configuration options for constructing a DecisionFlow."""

    confidence_threshold: float = 0.40
    compound_threshold: float = 0.70
    retriever_type: Literal["lexical", "exhaustive"] = "lexical"
    domain_filter_mode: Literal["none", "strict", "boost"] = "boost"


def create_decision_flow(config: FlowConfig) -> DecisionFlow:
    """Create a configured DecisionFlow."""
    retriever: CandidateRetriever
    if config.retriever_type == "exhaustive":
        retriever = ExhaustiveCandidateRetriever()
    else:
        retriever = LexicalCandidateRetriever(
            domain_filter_mode=config.domain_filter_mode
        )

    return DecisionFlow(
        processor=TokenizingRequestProcessor(),
        retriever=retriever,
        hydrator=HierarchicalCandidateHydrator(),
        scorer=EngineScorer(),
        resolver=TargetBindingDecisionResolver(
            confidence_threshold=config.confidence_threshold,
            compound_threshold=config.compound_threshold,
        ),
    )
