"""Base interface for Stage 3 candidate hydration."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Question
from ..retrieval.models import RetrievedCandidates


class CandidateHydrator(ABC):
    """Abstract interface for candidate hydration and question construction."""

    @abstractmethod
    def hydrate(
        self,
        candidates: RetrievedCandidates,
    ) -> dict[str, Question]:
        """Hydrate candidate metadata and construct System One questions."""
