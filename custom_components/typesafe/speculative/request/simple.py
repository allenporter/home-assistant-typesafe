"""Simple pass-through request processor for Stage 1."""

from __future__ import annotations

from .base import RequestProcessor
from .models import ParsedRequest
from .tokenizing import tokenize


class SimpleRequestProcessor(RequestProcessor):
    """Simple pass-through request processor performing basic tokenization and normalization."""

    def process(
        self, text: str, originating_area_id: str | None = None
    ) -> ParsedRequest:
        """Normalize text and extract lowercase word tokens without regex extraction."""
        return ParsedRequest(
            raw_text=text,
            normalized_text=text.strip().lower(),
            tokens=tokenize(text),
            originating_area_id=originating_area_id,
        )
