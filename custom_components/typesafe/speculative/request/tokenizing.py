"""Tokenizing and syntactic slot extracting request processor for Stage 1."""

from __future__ import annotations

import re

from .base import RequestProcessor
from .models import ParsedRequest


def tokenize(text: str) -> set[str]:
    """Tokenize a string into a set of unique lowercase alphanumeric words."""
    return set(re.findall(r"\b\w+\b", text.lower()))


class TokenizingRequestProcessor(RequestProcessor):
    """Request processor performing tokenization, normalization, and syntactic slot extraction."""

    def process(
        self, text: str, originating_area_id: str | None = None
    ) -> ParsedRequest:
        """Normalize text, extract tokens, numbers, percentages, and room prior."""
        raw_text = text
        normalized_text = text.strip().lower()
        tokens = tokenize(text)

        raw_percentages: list[int] = []
        for match in re.finditer(r"(\d+)\s*%", text):
            raw_percentages.append(int(match.group(1)))

        raw_temperatures: list[float] = []
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(?:degrees|deg|°)", text, re.IGNORECASE
        ):
            raw_temperatures.append(float(match.group(1)))

        raw_numbers: list[float] = []
        for match in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text):
            raw_numbers.append(float(match.group(1)))

        return ParsedRequest(
            raw_text=raw_text,
            normalized_text=normalized_text,
            tokens=tokens,
            raw_numbers=raw_numbers,
            raw_percentages=raw_percentages,
            raw_temperatures=raw_temperatures,
            originating_area_id=originating_area_id,
        )
