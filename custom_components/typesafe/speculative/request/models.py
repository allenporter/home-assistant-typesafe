"""Data models for parsed requests in Stage 1."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class ParsedRequest:
    """Normalized, tokenized, and syntactically parsed user request."""

    raw_text: str
    normalized_text: str
    tokens: set[str] = field(default_factory=set)
    raw_numbers: list[float] = field(default_factory=list)
    raw_percentages: list[int] = field(default_factory=list)
    raw_temperatures: list[float] = field(default_factory=list)
    originating_area_id: str | None = None
