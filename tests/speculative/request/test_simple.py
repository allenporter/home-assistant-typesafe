"""Unit tests for SimpleRequestProcessor in Stage 1."""

from __future__ import annotations

from custom_components.typesafe.speculative.request.simple import SimpleRequestProcessor


def test_simple_request_processor() -> None:
    """Test SimpleRequestProcessor normalizes and tokenizes without regex extraction."""
    simple = SimpleRequestProcessor()
    parsed = simple.process(
        "Set kitchen light to 50% and 72 deg", originating_area_id="kitchen"
    )

    assert parsed.raw_text == "Set kitchen light to 50% and 72 deg"
    assert parsed.normalized_text == "set kitchen light to 50% and 72 deg"
    assert "kitchen" in parsed.tokens
    assert "light" in parsed.tokens
    assert parsed.raw_percentages == []
    assert parsed.raw_temperatures == []
    assert parsed.raw_numbers == []
    assert parsed.originating_area_id == "kitchen"
