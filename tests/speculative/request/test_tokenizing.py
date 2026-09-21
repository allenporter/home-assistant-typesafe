"""Unit tests for TokenizingRequestProcessor in Stage 1."""

from __future__ import annotations

import pytest

from custom_components.typesafe.speculative.request.tokenizing import (
    TokenizingRequestProcessor,
    tokenize,
)


@pytest.fixture(name="processor")
def processor_fixture() -> TokenizingRequestProcessor:
    """Fixture providing tokenizing request processor."""
    return TokenizingRequestProcessor()


def test_process_basic_utterance(processor: TokenizingRequestProcessor) -> None:
    """Test standard tokenization and normalization of utterance."""
    parsed = processor.process("Turn On The Kitchen Light!")

    assert parsed.normalized_text == "turn on the kitchen light!"
    assert "turn" in parsed.tokens
    assert "kitchen" in parsed.tokens
    assert "light" in parsed.tokens
    assert parsed.raw_percentages == []
    assert parsed.raw_temperatures == []
    assert parsed.originating_area_id is None


def test_process_percentage_extraction(processor: TokenizingRequestProcessor) -> None:
    """Test extracting percentage numbers from utterance."""
    parsed = processor.process("Set kitchen light to 50%")
    assert parsed.raw_percentages == [50]
    assert 50.0 in parsed.raw_numbers

    parsed_multiple = processor.process("from 20% to 80 %")
    assert parsed_multiple.raw_percentages == [20, 80]


def test_process_temperature_extraction(processor: TokenizingRequestProcessor) -> None:
    """Test extracting temperature numbers with degrees/deg/° markers."""
    parsed = processor.process("Set thermostat to 72 degrees")
    assert parsed.raw_temperatures == [72.0]
    assert 72.0 in parsed.raw_numbers

    parsed_deg = processor.process("Set temp to 21.5°")
    assert parsed_deg.raw_temperatures == [21.5]

    parsed_deg_symbol = processor.process("Cool down to 68 deg")
    assert parsed_deg_symbol.raw_temperatures == [68.0]


def test_process_originating_area_id(processor: TokenizingRequestProcessor) -> None:
    """Test originating area ID is preserved in parsed request."""
    parsed = processor.process("Turn on the lights", originating_area_id="kitchen")
    assert parsed.originating_area_id == "kitchen"


def test_tokenize() -> None:
    """Test tokenization extracts lowercase alphanumeric tokens."""
    tokens = tokenize("Turn on the Kitchen Light, please!")
    assert tokens == {"turn", "on", "the", "kitchen", "light", "please"}
