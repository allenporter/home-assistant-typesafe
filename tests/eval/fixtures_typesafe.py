"""TypeSafe / Jev-specific evaluation fixtures.

These fixtures provide client and decision engine adapters tailored to
the TypeSafe System One API and the Jev model family.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from typing import cast

import httpx
import pytest

from custom_components.typesafe.client import TypeSafeClient
from custom_components.typesafe.const import DEFAULT_MODEL
from custom_components.typesafe.engine import TypeSafeDecisionEngine
from custom_components.typesafe.speculative.strategy import (
    DecisionStrategy,
    DomainBoostedFanOutStrategy,
    IntentPrunedFanOutStrategy,
    SpeculativeFanOutStrategy,
    StandardFanOutStrategy,
)
from tests.conftest import MockTypeSafeClient


@pytest.fixture(name="typesafe_strategy")
def typesafe_strategy_fixture() -> SpeculativeFanOutStrategy:
    """Fixture providing a default DecisionStrategy for TypeSafe."""
    return SpeculativeFanOutStrategy()


# Alias for convenience in eval tests
@pytest.fixture(name="strategy")
def strategy_alias_fixture(typesafe_strategy: DecisionStrategy) -> DecisionStrategy:
    """Alias for typesafe_strategy fixture."""
    return typesafe_strategy


@pytest.fixture(name="strategy_standard")
def strategy_standard_fixture() -> StandardFanOutStrategy:
    """Fixture providing an unpruned StandardFanOutStrategy."""
    return StandardFanOutStrategy()


@pytest.fixture(name="strategy_pruned")
def strategy_pruned_fixture() -> IntentPrunedFanOutStrategy:
    """Fixture providing an IntentPrunedFanOutStrategy."""
    return IntentPrunedFanOutStrategy()


@pytest.fixture(name="strategy_boosted")
def strategy_boosted_fixture() -> DomainBoostedFanOutStrategy:
    """Fixture providing a DomainBoostedFanOutStrategy."""
    return DomainBoostedFanOutStrategy()


@pytest.fixture(name="mock_typesafe_engine")
def mock_typesafe_engine_fixture(
    mock_client: MockTypeSafeClient,
) -> TypeSafeDecisionEngine:
    """Fixture providing a TypeSafeDecisionEngine initialized with mock_client."""
    return TypeSafeDecisionEngine(cast(TypeSafeClient, mock_client))


@pytest.fixture(name="live_typesafe_client")
async def live_typesafe_client_fixture() -> AsyncGenerator[TypeSafeClient, None]:
    """Provide a live TypeSafeClient configured with TYPESAFE_API_KEY.

    Fails the test loudly if TYPESAFE_API_KEY is not set in the environment.
    """
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        pytest.fail("TYPESAFE_API_KEY environment variable is not set")
    assert api_key is not None

    async with httpx.AsyncClient() as http_client:
        yield TypeSafeClient(
            api_key=api_key,
            http_client=http_client,
            model=DEFAULT_MODEL,
        )


@pytest.fixture(name="live_typesafe_engine")
def live_typesafe_engine_fixture(
    live_typesafe_client: TypeSafeClient,
) -> TypeSafeDecisionEngine:
    """Provide a live TypeSafeDecisionEngine connected to the real TypeSafe API."""
    return TypeSafeDecisionEngine(live_typesafe_client)
