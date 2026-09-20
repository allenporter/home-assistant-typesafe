"""Pytest configuration for evaluation suites.

Composes standard, vendor-agnostic smart home fixtures with TypeSafe/Jev-specific fixtures.
To adapt these evaluation suites for another AI model or provider, replace `fixtures_typesafe`
with the corresponding provider module while preserving `fixtures_standard`.
"""

from __future__ import annotations

from .fixtures_standard import (
    MockClimateIntentHandler,
    MockFallbackAgent,
    climate_handler_fixture as climate_handler,
    device_action_cases_fixture as device_action_cases,
    farmhouse_context_fixture as farmhouse_context,
    standard_intents_fixture as standard_intents,
)
from .fixtures_typesafe import (
    live_typesafe_client_fixture as live_typesafe_client,
    live_typesafe_engine_fixture as live_typesafe_engine,
    mock_typesafe_engine_fixture as mock_typesafe_engine,
    strategy_alias_fixture as strategy,
    typesafe_strategy_fixture as typesafe_strategy,
)

__all__ = [
    "MockClimateIntentHandler",
    "MockFallbackAgent",
    "climate_handler",
    "device_action_cases",
    "farmhouse_context",
    "live_typesafe_client",
    "live_typesafe_engine",
    "mock_typesafe_engine",
    "standard_intents",
    "strategy",
    "typesafe_strategy",
]
