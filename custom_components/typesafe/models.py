"""Models for the TypeSafe integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .client import TypeSafeClient
from .speculative.flow import DecisionFlow
from .speculative.scoring.engine import DecisionEngine

type TypeSafeConfigEntry = ConfigEntry[TypeSafeData]


@dataclass(slots=True)
class TypeSafeData:
    """Runtime data stored in ConfigEntry."""

    client: TypeSafeClient
    engine: DecisionEngine
    flow: DecisionFlow
