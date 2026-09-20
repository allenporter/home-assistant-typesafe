"""Models for the TypeSafe integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .client import TypeSafeClient
from .strategy import DecisionStrategy

type TypeSafeConfigEntry = ConfigEntry[TypeSafeData]


@dataclass(slots=True)
class TypeSafeData:
    """Runtime data stored in ConfigEntry."""

    client: TypeSafeClient
    strategy: DecisionStrategy
