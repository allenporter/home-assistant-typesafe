"""Test fixture loader for synthetic home datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.speculative.strategy import StrategyContext

DEFAULT_FIXTURES_PATH = (
    Path(__file__).parents[1] / "testdata" / "family-farmhouse-us" / "_fixtures.yaml"
)
DEFAULT_ACTIONS_DIR = Path(__file__).parents[1] / "testdata" / "family-farmhouse-us"

# Standard intent action mapping for synthetic home action files
ACTION_TO_INTENT: dict[str, str] = {
    "Turn on": "HassTurnOn",
    "Turn off": "HassTurnOff",
    "Set brightness": "HassLightSet",
    "Pause": "HassMediaPause",
    "Unpause": "HassMediaUnpause",
    "Next track": "HassMediaNextTrack",
    "Previous track": "HassMediaPreviousTrack",
    "Set volume": "HassSetVolume",
    "Open": "HassOpenCover",
    "Close": "HassCloseCover",
    "Stop": "HassStopMoving",
}


@dataclass(slots=True)
class DeviceActionCase:
    """A test case representing a labeled device action utterance."""

    sentence: str
    action: str
    expected_intent: str
    device_name: str
    area: str
    device_type: str | None = None
    expected_entity_id: str | None = None


class DummyIntentHandler(intent.IntentHandler):
    """Dummy intent handler for tests mimicking Home Assistant registered handlers."""

    def __init__(
        self,
        intent_type: str,
        description: str | None = None,
        platforms: set[str] | None = None,
        supported_slots: set[str] | None = None,
        required_slots: set[str] | None = None,
    ) -> None:
        """Initialize dummy intent handler."""
        self.intent_type = intent_type
        self.description = description
        self.platforms = platforms
        self._supported_slots = (
            supported_slots
            if supported_slots is not None
            else {"name", "area", "floor", "domain"}
        )
        self.required_slots = {s: None for s in (required_slots or ())}
        self.optional_slots = {
            s: None for s in self._supported_slots if s not in self.required_slots
        }

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle intent."""
        res = intent.IntentResponse(language=intent_obj.language)
        res.async_set_speech(f"Handled {self.intent_type}")
        return res


STANDARD_INTENT_CONFIGS: list[
    tuple[str, str, set[str] | None, set[str] | None, set[str] | None]
] = [
    (
        "HassTurnOn",
        "Turn on or activate device, light, or appliance",
        {"light", "switch", "fan", "cover", "media_player", "climate", "valve"},
        {"name", "area", "floor", "domain"},
        None,
    ),
    (
        "HassTurnOff",
        "Turn off or stop device, light, or appliance",
        {"light", "switch", "fan", "cover", "media_player", "climate", "valve"},
        {"name", "area", "floor", "domain"},
        None,
    ),
    (
        "HassToggle",
        "Toggles a device or switch state.",
        {"light", "switch", "fan"},
        {"name", "area", "floor", "domain"},
        None,
    ),
    (
        "HassLightSet",
        "Sets the brightness percentage, dimming, dim, bright, or color of a light.",
        {"light"},
        {"name", "area", "floor", "domain", "brightness", "color"},
        None,
    ),
    (
        "HassMediaPause",
        "Pauses media playback, music, or speaker.",
        {"media_player"},
        {"name", "area", "floor"},
        None,
    ),
    (
        "HassMediaUnpause",
        "Resumes or unpauses media playback or music.",
        {"media_player"},
        {"name", "area", "floor"},
        None,
    ),
    (
        "HassMediaNextTrack",
        "Skips to the next track or song.",
        {"media_player"},
        {"name", "area", "floor"},
        None,
    ),
    (
        "HassMediaPreviousTrack",
        "Goes back to the previous track or song.",
        {"media_player"},
        {"name", "area", "floor"},
        None,
    ),
    (
        "HassSetVolume",
        "Sets the volume percentage or level of a speaker or media player, louder, softer, quieter.",
        {"media_player"},
        {"name", "area", "floor", "volume_level"},
        None,
    ),
    (
        "HassOpenCover",
        "Opens a cover, garage door, blinds, or shades",
        {"cover"},
        {"name", "area", "floor", "domain"},
        None,
    ),
    (
        "HassCloseCover",
        "Closes a cover, garage door, blinds, or shades",
        {"cover"},
        {"name", "area", "floor", "domain"},
        None,
    ),
    (
        "HassStopMoving",
        "Stops movement of a cover, garage door, or shades",
        {"cover"},
        {"name", "area", "floor", "domain"},
        None,
    ),
]


def register_standard_intents(hass: HomeAssistant) -> None:
    """Register standard Home Assistant intents in the intent registry."""
    registered = {h.intent_type for h in intent.async_get(hass)}
    for it, desc, plats, supp_slots, req_slots in STANDARD_INTENT_CONFIGS:
        if it not in registered:
            intent.async_register(
                hass,
                DummyIntentHandler(
                    it,
                    description=desc,
                    platforms=plats,
                    supported_slots=supp_slots,
                    required_slots=req_slots,
                ),
            )


def load_synthetic_home_fixtures(
    hass: HomeAssistant,
    fixtures_path: Path = DEFAULT_FIXTURES_PATH,
) -> StrategyContext:
    """Load a synthetic home _fixtures.yaml into Home Assistant registries and states."""
    data: dict[str, Any] = yaml.safe_load(fixtures_path.read_text(encoding="utf-8"))

    mock_entry = MockConfigEntry(domain="synthetic_home")
    mock_entry.add_to_hass(hass)

    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    device_reg = dr.async_get(hass)

    # Populate Areas
    area_id_map: dict[str, str] = {}
    for area in data.get("areas", []):
        entry = area_reg.async_get_or_create(area["name"])
        area_id_map[area["id"]] = entry.id

    # Populate Devices
    device_id_map: dict[str, str] = {}
    for dev in data.get("devices", []):
        area_key = dev.get("area")
        area_id = area_id_map.get(area_key, area_key) if area_key else None
        dev_entry = device_reg.async_get_or_create(
            config_entry_id=mock_entry.entry_id,
            identifiers={("synthetic_home", dev["id"])},
            name=dev["name"],
        )
        if area_id:
            device_reg.async_update_device(dev_entry.id, area_id=area_id)
        device_id_map[dev["id"]] = dev_entry.id

    # Populate Entities and States
    for ent in data.get("entities", []):
        entity_id = ent["id"]
        domain = entity_id.split(".")[0]
        name = ent.get("name", entity_id)

        area_key = ent.get("area")
        area_id = area_id_map.get(area_key, area_key) if area_key else None

        dev_key = ent.get("device")
        dev_id = device_id_map.get(dev_key) if dev_key else None

        reg_entry = entity_reg.async_get_or_create(
            domain=domain,
            platform="synthetic_home",
            unique_id=entity_id,
            suggested_object_id=entity_id.split(".")[1],
            original_name=name,
            config_entry=mock_entry,
            device_id=dev_id,
        )
        if area_id:
            entity_reg.async_update_entity(reg_entry.entity_id, area_id=area_id)

        raw_state = ent.get("state")
        if raw_state is None:
            state_val = "off"
        elif isinstance(raw_state, bool):
            state_val = "on" if raw_state else "off"
        else:
            state_val = str(raw_state)

        attributes = dict(ent.get("attributes", {}))
        attributes.setdefault("friendly_name", name)

        hass.states.async_set(entity_id, state_val, attributes)

    # Register standard intents
    register_standard_intents(hass)

    return StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=hass.states.async_all(),
        home_name="Family Farmhouse",
        language="en",
    )


def load_device_action_cases(
    actions_dir: Path = DEFAULT_ACTIONS_DIR,
) -> list[DeviceActionCase]:
    """Load all device action utterance test cases from action YAML files."""
    cases: list[DeviceActionCase] = []
    yaml_files = sorted(actions_dir.glob("*.yaml"))

    for f in yaml_files:
        if f.name == "_fixtures.yaml":
            continue
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        device_info = data.get("device", {})
        device_name = device_info.get("name", "")
        area = device_info.get("area", "")
        device_type = device_info.get("device_type")

        raw_actions = data.get("actions", {}).get("actions", [])
        for act in raw_actions:
            action_name = act.get("action", "")
            if action_name not in ACTION_TO_INTENT:
                continue
            expected_intent = ACTION_TO_INTENT[action_name]
            for sentence in act.get("sentences", []):
                cases.append(
                    DeviceActionCase(
                        sentence=sentence,
                        action=action_name,
                        expected_intent=expected_intent,
                        device_name=device_name,
                        area=area,
                        device_type=device_type,
                    )
                )

    return cases
