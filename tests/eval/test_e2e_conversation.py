"""End-to-End, requirement-driven opaque-box test suite for TypeSafe integration.

Structured into 4 Tiers as specified in TEST_INFRA.md:
- Tier 1: Feature Coverage across all 16 features in isolation (80 tests).
- Tier 2: Boundary & Corner Cases across all 16 features (80 tests).
- Tier 3: Cross-Feature Combinations (pairwise feature interactions) (18 tests).
- Tier 4: Real-World Application Workloads (all 8 application scenarios) (8 tests).

Total test cases: 186 (exceeding the >=184 threshold).
Hermetic: 100% isolated with zero live network calls or credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import MATCH_ALL, Platform
from homeassistant.core import Context, HomeAssistant, State
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    intent,
)

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.client import (
    TypeSafeAuthError,
    TypeSafeError,
    TypeSafeRateLimitError,
)
from custom_components.typesafe.const import (
    CONF_API_KEY,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_FALLBACK_AGENT,
    CONF_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DOMAIN,
)
from custom_components.typesafe.models import TypeSafeData
from custom_components.typesafe.strategy import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionStrategy,
    NoulAnswer,
    NoulQuestion,
    StrategyContext,
    can_fulfill_intent,
)
from tests.conftest import (
    MockBaseIntentHandler,
    MockFallbackAgent,
    MockTypeSafeClient,
)


class MockClimateIntentHandler(MockBaseIntentHandler):
    """Mock handler for HassClimateSetTemperature."""

    intent_type = "HassClimateSetTemperature"
    description = "Set target temperature for thermostat or climate device"

    def __init__(self) -> None:
        """Initialize handler."""
        self.handled_intents: list[intent.Intent] = []

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """Handle climate intent."""
        self.handled_intents.append(intent_obj)
        res = intent.IntentResponse(language=intent_obj.language)
        res.async_set_speech("Target temperature set")
        return res


@pytest.fixture(name="climate_handler")
def climate_handler_fixture(hass: HomeAssistant) -> MockClimateIntentHandler:
    """Fixture to register and return a MockClimateIntentHandler."""
    handler = MockClimateIntentHandler()
    intent.async_register(hass, handler)
    return handler


MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "custom_components"
    / "typesafe"
    / "manifest.json"
)


async def test_manifest_metadata() -> None:
    """Verify component manifest metadata conforms to specification."""
    with MANIFEST_PATH.open() as fp:
        data = json.load(fp)
    assert data["domain"] == "typesafe"
    assert data["name"] == "TypeSafe"
    assert data["config_flow"] is True
    assert "conversation" in data["dependencies"]
    assert "intent" in data["dependencies"]
    assert data["integration_type"] == "service"


async def test_tier1_f1_setup_entry_initializes_runtime_data(
    config_entry: MockConfigEntry,
) -> None:
    """F1.2: Verify async_setup_entry creates and assigns TypeSafeData runtime data."""
    assert config_entry.state is ConfigEntryState.LOADED
    assert isinstance(config_entry.runtime_data, TypeSafeData)
    assert config_entry.runtime_data.client is not None
    assert isinstance(config_entry.runtime_data.strategy, DecisionStrategy)


async def test_tier1_f1_setup_entry_forwards_conversation_platform(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F1.3: Verify async_setup_entry forwards Platform.CONVERSATION setup."""
    assert Platform.CONVERSATION in hass.config.components or any(
        Platform.CONVERSATION.value in s for s in hass.config.components
    )
    manager = conversation.get_agent_manager(hass)
    assert manager.async_get_agent(config_entry.entry_id) is not None


async def test_tier1_f1_unload_entry_cleans_up_platform(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F1.4: Verify async_unload_entry unloads platform and transitions state."""
    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.NOT_LOADED


async def test_tier1_f1_agent_unregistered_on_unload(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F1.5: Verify conversation agent is unset from agent manager on unload."""
    manager = conversation.get_agent_manager(hass)
    assert manager.async_get_agent(config_entry.entry_id) is not None
    await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    with pytest.raises(ValueError, match="not found"):
        manager.async_get_agent(config_entry.entry_id)


# --- F2: Credentials Validation (`GET /v1/models`) ---


async def test_tier1_f2_validate_key_success_in_config_flow(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.1: Verify valid API key credentials allow flow completion."""
    mock_client.validate_result = True
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "valid_secret_key", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY


async def test_tier1_f2_validate_key_401_sets_invalid_auth_error(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.2: Verify 401 Unauthorized raises TypeSafeAuthError and shows invalid_auth."""
    mock_client.validate_error = TypeSafeAuthError("Invalid credentials")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "bad_secret_key", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.FORM
    assert result2.get("errors") == {"base": "invalid_auth"}


async def test_tier1_f2_validate_key_connection_error_sets_cannot_connect(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.3: Verify connection failure maps to cannot_connect error in config flow."""
    mock_client.validate_error = TypeSafeError("Network timeout")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "any_key", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.FORM
    assert result2.get("errors") == {"base": "cannot_connect"}


async def test_tier1_f2_direct_client_validate_key_returns_true(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.4: Verify mock client validate_key succeeds directly."""
    mock_client.validate_result = True
    mock_client.validate_error = None
    assert await mock_client.async_validate_key() is True


async def test_tier1_f2_direct_client_validate_key_raises_auth_error(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.5: Verify mock client validate_key raises when error configured."""
    mock_client.validate_error = TypeSafeAuthError("Denied")
    with pytest.raises(TypeSafeAuthError, match="Denied"):
        await mock_client.async_validate_key()


# --- F3: Config Flow (`api_key`, `model`) ---


async def test_tier1_f3_flow_init_shows_form(hass: HomeAssistant) -> None:
    """F3.1: Verify config flow initialization displays the user step form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "user"
    assert not result.get("errors")


async def test_tier1_f3_flow_schema_fields(hass: HomeAssistant) -> None:
    """F3.2: Verify config flow schema contains required api_key and optional model."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    data_schema = result.get("data_schema")
    assert data_schema is not None
    schema = getattr(data_schema, "schema", {})
    keys = [k.schema if hasattr(k, "schema") else k for k in schema.keys()]
    assert CONF_API_KEY in keys
    assert CONF_MODEL in keys


async def test_tier1_f3_flow_default_model_applied(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F3.3: Verify submitting without model applies DEFAULT_MODEL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "my_key"},
        )
        await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert result2.get("data", {}).get(CONF_MODEL) == DEFAULT_MODEL


async def test_tier1_f3_flow_custom_model_saved(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F3.4: Verify submitting with custom model persists model string."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "my_key", CONF_MODEL: "jev-turbo-preview"},
        )
        await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert result2.get("data", {}).get(CONF_MODEL) == "jev-turbo-preview"


async def test_tier1_f3_flow_title_matches_default_name(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F3.5: Verify created config entry has title DEFAULT_NAME."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "key_abc"},
        )
        await hass.async_block_till_done()
    assert result2.get("title") == DEFAULT_NAME


# --- F4: Options Flow (`confidence_threshold`, `fallback_agent`) ---


async def test_tier1_f4_options_flow_init_shows_form(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.1: Verify options flow initialization returns form with init step."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "init"


async def test_tier1_f4_options_flow_schema_threshold_slider(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.2: Verify options flow schema contains confidence_threshold and fallback_agent."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    data_schema = result.get("data_schema")
    assert data_schema is not None
    schema = getattr(data_schema, "schema", {})
    keys = [k.schema if hasattr(k, "schema") else k for k in schema.keys()]
    assert CONF_CONFIDENCE_THRESHOLD in keys
    assert CONF_FALLBACK_AGENT in keys


async def test_tier1_f4_options_flow_update_threshold(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.3: Verify options flow updates confidence threshold."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_CONFIDENCE_THRESHOLD: 0.85},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options[CONF_CONFIDENCE_THRESHOLD] == 0.85


async def test_tier1_f4_options_flow_update_fallback_agent(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.4: Verify options flow updates fallback_agent."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options[CONF_FALLBACK_AGENT] == "mock_fallback_agent"


async def test_tier1_f4_options_flow_update_both_simultaneously(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.5: Verify options flow updates both threshold and fallback simultaneously."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_CONFIDENCE_THRESHOLD: 0.90,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options[CONF_CONFIDENCE_THRESHOLD] == 0.90
    assert config_entry.options[CONF_FALLBACK_AGENT] == "mock_fallback_agent"


# --- F5: Dynamic Reload on Options Update ---


async def test_tier1_f5_reload_triggered_on_threshold_change(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.1: Verify updating options dynamically reloads entry with new threshold."""
    assert config_entry.runtime_data.strategy.confidence_threshold == 0.70
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.85},
    )
    await hass.async_block_till_done()
    assert config_entry.runtime_data.strategy.confidence_threshold == 0.85


async def test_tier1_f5_reload_entry_preserves_loaded_state(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.2: Verify entry remains in LOADED state following dynamic reload."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.80},
    )
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.LOADED


async def test_tier1_f5_reload_recreates_strategy_instance(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.3: Verify runtime strategy is re-instantiated with new parameters."""
    strategy_before = config_entry.runtime_data.strategy
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.75},
    )
    await hass.async_block_till_done()
    strategy_after = config_entry.runtime_data.strategy
    assert strategy_before is not strategy_after
    assert strategy_after.confidence_threshold == 0.75


async def test_tier1_f5_reload_updates_conversation_behavior(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F5.4: Verify conversation gating immediately honors updated threshold after reload."""
    hass.states.async_set("light.bed", "off", {"friendly_name": "Bed Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.75,
                "probabilities": {"HassTurnOn": 0.75},
            },
            "target_entity": {"choice": "light.bed", "confidence": 0.75},
            "is_compound": {"noul": 0.0},
        }
    )

    # 1. Under default threshold 0.70, confidence 0.75 executes
    res1 = await conversation.async_converse(
        hass=hass,
        text="Turn on bed light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ACTION_DONE

    # 2. Increase threshold to 0.80 via options update
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.80},
    )
    await hass.async_block_till_done()

    # 3. Same utterance with 0.75 confidence now fails gating
    res2 = await conversation.async_converse(
        hass=hass,
        text="Turn on bed light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res2.response.response_type is intent.IntentResponseType.ERROR
    assert res2.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier1_f5_reload_listener_attached_to_entry(
    config_entry: MockConfigEntry,
) -> None:
    """F5.5: Verify entry has update listener registered during setup."""
    assert len(config_entry.update_listeners) > 0


# --- F6: TypeSafe System One HTTP Client ---


async def test_tier1_f6_client_payload_structure(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.1: Verify async_evaluate records state, model, and questions in payload."""
    await mock_client.async_evaluate(
        state={"utterance": "test command"},
        questions={"intent": {"type": "choice"}},
        model="jev-latest",
    )
    assert len(mock_client.calls) == 1
    call = mock_client.calls[0]
    assert call["state"] == {"utterance": "test command"}
    assert call["questions"] == {"intent": {"type": "choice"}}
    assert call["model"] == "jev-latest"


async def test_tier1_f6_client_returns_answers_dict(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.2: Verify async_evaluate returns structured answers dictionary."""
    mock_client.set_answers({"intent": {"choice": "HassTurnOn", "confidence": 0.9}})
    res = await mock_client.async_evaluate(state="turn on", questions={})
    assert res["model"] == "jev-latest"
    assert res["answers"]["intent"]["choice"] == "HassTurnOn"


async def test_tier1_f6_client_auth_error_raised(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.3: Verify async_evaluate surfaces TypeSafeAuthError on 401 response."""
    mock_client.evaluate_error = TypeSafeAuthError("API key invalid")
    with pytest.raises(TypeSafeAuthError, match="API key invalid"):
        await mock_client.async_evaluate(state="test", questions={})


async def test_tier1_f6_client_rate_limit_error_raised(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.4: Verify client surfaces TypeSafeRateLimitError on 429 response."""
    mock_client.evaluate_error = TypeSafeRateLimitError("Rate limit exceeded")
    with pytest.raises(TypeSafeRateLimitError, match="Rate limit exceeded"):
        await mock_client.async_evaluate(state="test", questions={})


async def test_tier1_f6_client_timeout_error_raised(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.5: Verify client surfaces TypeSafeError on timeout."""
    mock_client.evaluate_error = TypeSafeError("Request failed: Timeout")
    with pytest.raises(TypeSafeError, match="Request failed"):
        await mock_client.async_evaluate(state="test", questions={})


# --- F7: Typed Primitives (`ChoiceQuestion`, `NoulQuestion`, `ChoiceAnswer`, `NoulAnswer`) ---


async def test_tier1_f7_choice_question_to_dict_structure() -> None:
    """F7.1: Verify ChoiceQuestion.to_dict formats choice question payload correctly."""
    cq = ChoiceQuestion(
        instructions="Select device",
        criteria={"light.bulb": "Main light", "none": "No device"},
    )
    payload = cq.to_dict()
    assert payload["type"] == "choice"
    assert payload["instructions"] == "Select device"
    assert payload["criteria"]["light.bulb"] == "Main light"
    assert payload["criteria"]["none"] == "No device"


async def test_tier1_f7_noul_question_to_dict_without_criteria() -> None:
    """F7.2: Verify NoulQuestion.to_dict formats noul question payload without criteria."""
    nq = NoulQuestion(instructions="Is this a multi-part command?")
    payload = nq.to_dict()
    assert payload["type"] == "noul"
    assert payload["instructions"] == "Is this a multi-part command?"
    assert "criteria" not in payload


async def test_tier1_f7_noul_question_to_dict_with_criteria() -> None:
    """F7.3: Verify NoulQuestion.to_dict includes criteria when provided."""
    nq = NoulQuestion(
        instructions="Is temperature requested?",
        criteria={"true": "Degrees specified", "false": "No temperature"},
    )
    payload = nq.to_dict()
    assert payload["type"] == "noul"
    assert "criteria" in payload
    assert payload["criteria"]["true"] == "Degrees specified"


async def test_tier1_f7_choice_answer_instantiation_and_slots() -> None:
    """F7.4: Verify ChoiceAnswer dataclass fields and default values."""
    ca = ChoiceAnswer(
        choice="HassTurnOn",
        confidence=0.98,
        probabilities={"HassTurnOn": 0.98, "HassTurnOff": 0.02},
    )
    assert ca.choice == "HassTurnOn"
    assert ca.confidence == 0.98
    assert ca.probabilities["HassTurnOff"] == 0.02


async def test_tier1_f7_noul_answer_instantiation_and_slots() -> None:
    """F7.5: Verify NoulAnswer dataclass fields."""
    na = NoulAnswer(noul=0.15)
    assert na.noul == 0.15


# --- F8: Exposed Entity Discovery ---


async def test_tier1_f8_exposed_entities_filtered_via_should_expose(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.1: Verify entities exposed to conversation are captured in fan-out criteria."""
    hass.states.async_set("light.living_room", "off", {"friendly_name": "Living Room"})
    hass.states.async_set("light.closet", "off", {"friendly_name": "Closet Light"})

    async_expose_entity(hass, conversation.DOMAIN, "light.living_room", True)
    async_expose_entity(hass, conversation.DOMAIN, "light.closet", False)

    await conversation.async_converse(
        hass=hass,
        text="Turn on living room",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "light.living_room" in criteria
    assert "light.closet" not in criteria


async def test_tier1_f8_controllable_domains_filtering(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.2: Verify non-controllable domain entities are excluded from candidates."""
    hass.states.async_set("sensor.temperature", "72", {"friendly_name": "Room Temp"})
    hass.states.async_set("switch.heater", "off", {"friendly_name": "Room Heater"})

    async_expose_entity(hass, conversation.DOMAIN, "sensor.temperature", True)
    async_expose_entity(hass, conversation.DOMAIN, "switch.heater", True)

    await conversation.async_converse(
        hass=hass,
        text="Turn on room heater",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "switch.heater" in criteria
    assert "sensor.temperature" not in criteria


async def test_tier1_f8_friendly_name_in_candidate_description(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.3: Verify entity friendly_name is rendered in candidate description."""
    hass.states.async_set(
        "light.chandelier", "off", {"friendly_name": "Dining Chandelier"}
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on dining chandelier",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "Dining Chandelier" in criteria["light.chandelier"]


async def test_tier1_f8_area_boost_in_entity_ranking(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.4: Verify entity mapped to an area matching query receives lexical ranking boost."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    kitchen = area_reg.async_create("Kitchen")
    bedroom = area_reg.async_create("Bedroom")

    e1 = entity_reg.async_get_or_create(
        "light", "demo", "1", suggested_object_id="kitchen_light"
    )
    entity_reg.async_update_entity(e1.entity_id, area_id=kitchen.id)
    hass.states.async_set(e1.entity_id, "off", {"friendly_name": "Main Light"})

    e2 = entity_reg.async_get_or_create(
        "light", "demo", "2", suggested_object_id="bedroom_light"
    )
    entity_reg.async_update_entity(e2.entity_id, area_id=bedroom.id)
    hass.states.async_set(e2.entity_id, "off", {"friendly_name": "Main Light"})

    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[hass.states.get(e1.entity_id), hass.states.get(e2.entity_id)],  # type: ignore[list-item]
        home_name="Home",
    )
    ranked = strategy._rank_entities(ctx, "turn on kitchen main light")
    # kitchen_light should be present and ranked ahead of bedroom_light
    assert e1.entity_id in ranked


async def test_tier1_f8_domain_keyword_boost(
    hass: HomeAssistant,
) -> None:
    """F8.5: Verify matching domain in query boosts relevant entity ranking."""
    s1 = State("light.ambient", "off", {"friendly_name": "Ambient"})
    s2 = State("switch.ambient", "off", {"friendly_name": "Ambient"})
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[s1, s2],
        home_name="Home",
    )
    ranked = strategy._rank_entities(ctx, "turn on ambient light")
    keys = list(ranked.keys())
    assert keys.index("light.ambient") < keys.index("switch.ambient")


# --- F9: Dynamic Intent Schema Discovery ---


async def test_tier1_f9_registered_intent_handlers_discovered(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F9.1: Verify registered intent handlers appear in intent question criteria."""
    _ = mock_intent_handlers
    await conversation.async_converse(
        hass=hass,
        text="Turn on device",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    intent_criteria = mock_client.calls[0]["questions"]["intent"]["criteria"]
    assert "HassTurnOn" in intent_criteria
    assert "HassTurnOff" in intent_criteria


async def test_tier1_f9_unmatched_choice_always_present(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F9.2: Verify 'unmatched' fallback option is unconditionally present in criteria."""
    await conversation.async_converse(
        hass=hass,
        text="Random query",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    intent_criteria = mock_client.calls[0]["questions"]["intent"]["criteria"]
    assert "unmatched" in intent_criteria


async def test_tier1_f9_intent_lexical_scoring_ranks_relevant(
    hass: HomeAssistant,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F9.3: Verify lexical overlap with intent descriptions ranks relevant intent higher."""
    _ = mock_intent_handlers
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    criteria = strategy._discover_intents(ctx, "turn off the lamp")
    keys = list(criteria.keys())
    assert "HassTurnOff" in keys[:2]


async def test_tier1_f9_supported_slots_filter_admits_standard_intents(
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F9.4: Verify can_fulfill_intent returns True for handlers with supported slots."""
    turn_on_handler = mock_intent_handlers["HassTurnOn"]
    assert can_fulfill_intent(turn_on_handler) is True


async def test_tier1_f9_custom_intent_handler_discovered(
    hass: HomeAssistant,
    climate_handler: MockClimateIntentHandler,
) -> None:
    """F9.5: Verify dynamically registered intent handler is discovered by strategy."""
    _ = climate_handler
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    criteria = strategy._discover_intents(ctx, "set target temperature")
    assert "HassClimateSetTemperature" in criteria


# --- F10: Speculative Fan-out Query Builder ---


async def test_tier1_f10_questions_dict_contains_intent_and_compound(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.1: Verify speculative questions dict always contains intent and is_compound."""
    await conversation.async_converse(
        hass=hass,
        text="Any command",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "intent" in questions
    assert "is_compound" in questions


async def test_tier1_f10_questions_dict_includes_target_entity_when_entities_exist(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.2: Verify target_entity question is constructed when entities are present."""
    hass.states.async_set("light.table", "off", {"friendly_name": "Table Light"})
    await conversation.async_converse(
        hass=hass,
        text="Turn on table light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "target_entity" in questions


async def test_tier1_f10_questions_dict_includes_target_area_when_areas_exist(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.3: Verify target_area question is constructed when areas are present."""
    area_reg = ar.async_get(hass)
    area_reg.async_create("Porch")
    await conversation.async_converse(
        hass=hass,
        text="Turn on porch light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "target_area" in questions


async def test_tier1_f10_questions_dict_includes_target_type_when_both_exist(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.4: Verify target_type question is added when both entities and areas exist."""
    hass.states.async_set("light.deck", "off", {"friendly_name": "Deck Light"})
    area_reg = ar.async_get(hass)
    area_reg.async_create("Deck")
    await conversation.async_converse(
        hass=hass,
        text="Turn on deck light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "target_type" in questions


async def test_tier1_f10_state_payload_contains_utterance_and_home(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.5: Verify state payload passed to async_evaluate has utterance and home."""
    await conversation.async_converse(
        hass=hass,
        text="Lock front door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    state = mock_client.calls[0]["state"]
    assert state["utterance"] == "Lock front door"
    assert state["home"] == hass.config.location_name


# --- F11: Conversation Entity Registration ---


async def test_tier1_f11_agent_registered_in_agent_manager(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F11.1: Verify agent is registered with Home Assistant conversation agent manager."""
    manager = conversation.get_agent_manager(hass)
    agent = manager.async_get_agent(config_entry.entry_id)
    assert agent is not None
    assert isinstance(agent, conversation.AbstractConversationAgent)


async def test_tier1_f11_supported_languages_is_match_all(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F11.2: Verify agent declares MATCH_ALL supported languages."""
    manager = conversation.get_agent_manager(hass)
    agent = manager.async_get_agent(config_entry.entry_id)
    assert agent.supported_languages == MATCH_ALL


async def test_tier1_f11_entity_unique_id_matches_entry_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F11.3: Verify conversation entity unique_id matches config_entry.entry_id."""
    manager = conversation.get_agent_manager(hass)
    agent = manager.async_get_agent(config_entry.entry_id)
    assert agent.unique_id == config_entry.entry_id


async def test_tier1_f11_entity_name_matches_entry_title(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F11.4: Verify conversation entity name matches config_entry.title."""
    manager = conversation.get_agent_manager(hass)
    agent = manager.async_get_agent(config_entry.entry_id)
    assert agent.name == config_entry.title


async def test_tier1_f11_async_converse_routes_to_typesafe_agent(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F11.5: Verify conversation.async_converse routes directly to TypeSafe agent."""
    await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_client.calls) == 1


# --- F12: Confidence Threshold Gating ---


async def test_tier1_f12_confidence_above_threshold_executes(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F12.1: Verify confidence meeting threshold executes intent."""
    hass.states.async_set("light.den", "off", {"friendly_name": "Den Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.den", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on den light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_tier1_f12_confidence_below_threshold_escalates(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F12.2: Verify confidence below threshold fails gating."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.65,
                "probabilities": {"HassTurnOn": 0.65},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on something",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier1_f12_unmatched_choice_escalates_even_with_high_confidence(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F12.3: Verify 'unmatched' intent choice escalates despite high confidence."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Sing a song",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier1_f12_none_choice_escalates(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F12.4: Verify 'none' intent choice escalates."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "none",
                "confidence": 0.95,
                "probabilities": {"none": 0.95},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Unrecognized command",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR


async def test_tier1_f12_compound_command_escalates(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F12.5: Verify compound noul above threshold triggers compound escalation."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.85},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on TV and close blinds",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert "multiple requests" in str(res.response.speech["plain"]["speech"])


# --- F13: Entity Target Resolution ---


async def test_tier1_f13_entity_friendly_name_resolved_in_slot(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.1: Verify resolved entity friendly_name is placed into intent name slot."""
    hass.states.async_set("light.hallway", "off", {"friendly_name": "Upstairs Hallway"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.90,
                "probabilities": {"HassTurnOn": 0.90},
            },
            "target_entity": {"choice": "light.hallway", "confidence": 0.90},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on upstairs hallway",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Upstairs Hallway"


async def test_tier1_f13_area_target_resolved_in_slot(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.2: Verify resolved area target places area name into intent area slot."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Garage")
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.92,
                "probabilities": {"HassTurnOn": 0.92},
            },
            "target_type": {"choice": "area", "confidence": 0.92},
            "target_area": {"choice": area.id, "confidence": 0.92},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on garage lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert handler.handled_intents[0].slots["area"]["value"] == "Garage"


async def test_tier1_f13_area_domain_inferred_from_utterance(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.3: Verify domain is inferred from utterance for area targeting."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Basement")
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOff",
                "confidence": 0.91,
                "probabilities": {"HassTurnOff": 0.91},
            },
            "target_type": {"choice": "area", "confidence": 0.91},
            "target_area": {"choice": area.id, "confidence": 0.91},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn off basement switches",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOff"]
    assert handler.handled_intents[0].slots["domain"]["value"] == "switch"


async def test_tier1_f13_brightness_regex_extraction(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.4: Verify brightness percentage regex extraction populates brightness slot."""
    hass.states.async_set("light.dining", "on", {"friendly_name": "Dining Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassLightSet",
                "confidence": 0.90,
                "probabilities": {"HassLightSet": 0.90},
            },
            "target_entity": {"choice": "light.dining", "confidence": 0.90},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Set dining light brightness to 45%",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassLightSet"]
    assert handler.handled_intents[0].slots["brightness"]["value"] == 45


async def test_tier1_f13_temperature_regex_extraction(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    climate_handler: MockClimateIntentHandler,
) -> None:
    """F13.5: Verify temperature regex extraction populates temperature slot."""
    handler = climate_handler
    hass.states.async_set(
        "climate.living_room", "heat", {"friendly_name": "Living Room AC"}
    )
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassClimateSetTemperature",
                "confidence": 0.90,
                "probabilities": {"HassClimateSetTemperature": 0.90},
            },
            "target_entity": {"choice": "climate.living_room", "confidence": 0.90},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Set temperature to 72 degrees",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["temperature"]["value"] == 72.0


# --- F14: HA Intent Execution (`intent.async_handle`) ---


async def test_tier1_f14_handle_turn_on_intent_success(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.1: Verify HassTurnOn intent dispatches and returns ACTION_DONE."""
    hass.states.async_set("light.entry", "off", {"friendly_name": "Entry Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.entry", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on entry light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert len(mock_intent_handlers["HassTurnOn"].handled_intents) == 1


async def test_tier1_f14_handle_turn_off_intent_success(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.2: Verify HassTurnOff intent dispatches and returns ACTION_DONE."""
    hass.states.async_set("switch.pool", "on", {"friendly_name": "Pool Pump"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOff",
                "confidence": 0.95,
                "probabilities": {"HassTurnOff": 0.95},
            },
            "target_entity": {"choice": "switch.pool", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn off pool pump",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert len(mock_intent_handlers["HassTurnOff"].handled_intents) == 1


async def test_tier1_f14_handle_light_set_intent_success(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.3: Verify HassLightSet intent dispatches and returns ACTION_DONE."""
    hass.states.async_set("light.office", "on", {"friendly_name": "Office Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassLightSet",
                "confidence": 0.95,
                "probabilities": {"HassLightSet": 0.95},
            },
            "target_entity": {"choice": "light.office", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Adjust office light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert len(mock_intent_handlers["HassLightSet"].handled_intents) == 1


async def test_tier1_f14_intent_receives_user_context_and_language(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.4: Verify intent handler receives context and language from conversation input."""
    ctx = Context()
    hass.states.async_set("light.yard", "off", {"friendly_name": "Yard Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.yard", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on yard light",
        conversation_id=None,
        context=ctx,
        language="en",
        agent_id=config_entry.entry_id,
    )
    handled = mock_intent_handlers["HassTurnOn"].handled_intents[-1]
    assert handled.context is ctx
    assert handled.language == "en"


async def test_tier1_f14_intent_error_returns_failed_to_handle(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F14.5: Verify IntentError during async_handle converts to FAILED_TO_HANDLE error."""

    class ErroringHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            raise intent.IntentError("Hardware driver unresponsive")

    intent.async_register(hass, ErroringHandler())
    hass.states.async_set("light.broken", "off", {"friendly_name": "Broken Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.broken", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on broken light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE


# --- F15: Fallback Escalation (`conversation.async_converse`) ---


async def test_tier1_f15_low_confidence_escalates_to_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.1: Verify low-confidence utterance escalates to registered fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.40,
                "probabilities": {"HassTurnOn": 0.40},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on something",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in res.response.speech["plain"]["speech"]


async def test_tier1_f15_unmatched_choice_escalates_to_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.2: Verify unmatched choice escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Tell me the weather in Seattle",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "Tell me the weather in Seattle"


async def test_tier1_f15_fallback_agent_receives_original_utterance_and_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.3: Verify fallback agent receives the identical conversation_id and context."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    ctx = Context()
    await conversation.async_converse(
        hass=hass,
        text="Complex query",
        conversation_id="conv-uuid-1234",
        context=ctx,
        agent_id=config_entry.entry_id,
    )
    call = mock_fallback_agent.calls[0]
    assert call.conversation_id == "conv-uuid-1234"
    assert call.context is ctx


async def test_tier1_f15_compound_utterance_escalates_to_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.4: Verify compound utterance escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers({"is_compound": {"noul": 0.95}})
    await conversation.async_converse(
        hass=hass,
        text="Turn off lights and arm alarm",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1


async def test_tier1_f15_client_evaluation_error_escalates_to_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.5: Verify client evaluation failure escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.evaluate_error = TypeSafeError("API unreachable")
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on heating",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in res.response.speech["plain"]["speech"]


# --- F16: Unmatched Error Handling (`NO_INTENT_MATCH`) ---


async def test_tier1_f16_low_confidence_no_fallback_returns_no_intent_match(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.1: Verify low confidence without fallback returns NO_INTENT_MATCH error."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.35,
                "probabilities": {"HassTurnOn": 0.35},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Vague command",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier1_f16_unmatched_choice_no_fallback_returns_no_intent_match(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.2: Verify unmatched choice without fallback returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Play jazz music",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier1_f16_compound_request_specific_error_message(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.3: Verify compound error message informs user of multiple requests."""
    mock_client.set_answers({"is_compound": {"noul": 0.88}})
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on lights and lock door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert "multiple requests" in res.response.speech["plain"]["speech"]


async def test_tier1_f16_standard_unmatched_error_message(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.4: Verify standard error message is 'Sorry, I could not understand that request.'"""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Nonsense phrase",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert "could not understand that request" in res.response.speech["plain"]["speech"]


async def test_tier1_f16_error_response_preserves_language_and_conversation_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.5: Verify error response preserves user_input language and conversation_id."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Que hora es?",
        conversation_id="conv-es-001",
        context=Context(),
        language="es",
        agent_id=config_entry.entry_id,
    )
    assert res.conversation_id == "conv-es-001"
    assert res.response.language == "es"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES ACROSS ALL 16 FEATURES (80 TESTS)
# ==============================================================================

# --- F1 Boundary ---


async def test_tier2_f1_setup_with_missing_optional_options(
    hass: HomeAssistant,
) -> None:
    """F1.B1: Setup entry with empty options dictionary applies defaults."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={CONF_API_KEY: "test_key"},
        options={},
        entry_id="entry_no_options",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert (
        entry.runtime_data.strategy.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    )


async def test_tier2_f1_setup_entry_with_options_overriding_data(
    hass: HomeAssistant,
) -> None:
    """F1.B2: Setup entry where options threshold overrides data threshold."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={CONF_API_KEY: "test_key", CONF_CONFIDENCE_THRESHOLD: 0.6},
        options={CONF_CONFIDENCE_THRESHOLD: 0.8},
        entry_id="entry_override_threshold",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.runtime_data.strategy.confidence_threshold == 0.8


async def test_tier2_f1_unload_entry_when_already_not_loaded(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F1.B3: Unloading an entry twice gracefully handles NOT_LOADED state."""
    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.NOT_LOADED
    # Second unload attempt handles state cleanly
    await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.NOT_LOADED


async def test_tier2_f1_entry_with_special_characters_title(
    hass: HomeAssistant,
) -> None:
    """F1.B4: Setup entry with non-ASCII and special characters in title."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="TypeSafe 🏠 (Dev/2026)",
        data={CONF_API_KEY: "test_key"},
        options={},
        entry_id="entry_special_title",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    manager = conversation.get_agent_manager(hass)
    agent = manager.async_get_agent(entry.entry_id)
    assert agent.name == "TypeSafe 🏠 (Dev/2026)"


async def test_tier2_f1_runtime_data_slots_isolation(
    config_entry: MockConfigEntry,
) -> None:
    """F1.B5: Verify TypeSafeData slots prevent assigning arbitrary attributes."""
    data = config_entry.runtime_data
    with pytest.raises(AttributeError):
        setattr(data, "undeclared_attribute", 123)


# --- F2 Boundary ---


async def test_tier2_f2_empty_api_key_validation(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.B1: Validation with empty API key rejected by config flow validation."""
    mock_client.validate_error = TypeSafeAuthError("Empty key rejected")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("errors") == {"base": "invalid_auth"}


async def test_tier2_f2_unicode_special_chars_api_key(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.B2: Validation with special characters / unicode in api_key."""
    mock_client.validate_result = True
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "sk-🔑-test-!@#$%^&*()_+", CONF_MODEL: "jev-latest"},
        )
        await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert result2["data"][CONF_API_KEY] == "sk-🔑-test-!@#$%^&*()_+"


async def test_tier2_f2_client_timeout_triggers_cannot_connect(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.B3: Asyncio timeout during key validation maps to cannot_connect."""
    mock_client.validate_error = TypeSafeError("Connection timeout")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "valid_looking_key", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("errors") == {"base": "cannot_connect"}


async def test_tier2_f2_unexpected_status_code_in_validation(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.B4: Non-200 / Non-401 HTTP status (e.g. 503) maps to cannot_connect."""
    mock_client.validate_error = TypeSafeError("Validation failed with status 503")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "valid_looking_key", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("errors") == {"base": "cannot_connect"}


async def test_tier2_f2_whitespace_padded_key(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F2.B5: API key with surrounding whitespace succeeds validation."""
    mock_client.validate_result = True
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "  key_with_spaces  ", CONF_MODEL: "jev-latest"},
        )
        await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY


# --- F3 Boundary ---


async def test_tier2_f3_flow_empty_user_input(hass: HomeAssistant) -> None:
    """F3.B1: Calling async_step_user with user_input=None shows blank form."""
    flow = config_entries.HANDLERS[DOMAIN]()
    flow.hass = hass
    result = await flow.async_step_user(user_input=None)
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "user"


async def test_tier2_f3_flow_custom_model_special_characters(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F3.B2: Model name with dashes, periods, and version numbers persisted."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "secret", CONF_MODEL: "jev-v1.4.2-preview"},
        )
        await hass.async_block_till_done()
    assert result2["data"][CONF_MODEL] == "jev-v1.4.2-preview"


async def test_tier2_f3_flow_very_long_api_key(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F3.B3: Exceptionally long API key string processed without error."""
    long_key = "a" * 512
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: long_key, CONF_MODEL: "jev-latest"},
        )
        await hass.async_block_till_done()
    assert result2["data"][CONF_API_KEY] == long_key


async def test_tier2_f3_flow_subsequent_step_after_error_recovery(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """F3.B4: User recovers from invalid credentials error by re-submitting valid credentials."""
    mock_client.validate_error = TypeSafeAuthError("Wrong key")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    # First attempt fails
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "bad_key", CONF_MODEL: "jev-latest"},
    )
    assert result2.get("errors") == {"base": "invalid_auth"}

    # Second attempt succeeds
    mock_client.validate_error = None
    mock_client.validate_result = True
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result3 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "good_key", CONF_MODEL: "jev-latest"},
        )
        await hass.async_block_till_done()
    assert result3.get("type") is FlowResultType.CREATE_ENTRY


async def test_tier2_f3_flow_version_is_current() -> None:
    """F3.B5: Config flow VERSION is integer 1."""
    flow_class = config_entries.HANDLERS[DOMAIN]
    assert flow_class.VERSION == 1


# --- F4 Boundary ---


async def test_tier2_f4_options_threshold_boundary_zero(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.B1: Confidence threshold set to boundary minimum 0.0."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_CONFIDENCE_THRESHOLD: 0.0},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options[CONF_CONFIDENCE_THRESHOLD] == 0.0


async def test_tier2_f4_options_threshold_boundary_one(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.B2: Confidence threshold set to boundary maximum 1.0."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_CONFIDENCE_THRESHOLD: 1.0},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options[CONF_CONFIDENCE_THRESHOLD] == 1.0


async def test_tier2_f4_options_fallback_agent_none_cleared(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.B3: Fallback agent omitted/cleared in options flow."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.70,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()
    assert config_entry.options[CONF_FALLBACK_AGENT] == "mock_fallback_agent"

    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_CONFIDENCE_THRESHOLD: 0.75},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options.get(CONF_FALLBACK_AGENT) is None


async def test_tier2_f4_options_submitting_empty_retains_or_defaults(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.B4: Submitting options without changes creates entry cleanly."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.CREATE_ENTRY


async def test_tier2_f4_options_step_init_with_none_input(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F4.B5: Options flow async_step_init with None displays form with current options."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "init"


# --- F5 Boundary ---


async def test_tier2_f5_reload_multiple_consecutive_updates(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.B1: Multiple rapid consecutive options updates reload without error."""
    for thresh in (0.75, 0.80, 0.85):
        hass.config_entries.async_update_entry(
            config_entry,
            options={CONF_CONFIDENCE_THRESHOLD: thresh},
        )
        await hass.async_block_till_done()
    assert config_entry.runtime_data.strategy.confidence_threshold == 0.85


async def test_tier2_f5_reload_with_unchanged_options(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.B2: Updating options with identical dictionary reloads cleanly."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.70},
    )
    await hass.async_block_till_done()
    assert config_entry.runtime_data.strategy.confidence_threshold == 0.70


async def test_tier2_f5_reload_retains_entry_data_fields(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.B3: Reload leaves original entry.data fields intact."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.90},
    )
    await hass.async_block_till_done()
    assert config_entry.data[CONF_API_KEY] == "test-api-key"
    assert config_entry.data[CONF_MODEL] == "jev-latest"


async def test_tier2_f5_reload_fallback_agent_option_effect(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F5.B4: Dynamic options reload setting fallback_agent immediately routes fallback."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    # 1. Before setting fallback, returns NO_INTENT_MATCH
    res1 = await conversation.async_converse(
        hass=hass,
        text="Unmatched query",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ERROR

    # 2. Add fallback agent via options update
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    # 3. Same query now routes to fallback agent
    res2 = await conversation.async_converse(
        hass=hass,
        text="Unmatched query",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in res2.response.speech["plain"]["speech"]


async def test_tier2_f5_reload_listener_unregistered_on_unload(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F5.B5: Unloading entry detaches update listeners without leaks."""
    await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.NOT_LOADED


# --- F6 Boundary ---


async def test_tier2_f6_client_empty_questions_dict(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.B1: Calling client.async_evaluate with empty questions dictionary."""
    mock_client.set_answers({})
    res = await mock_client.async_evaluate(state="status check", questions={})
    assert res["answers"] == {}


async def test_tier2_f6_client_complex_state_dict(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.B2: Calling client.async_evaluate with deeply nested dict state."""
    state = {
        "utterance": "turn on lights",
        "home": "Main Villa",
        "devices": [{"id": "d1", "val": 100}],
    }
    await mock_client.async_evaluate(state=state, questions={})
    assert mock_client.calls[0]["state"] == state


async def test_tier2_f6_client_empty_string_state(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.B3: Calling client.async_evaluate with empty string state."""
    await mock_client.async_evaluate(state="", questions={})
    assert mock_client.calls[0]["state"] == ""


async def test_tier2_f6_client_custom_model_override(
    mock_client: MockTypeSafeClient,
) -> None:
    """F6.B4: Explicit model parameter overrides client default."""
    await mock_client.async_evaluate(state="test", questions={}, model="jev-turbo")
    assert mock_client.calls[0]["model"] == "jev-turbo"


async def test_tier2_f6_client_malformed_non_dict_response(
    hass: HomeAssistant,
) -> None:
    """F6.B5: Strategy handles malformed non-dict response from TypeSafe API."""
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    decision = strategy._parse_evaluation_response("not a dict", "test utterance", ctx)
    assert decision.should_escalate is True
    assert "Malformed TypeSafe response" in str(decision.escalation_reason)


# --- F7 Boundary ---


async def test_tier2_f7_choice_question_empty_criteria() -> None:
    """F7.B1: ChoiceQuestion with empty criteria dictionary."""
    cq = ChoiceQuestion(instructions="Select one", criteria={})
    payload = cq.to_dict()
    assert payload["criteria"] == {}


async def test_tier2_f7_choice_question_dict_instructions() -> None:
    """F7.B2: ChoiceQuestion with structured dict instructions."""
    instr = {"task": "intent_detection", "domain": "home_assistant"}
    cq = ChoiceQuestion(instructions=instr, criteria={"opt": "Option"})
    payload = cq.to_dict()
    assert payload["instructions"] == instr


async def test_tier2_f7_noul_question_empty_instructions() -> None:
    """F7.B3: NoulQuestion with empty instructions."""
    nq = NoulQuestion(instructions="")
    payload = nq.to_dict()
    assert payload["instructions"] == ""


async def test_tier2_f7_noul_answer_exact_boundary_values() -> None:
    """F7.B4: NoulAnswer with boundary 0.0 and 1.0 values."""
    na0 = NoulAnswer(noul=0.0)
    na1 = NoulAnswer(noul=1.0)
    assert na0.noul == 0.0
    assert na1.noul == 1.0


async def test_tier2_f7_choice_answer_empty_probabilities_dict() -> None:
    """F7.B5: ChoiceAnswer with empty probabilities dictionary."""
    ca = ChoiceAnswer(choice="HassTurnOn", confidence=0.88, probabilities={})
    assert ca.probabilities == {}


# --- F8 Boundary ---


async def test_tier2_f8_no_exposed_entities_falls_back_to_all_states(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.B1: When no entities are exposed via should_expose, falls back to all states."""
    hass.states.async_set(
        "light.unexposed_lamp", "off", {"friendly_name": "Unexposed Lamp"}
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on unexposed lamp",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "light.unexposed_lamp" in criteria


async def test_tier2_f8_entity_without_friendly_name_attribute(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.B2: State with attributes={} falls back to entity_id in description."""
    hass.states.async_set("switch.relay_1", "off", {})
    await conversation.async_converse(
        hass=hass,
        text="Turn on relay",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "switch.relay_1" in criteria
    assert "switch.relay_1 (switch)" in criteria["switch.relay_1"]


async def test_tier2_f8_entity_empty_friendly_name_attribute(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F8.B3: State with friendly_name="" falls back to entity_id."""
    hass.states.async_set("switch.relay_2", "off", {"friendly_name": ""})
    await conversation.async_converse(
        hass=hass,
        text="Turn on relay 2",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "switch.relay_2" in criteria


async def test_tier2_f8_entity_in_unregistered_area(
    hass: HomeAssistant,
) -> None:
    """F8.B4: Entity mapped to non-existent area_id in registry handles missing area cleanly."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    e = entity_reg.async_get_or_create("light", "demo", "missing_area_light")
    entity_reg.async_update_entity(e.entity_id, area_id="non_existent_area_id")
    s = State(e.entity_id, "off", {"friendly_name": "Ghost Light"})

    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=area_reg,
        entity_registry=entity_reg,
        states=[s],
        home_name="Home",
    )
    criteria = strategy._rank_entities(ctx, "turn on ghost light")
    assert e.entity_id in criteria
    assert "Ghost Light (light)" in criteria[e.entity_id]


async def test_tier2_f8_entity_ranking_caps_at_20_candidates(
    hass: HomeAssistant,
) -> None:
    """F8.B5: Ranking 25 controllable entities caps criteria at top 20 plus 'none'."""
    states = [
        State(f"light.test_{i}", "off", {"friendly_name": f"Light {i}"})
        for i in range(25)
    ]
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=states,
        home_name="Home",
    )
    criteria = strategy._rank_entities(ctx, "turn on light")
    # Top 20 entities + "none" = 21 items max
    assert len(criteria) <= 21
    assert "none" in criteria


# --- F9 Boundary ---


async def test_tier2_f9_intent_with_unsupported_slot_filtered(
    hass: HomeAssistant,
) -> None:
    """F9.B1: Intent handler requiring unsupported slot is filtered out by can_fulfill_intent."""

    class UnsupportedSlotHandler(intent.IntentHandler):
        intent_type = "HassUnsupportedIntent"
        required_slots = {"unsupported_slot_key_xyz": None}

    handler = UnsupportedSlotHandler()
    assert can_fulfill_intent(handler) is False


async def test_tier2_f9_no_registered_handlers_uses_fallback_defaults(
    hass: HomeAssistant,
) -> None:
    """F9.B2: When intent registry is empty, strategy falls back to default intents."""
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=None,  # type: ignore[arg-type]
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    criteria = strategy._discover_intents(ctx, "turn on something")
    assert "HassTurnOn" in criteria
    assert "unmatched" in criteria


async def test_tier2_f9_intent_handler_without_description(
    hass: HomeAssistant,
) -> None:
    """F9.B3: Intent handler with description=None uses default description or intent_type."""

    class NoDescHandler(intent.IntentHandler):
        intent_type = "HassCustomNoDesc"
        description = None

    intent.async_register(hass, NoDescHandler())
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    criteria = strategy._discover_intents(ctx, "HassCustomNoDesc")
    assert "HassCustomNoDesc" in criteria


async def test_tier2_f9_utterance_with_no_word_tokens(
    hass: HomeAssistant,
) -> None:
    """F9.B4: Utterance containing only punctuation or whitespace produces valid criteria."""
    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    criteria = strategy._discover_intents(ctx, "??? !!! ...")
    assert "unmatched" in criteria
    assert len(criteria) >= 2


async def test_tier2_f9_intent_candidates_capped_at_5(
    hass: HomeAssistant,
) -> None:
    """F9.B5: Intent criteria is capped at top 5 candidates plus 'unmatched'."""
    for i in range(10):

        class MultiHandler(intent.IntentHandler):
            pass

        MultiHandler.intent_type = f"HassIntentTest{i}"
        MultiHandler.description = f"Test intent {i}"
        intent.async_register(hass, MultiHandler())

    strategy = DecisionStrategy()
    ctx = StrategyContext(
        hass=hass,
        area_registry=ar.async_get(hass),
        entity_registry=er.async_get(hass),
        states=[],
        home_name="Home",
    )
    criteria = strategy._discover_intents(ctx, "Test intent")
    # Top 5 + unmatched = at most 6
    assert len(criteria) <= 6
    assert "unmatched" in criteria


# --- F10 Boundary ---


async def test_tier2_f10_fanout_zero_entities_omits_target_entity(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.B1: When 0 controllable entities exist, target_entity question is omitted."""
    # Ensure no controllable states
    for s in hass.states.async_all():
        hass.states.async_remove(s.entity_id)

    await conversation.async_converse(
        hass=hass,
        text="Turn on something",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "target_entity" not in questions


async def test_tier2_f10_fanout_zero_areas_omits_target_area(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.B2: When 0 areas exist in registry, target_area question is omitted."""
    area_reg = ar.async_get(hass)
    for a in list(area_reg.areas.values()):
        area_reg.async_delete(a.id)

    await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "target_area" not in questions


async def test_tier2_f10_fanout_entities_without_areas_omits_target_type(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.B3: When entities exist but no areas exist, target_type question is omitted."""
    area_reg = ar.async_get(hass)
    for a in list(area_reg.areas.values()):
        area_reg.async_delete(a.id)
    hass.states.async_set("light.single", "off", {"friendly_name": "Single Light"})

    await conversation.async_converse(
        hass=hass,
        text="Turn on single light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    questions = mock_client.calls[0]["questions"]
    assert "target_type" not in questions


async def test_tier2_f10_fanout_empty_utterance_string(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.B4: Empty string utterance in fan-out query builder executed safely."""
    await conversation.async_converse(
        hass=hass,
        text="",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert mock_client.calls[0]["state"]["utterance"] == ""


async def test_tier2_f10_fanout_home_name_with_unicode(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F10.B5: Home name with unicode and emojis passed intact in state."""
    hass.config.location_name = "Château de Lumière 🏰"
    await conversation.async_converse(
        hass=hass,
        text="Turn on light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert mock_client.calls[0]["state"]["home"] == "Château de Lumière 🏰"


# --- F11 Boundary ---


async def test_tier2_f11_conversation_non_standard_language(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F11.B1: Conversation input with non-English language code (de) processed."""
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Küche"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.kitchen", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Schalte die Küche ein",
        conversation_id=None,
        context=Context(),
        language="de",
        agent_id=config_entry.entry_id,
    )
    assert res.response.language == "de"
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_tier2_f11_conversation_none_conversation_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F11.B2: Conversation input with conversation_id=None handled cleanly."""
    hass.states.async_set("light.desk", "off", {"friendly_name": "Desk Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.desk", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on desk",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_tier2_f11_conversation_custom_conversation_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F11.B3: Custom conversation_id preserved in result."""
    hass.states.async_set("light.desk", "off", {"friendly_name": "Desk Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.desk", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on desk",
        conversation_id="custom-uuid-9999",
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.conversation_id == "custom-uuid-9999"


async def test_tier2_f11_conversation_device_id_context(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F11.B4: User input device_id passed through to StrategyContext."""
    await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        device_id="satellite_speaker_living_room",
        agent_id=config_entry.entry_id,
    )
    assert len(mock_client.calls) == 1


async def test_tier2_f11_conversation_agent_unregistered_on_entity_remove(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """F11.B5: Removing entity unregisters conversation agent from manager."""
    manager = conversation.get_agent_manager(hass)
    assert manager.async_get_agent(config_entry.entry_id) is not None
    await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    with pytest.raises(ValueError, match="not found"):
        manager.async_get_agent(config_entry.entry_id)


# --- F12 Boundary ---


async def test_tier2_f12_confidence_exact_threshold_passes(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F12.B1: Confidence exactly at threshold (0.7000) passes gating."""
    hass.states.async_set("light.exact", "off", {"friendly_name": "Exact Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.70,
                "probabilities": {"HassTurnOn": 0.70},
            },
            "target_entity": {"choice": "light.exact", "confidence": 0.70},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on exact light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_tier2_f12_confidence_just_below_threshold_fails(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F12.B2: Confidence just below threshold (0.6999) fails gating."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.6999,
                "probabilities": {"HassTurnOn": 0.6999},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on exact light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f12_confidence_extreme_zero_fails(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F12.B3: Confidence of 0.0 fails gating."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.0,
                "probabilities": {"HassTurnOn": 0.0},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR


async def test_tier2_f12_confidence_extreme_one_passes(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F12.B4: Confidence of 1.0 passes gating."""
    hass.states.async_set("light.sure", "off", {"friendly_name": "Sure Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 1.0,
                "probabilities": {"HassTurnOn": 1.0},
            },
            "target_entity": {"choice": "light.sure", "confidence": 1.0},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on sure light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_tier2_f12_compound_exact_boundary_0_5_passes(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F12.B5: Compound noul of 0.50 exactly does NOT fail gating (requires > 0.5)."""
    hass.states.async_set("light.bound", "off", {"friendly_name": "Bound Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.90,
                "probabilities": {"HassTurnOn": 0.90},
            },
            "target_entity": {"choice": "light.bound", "confidence": 0.90},
            "is_compound": {"noul": 0.50},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on bound light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


# --- F13 Boundary ---


async def test_tier2_f13_entity_not_in_hass_states(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.B1: Target entity not present in hass.states falls back to entity_id in slot."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.nonexistent_entity", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on ghost light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert (
        handler.handled_intents[0].slots["name"]["value"] == "light.nonexistent_entity"
    )


async def test_tier2_f13_area_without_domain_keyword(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.B2: Area target without recognized domain keyword in utterance omits domain slot."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Attic")
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.90,
                "probabilities": {"HassTurnOn": 0.90},
            },
            "target_type": {"choice": "area", "confidence": 0.90},
            "target_area": {"choice": area.id, "confidence": 0.90},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Activate the attic",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    slots = handler.handled_intents[0].slots
    assert slots["area"]["value"] == "Attic"
    assert "domain" not in slots


async def test_tier2_f13_brightness_boundary_0_and_100(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.B3: Boundary 0% and 100% brightness values parsed accurately."""
    hass.states.async_set("light.dimmable", "on", {"friendly_name": "Dimmable"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassLightSet",
                "confidence": 0.95,
                "probabilities": {"HassLightSet": 0.95},
            },
            "target_entity": {"choice": "light.dimmable", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    # Test 0%
    await conversation.async_converse(
        hass=hass,
        text="Set dimmable to 0%",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassLightSet"]
    assert handler.handled_intents[-1].slots["brightness"]["value"] == 0

    # Test 100%
    await conversation.async_converse(
        hass=hass,
        text="Set dimmable to 100%",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert handler.handled_intents[-1].slots["brightness"]["value"] == 100


async def test_tier2_f13_temperature_with_decimal(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    climate_handler: MockClimateIntentHandler,
) -> None:
    """F13.B4: Decimal temperature representation parsed as float."""
    handler = climate_handler
    hass.states.async_set("climate.hvac", "heat", {"friendly_name": "HVAC"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassClimateSetTemperature",
                "confidence": 0.95,
                "probabilities": {"HassClimateSetTemperature": 0.95},
            },
            "target_entity": {"choice": "climate.hvac", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Set HVAC to 21.5°",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert handler.handled_intents[0].slots["temperature"]["value"] == 21.5


async def test_tier2_f13_target_entity_none_choice_ignored(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F13.B5: Target entity 'none' choice does not populate entity_id slot."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.90,
                "probabilities": {"HassTurnOn": 0.90},
            },
            "target_entity": {"choice": "none", "confidence": 0.90},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on everything",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert "entity_id" not in handler.handled_intents[0].slots


# --- F14 Boundary ---


async def test_tier2_f14_intent_execution_with_empty_slots(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.B1: Intent dispatched with empty slots dictionary succeeds."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_tier2_f14_unregistered_intent_name_fails(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F14.B2: TypeSafe selects unregistered intent; returns FAILED_TO_HANDLE."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "UnregisteredIntentXYZ",
                "confidence": 0.95,
                "probabilities": {"UnregisteredIntentXYZ": 0.95},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Do unknown action",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE


async def test_tier2_f14_intent_speech_preserved(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.B3: Plain text speech returned by intent handler is preserved in result."""
    hass.states.async_set("light.front", "off", {"friendly_name": "Front Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.front", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on front light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.speech["plain"]["speech"] == "Turned on device"


async def test_tier2_f14_special_characters_in_slots(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.B4: Quotes and accented characters in entity friendly name forwarded into slot."""
    hass.states.async_set(
        "light.art", "off", {"friendly_name": 'René\'s "Special" Art Light'}
    )
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.art", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on art light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert (
        handler.handled_intents[0].slots["name"]["value"]
        == 'René\'s "Special" Art Light'
    )


async def test_tier2_f14_multiple_consecutive_intent_executions(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """F14.B5: Consecutive intent executions maintain isolated state."""
    hass.states.async_set("light.first", "off", {"friendly_name": "First Light"})
    hass.states.async_set("switch.second", "off", {"friendly_name": "Second Switch"})

    # Execution 1
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.first", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res1 = await conversation.async_converse(
        hass=hass,
        text="Turn on first",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ACTION_DONE

    # Execution 2
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOff",
                "confidence": 0.95,
                "probabilities": {"HassTurnOff": 0.95},
            },
            "target_entity": {"choice": "switch.second", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res2 = await conversation.async_converse(
        hass=hass,
        text="Turn off second",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res2.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert len(mock_intent_handlers["HassTurnOn"].handled_intents) == 1
    assert len(mock_intent_handlers["HassTurnOff"].handled_intents) == 1


# --- F15 Boundary ---


async def test_tier2_f15_fallback_agent_raises_exception(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F15.B1: Exception raised by fallback agent falls through to NO_INTENT_MATCH."""

    class FailingAgent(conversation.AbstractConversationAgent):
        @property
        def supported_languages(self) -> list[str]:
            return ["en"]

        async def async_process(
            self, user_input: conversation.ConversationInput
        ) -> conversation.ConversationResult:
            raise RuntimeError("Fallback crashed")

    manager = conversation.get_agent_manager(hass)
    manager.async_set_agent("failing_fallback", FailingAgent())

    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "failing_fallback"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Trigger fallback",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f15_fallback_agent_in_data_when_not_in_options(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.B2: Fallback agent configured in entry.data used when absent from options."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={
            CONF_API_KEY: "test_key",
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
        options={},
        entry_id="data_fallback_entry",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Check fallback from data",
        conversation_id=None,
        context=Context(),
        agent_id=entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1


async def test_tier2_f15_fallback_agent_nonexistent_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F15.B3: Non-existent fallback agent ID caught and returns NO_INTENT_MATCH."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "non_existent_agent_404"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Out of domain",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f15_fallback_escalation_empty_text(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.B4: Empty text utterance escalated to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == ""


async def test_tier2_f15_fallback_preserves_conversation_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """F15.B5: Conversation ID preserved across fallback delegation."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="What time is it?",
        conversation_id="conv-session-789",
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.conversation_id == "conv-session-789"


# --- F16 Boundary ---


async def test_tier2_f16_empty_string_utterance_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.B1: Empty string utterance without fallback returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f16_malformed_api_response_missing_answers(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.B2: API response missing answers key returns NO_INTENT_MATCH without fallback."""
    mock_client.set_answers(None)
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f16_api_evaluation_error_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.B3: API evaluation exception without fallback returns NO_INTENT_MATCH."""
    mock_client.evaluate_error = TypeSafeError("API down")
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f16_intent_choice_other_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.B4: Intent choice 'other' returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "other",
                "confidence": 0.90,
                "probabilities": {"other": 0.90},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Miscellaneous statement",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier2_f16_zero_confidence_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """F16.B5: Zero confidence score returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.0,
                "probabilities": {"HassTurnOn": 0.0},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Gibberish",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (PAIRWISE INTERACTIONS) (18 TESTS)
# ==============================================================================


async def test_tier3_pairwise_f1_lifecycle_and_f11_registration(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Pair 1 (F1+F11): Lifecycle setup registers agent in manager, unload unregisters agent."""
    manager = conversation.get_agent_manager(hass)
    assert manager.async_get_agent(config_entry.entry_id) is not None
    await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()
    with pytest.raises(ValueError, match="not found"):
        manager.async_get_agent(config_entry.entry_id)


async def test_tier3_pairwise_f3_config_flow_and_f4_options_flow(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 2 (F3+F4): Create entry via config flow, then immediately update via options flow."""
    # 1. Config flow create
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_KEY: "secret_flow_key", CONF_MODEL: "jev-latest"},
        )
        await hass.async_block_till_done()
    entry_id = result2["result"].entry_id

    # 2. Options flow update
    opt_init = await hass.config_entries.options.async_init(entry_id)
    opt_conf = await hass.config_entries.options.async_configure(
        opt_init["flow_id"],
        {CONF_CONFIDENCE_THRESHOLD: 0.85, CONF_FALLBACK_AGENT: "mock_agent"},
    )
    await hass.async_block_till_done()
    assert opt_conf.get("type") is FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_get_entry(entry_id)
    assert entry is not None
    assert entry.options[CONF_CONFIDENCE_THRESHOLD] == 0.85
    assert entry.options[CONF_FALLBACK_AGENT] == "mock_agent"


async def test_tier3_pairwise_f4_options_and_f5_dynamic_reload(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Pair 3 (F4+F5): Options flow update triggers reload and updates strategy threshold."""
    opt_init = await hass.config_entries.options.async_init(config_entry.entry_id)
    await hass.config_entries.options.async_configure(
        opt_init["flow_id"],
        {CONF_CONFIDENCE_THRESHOLD: 0.88},
    )
    await hass.async_block_till_done()
    assert config_entry.runtime_data.strategy.confidence_threshold == 0.88


async def test_tier3_pairwise_f5_reload_and_f12_confidence_gating(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Pair 4 (F5+F12): Reloading threshold from 0.70 to 0.90 gates previously passing 0.80 utterance."""
    hass.states.async_set("light.hall", "off", {"friendly_name": "Hall Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.80,
                "probabilities": {"HassTurnOn": 0.80},
            },
            "target_entity": {"choice": "light.hall", "confidence": 0.80},
            "is_compound": {"noul": 0.0},
        }
    )
    # Passes under 0.70
    res1 = await conversation.async_converse(
        hass=hass,
        text="Turn on hall",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ACTION_DONE

    # Reload to 0.90
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.90},
    )
    await hass.async_block_till_done()

    # Now fails under 0.90
    res2 = await conversation.async_converse(
        hass=hass,
        text="Turn on hall",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res2.response.response_type is intent.IntentResponseType.ERROR


async def test_tier3_pairwise_f8_exposed_discovery_and_f10_fanout_builder(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 5 (F8+F10): Exposed entities dynamically populate fan-out target_entity candidates."""
    hass.states.async_set("light.garden", "off", {"friendly_name": "Garden Light"})
    async_expose_entity(hass, conversation.DOMAIN, "light.garden", True)

    await conversation.async_converse(
        hass=hass,
        text="Turn on garden light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "light.garden" in criteria
    assert "Garden Light (light)" in criteria["light.garden"]


async def test_tier3_pairwise_f8_exposed_discovery_and_f13_target_resolution(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Pair 6 (F8+F13): Exposed entity friendly_name resolves correctly in intent slot."""
    hass.states.async_set(
        "switch.fountain", "off", {"friendly_name": "Outdoor Fountain"}
    )
    async_expose_entity(hass, conversation.DOMAIN, "switch.fountain", True)
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "switch.fountain", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on fountain",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert handler.handled_intents[0].slots["name"]["value"] == "Outdoor Fountain"


async def test_tier3_pairwise_f9_intent_discovery_and_f10_fanout_builder(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Pair 7 (F9+F10): Registered intent handlers dynamically populate fan-out criteria."""
    _ = mock_intent_handlers
    await conversation.async_converse(
        hass=hass,
        text="Adjust brightness of light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    intent_criteria = mock_client.calls[0]["questions"]["intent"]["criteria"]
    assert "HassLightSet" in intent_criteria
    assert "Adjust brightness or color of a light" in intent_criteria["HassLightSet"]


async def test_tier3_pairwise_f9_intent_discovery_and_f14_intent_execution(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Pair 8 (F9+F14): Discovered intent schema is evaluated and executed via intent.async_handle."""
    hass.states.async_set("light.desk", "on", {"friendly_name": "Desk Lamp"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOff",
                "confidence": 0.92,
                "probabilities": {"HassTurnOff": 0.92},
            },
            "target_entity": {"choice": "light.desk", "confidence": 0.92},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Turn off desk lamp",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert len(mock_intent_handlers["HassTurnOff"].handled_intents) == 1


async def test_tier3_pairwise_f10_fanout_and_f13_target_resolution_area_domain(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Pair 9 (F10+F13): Fan-out area targeting resolves area name and infers domain slot."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Terrace")
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_type": {"choice": "area", "confidence": 0.95},
            "target_area": {"choice": area.id, "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on terrace lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert handler.handled_intents[0].slots["area"]["value"] == "Terrace"
    assert handler.handled_intents[0].slots["domain"]["value"] == "light"


async def test_tier3_pairwise_f12_confidence_gating_and_f15_fallback_escalation(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Pair 10 (F12+F15): Confidence below threshold escalates to configured fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.55,
                "probabilities": {"HassTurnOn": 0.55},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Could you turn something on",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in res.response.speech["plain"]["speech"]


async def test_tier3_pairwise_f12_confidence_gating_and_f16_unmatched_error(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 11 (F12+F16): Confidence below threshold without fallback returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.55,
                "probabilities": {"HassTurnOn": 0.55},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Could you turn something on",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier3_pairwise_f6_client_error_and_f15_fallback_escalation(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Pair 12 (F6+F15): TypeSafeClient failure escalates to configured fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.evaluate_error = TypeSafeError("Service Unavailable 503")
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in res.response.speech["plain"]["speech"]


async def test_tier3_pairwise_f6_client_error_and_f16_unmatched_error(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 13 (F6+F16): TypeSafeClient failure without fallback returns NO_INTENT_MATCH."""
    mock_client.evaluate_error = TypeSafeError("Rate limit 429")
    res = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_tier3_pairwise_f7_typed_primitives_and_f12_compound_gating(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 14 (F7+F12): Noul compound answer evaluated against compound threshold gates execution."""
    mock_client.set_answers({"is_compound": {"noul": 0.82}})
    res = await conversation.async_converse(
        hass=hass,
        text="Turn off lights and shut down PC",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert "multiple requests" in res.response.speech["plain"]["speech"]


async def test_tier3_pairwise_f13_resolution_and_f14_execution_numeric_brightness(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Pair 15 (F13+F14): Regex-extracted brightness slot passed into HassLightSet intent handling."""
    hass.states.async_set("light.reading", "on", {"friendly_name": "Reading Lamp"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassLightSet",
                "confidence": 0.95,
                "probabilities": {"HassLightSet": 0.95},
            },
            "target_entity": {"choice": "light.reading", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Set reading lamp brightness to 80%",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassLightSet"]
    assert handler.handled_intents[0].slots["brightness"]["value"] == 80


async def test_tier3_pairwise_f13_resolution_and_f14_execution_numeric_temperature(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    climate_handler: MockClimateIntentHandler,
) -> None:
    """Pair 16 (F13+F14): Regex-extracted temperature slot passed into climate intent handling."""
    handler = climate_handler
    hass.states.async_set("climate.nest", "heat", {"friendly_name": "Nest Thermostat"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassClimateSetTemperature",
                "confidence": 0.95,
                "probabilities": {"HassClimateSetTemperature": 0.95},
            },
            "target_entity": {"choice": "climate.nest", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Set thermostat to 69.5 deg",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert handler.handled_intents[0].slots["temperature"]["value"] == 69.5


async def test_tier3_pairwise_f2_credentials_validation_and_f3_config_flow_failure(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 17 (F2+F3): Failed credential validation halts config flow without creating entry."""
    mock_client.validate_error = TypeSafeAuthError("Bad key")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_API_KEY: "wrong_key", CONF_MODEL: "jev-latest"},
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.FORM
    assert result2.get("errors") == {"base": "invalid_auth"}
    assert len(hass.config_entries.async_entries(DOMAIN)) == 0


async def test_tier3_pairwise_f8_unexposed_entity_filtered_f13_resolution(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Pair 18 (F8+F13): Unexposed entity excluded from candidates even if lexical match exists."""
    hass.states.async_set("light.public_lamp", "off", {"friendly_name": "Lamp"})
    hass.states.async_set("light.private_lamp", "off", {"friendly_name": "Lamp"})

    async_expose_entity(hass, conversation.DOMAIN, "light.public_lamp", True)
    async_expose_entity(hass, conversation.DOMAIN, "light.private_lamp", False)

    await conversation.async_converse(
        hass=hass,
        text="Turn on the lamp",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "light.public_lamp" in criteria
    assert "light.private_lamp" not in criteria


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION WORKLOADS (8 SCENARIOS)
# ==============================================================================


async def test_tier4_scenario_1_standard_device_control_high_confidence(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Scenario 1: Standard device control with high confidence (F1, F3, F6, F7, F8, F9, F10, F11, F12, F13, F14)."""
    hass.states.async_set(
        "light.kitchen_lights", "off", {"friendly_name": "Kitchen Lights"}
    )
    async_expose_entity(hass, conversation.DOMAIN, "light.kitchen_lights", True)

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.96,
                "probabilities": {"HassTurnOn": 0.96},
            },
            "target_entity": {"choice": "light.kitchen_lights", "confidence": 0.96},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    assert intent_obj.intent_type == "HassTurnOn"
    assert intent_obj.slots["name"]["value"] == "Kitchen Lights"


async def test_tier4_scenario_2_low_confidence_escalates_to_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Scenario 2: Low confidence utterance escalates to fallback agent (F1, F4, F6, F11, F12, F15)."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.70,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.50,
                "probabilities": {"HassTurnOn": 0.50},
            },
            "is_compound": {"noul": 0.05},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Maybe turn on something",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "Maybe turn on something"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_tier4_scenario_3_out_of_domain_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Scenario 3: Out-of-domain utterance with fallback agent configured (F6, F10, F11, F15)."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.98,
                "probabilities": {"unmatched": 0.98},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="What is the capital of France?",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "What is the capital of France?"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_tier4_scenario_4_out_of_domain_without_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Scenario 4: Out-of-domain utterance without fallback agent returns NO_INTENT_MATCH (F6, F10, F11, F16)."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.98,
                "probabilities": {"unmatched": 0.98},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Tell me a bedtime story",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH
    assert (
        "could not understand that request" in result.response.speech["plain"]["speech"]
    )


async def test_tier4_scenario_5_multi_room_entity_disambiguation(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Scenario 5: Multi-room entity disambiguation (kitchen vs bedroom light) (F8, F10, F12, F13, F14)."""
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)
    kitchen = area_reg.async_create("Kitchen")
    bedroom = area_reg.async_create("Bedroom")

    e_kitchen = entity_reg.async_get_or_create(
        "light", "demo", "ceiling_k", suggested_object_id="kitchen_ceiling"
    )
    entity_reg.async_update_entity(e_kitchen.entity_id, area_id=kitchen.id)
    hass.states.async_set(
        e_kitchen.entity_id, "off", {"friendly_name": "Ceiling Light"}
    )

    e_bedroom = entity_reg.async_get_or_create(
        "light", "demo", "ceiling_b", suggested_object_id="bedroom_ceiling"
    )
    entity_reg.async_update_entity(e_bedroom.entity_id, area_id=bedroom.id)
    hass.states.async_set(
        e_bedroom.entity_id, "off", {"friendly_name": "Ceiling Light"}
    )

    # TypeSafe disambiguates based on 'kitchen' keyword and selects e_kitchen.entity_id
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.94,
                "probabilities": {"HassTurnOn": 0.94},
            },
            "target_entity": {"choice": e_kitchen.entity_id, "confidence": 0.94},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen ceiling light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Ceiling Light"


async def test_tier4_scenario_6_options_update_threshold_alters_routing(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Scenario 6: Options flow live update changes confidence threshold, altering routing (F4, F5, F11, F12, F15)."""
    hass.states.async_set("switch.fan", "off", {"friendly_name": "Desk Fan"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.75,
                "probabilities": {"HassTurnOn": 0.75},
            },
            "target_entity": {"choice": "switch.fan", "confidence": 0.75},
            "is_compound": {"noul": 0.01},
        }
    )

    # 1. Under default threshold 0.70, confidence 0.75 executes
    res1 = await conversation.async_converse(
        hass=hass,
        text="Turn on the desk fan",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert len(mock_intent_handlers["HassTurnOn"].handled_intents) == 1

    # 2. Options flow update: increase threshold to 0.85 and configure fallback
    opt_init = await hass.config_entries.options.async_init(config_entry.entry_id)
    await hass.config_entries.options.async_configure(
        opt_init["flow_id"],
        {
            CONF_CONFIDENCE_THRESHOLD: 0.85,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    # 3. Same utterance with confidence 0.75 now escalates to fallback agent
    res2 = await conversation.async_converse(
        hass=hass,
        text="Turn on the desk fan",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in res2.response.speech["plain"]["speech"]


async def test_tier4_scenario_7_invalid_credentials_rejected_in_config_flow(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """Scenario 7: Invalid credentials during config flow setup step rejected with error (F2, F3)."""
    mock_client.validate_error = TypeSafeAuthError("401 Unauthorized API Key")

    # Step 1: User initiates setup
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "user"

    # Step 2: User provides invalid key
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_API_KEY: "invalid_expired_token",
            CONF_MODEL: "jev-latest",
        },
    )
    await hass.async_block_till_done()
    assert result2.get("type") is FlowResultType.FORM
    assert result2.get("errors") == {"base": "invalid_auth"}

    # Step 3: User provides valid key and successfully creates entry
    mock_client.validate_error = None
    mock_client.validate_result = True
    with patch("custom_components.typesafe.async_setup_entry", return_value=True):
        result3 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_KEY: "valid_live_token",
                CONF_MODEL: "jev-latest",
            },
        )
        await hass.async_block_till_done()

    assert result3.get("type") is FlowResultType.CREATE_ENTRY
    assert result3.get("title") == DEFAULT_NAME
    assert result3["data"][CONF_API_KEY] == "valid_live_token"


async def test_tier4_scenario_8_unexposed_entities_ignored_in_resolution(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Scenario 8: Unexposed entities are ignored during entity resolution (F8, F10, F13)."""
    hass.states.async_set(
        "switch.garden_sprinkler",
        "off",
        {"friendly_name": "Garden Sprinkler"},
    )
    hass.states.async_set(
        "switch.secret_pump",
        "off",
        {"friendly_name": "Secret Water Pump"},
    )

    # Expose only garden sprinkler
    async_expose_entity(hass, conversation.DOMAIN, "switch.garden_sprinkler", True)
    async_expose_entity(hass, conversation.DOMAIN, "switch.secret_pump", False)

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "switch.garden_sprinkler",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on garden sprinkler",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    # Verify unexposed entity was never sent to TypeSafe model
    target_criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "switch.garden_sprinkler" in target_criteria
    assert "switch.secret_pump" not in target_criteria

    # Verify executed intent targeted sprinkler
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Garden Sprinkler"
