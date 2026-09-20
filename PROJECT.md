# Project: Home Assistant TypeSafe System One Integration (`custom_components/typesafe`)

## Architecture

- **Integration Domain**: `custom_components/typesafe`
- **AI Core**: TypeSafe AI System One API (Jev model family, default `jev-latest`)
- **Interaction Model**: Speculative fan-out evaluating structured typed questions (`Choice`, `Noul`, `Score`) in parallel over user utterances.
- **Entity Discovery**: Dynamic exposure filtering using `async_should_expose(hass, conversation.DOMAIN, entity_id)` across controllable domains (`light`, `switch`, `climate`, `cover`, `media_player`, `lock`, etc.).
- **Intent Dispatch**: Introspects registered Home Assistant intent schemas, builds intent candidates, matches high-confidence intents, maps slots, and executes via `intent.async_handle`.
- **Fallback Escalation**: Confidence-gated escalation delegating low-confidence (< threshold) or out-of-domain utterances to `fallback_agent` via `conversation.async_converse`.
- **Hermetic Testing**: Zero external network or live credentials in unit/integration tests; uses deterministic fake clients and mock conversation agents.

## Feature Inventory

| #   | Feature                              | Description                                                                                                                   | Milestone | Source                       |
| --- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | --------- | ---------------------------- |
| 1   | Component Manifest & Setup           | `manifest.json`, `const.py`, `models.py`, `__init__.py` lifecycle and Platform.CONVERSATION setup                             | M1        | Survey (Laya/Jev)            |
| 2   | Credentials Validation               | Validate TypeSafe API key via `GET /v1/models` in config flow                                                                 | M1        | Survey (TypeSafe Spec)       |
| 3   | Config Flow                          | Setup step configuring `api_key` and optional `model` (default `jev-latest`)                                                  | M1        | ORIGINAL_REQUEST §R1         |
| 4   | Options Flow                         | Configurable `confidence_threshold` (slider) and `fallback_agent` selector                                                    | M1        | ORIGINAL_REQUEST §R1         |
| 5   | Dynamic Reload Listener              | Automatically reload config entry when options are updated                                                                    | M1        | Survey (Laya)                |
| 6   | TypeSafe HTTP Client                 | Asynchronous HTTP client for `POST /v1/systemone` using HA `aiohttp_client` session                                           | M1        | Survey (TypeSafe Spec)       |
| 7   | Typed Primitives                     | Structured representations for `ChoiceQuestion`, `NoulQuestion`, `ChoiceAnswer`, `NoulAnswer`                                 | M1        | Survey (TypeSafe Spec)       |
| 8   | Exposed Entity Discovery             | Discover exposed entities via `async_should_expose`, format candidate descriptions and area mapping                           | M1        | Survey (Laya)                |
| 9   | Dynamic Intent Discovery             | Introspect HA intent registry (`intent.async_get`), filter supported slots and descriptions                                   | M1        | Survey (Laya)                |
| 10  | Speculative Fan-out Question Builder | Construct parallel `intent` and `target_entity` questions for single forward pass                                             | M1        | Survey (TypeSafe Spec)       |
| 11  | Conversation Entity Registration     | `TypeSafeConversationEntity` subclassing `ConversationEntity` & `AbstractConversationAgent`, registered via `async_set_agent` | M1        | ORIGINAL_REQUEST §R2         |
| 12  | Confidence Threshold Gating          | Gate decisions on `confidence >= confidence_threshold` and non-fallback choice                                                | M1        | ORIGINAL_REQUEST §R2, §R3    |
| 13  | Entity Target Resolution             | Map predicted target entity ID to friendly name and format HA intent slots                                                    | M1        | Survey (Laya)                |
| 14  | HA Intent Execution                  | Dispatch matched intent to `intent.async_handle` and return formatted intent result                                           | M1        | ORIGINAL_REQUEST §R2         |
| 15  | Fallback Escalation                  | Escalate to secondary agent via `conversation.async_converse` on low confidence or unmatched intent                           | M1        | ORIGINAL_REQUEST §R3         |
| 16  | Unmatched Error Handling             | Return `NO_INTENT_MATCH` error response when fallback agent is not configured                                                 | M1        | Survey (Laya)                |
| 17  | Hermetic Test Scaffolding            | Deterministic mock engine/client and test fixtures in `tests/conftest.py`                                                     | M1        | ORIGINAL_REQUEST §Guardrails |
| 18  | E2E Test Suite (Tiers 1-4)           | Comprehensive opaque-box test suite covering features, boundaries, combinations, and scenarios; publish `TEST_READY.md`       | M2        | Project Pattern              |
| 19  | 100% Passing Tests & Lint            | `./script/test` 100% pass, `./script/lint` 100% pass across all 10 linters                                                    | M3        | ORIGINAL_REQUEST §Guardrails |
| 20  | Adversarial Coverage Hardening       | Stress testing edge cases, rate limits, timeouts, and ambiguous utterances (Tier 5)                                           | M4        | Project Pattern              |

## Milestones

| #   | Name                                         | Scope                                                                                                                                                                                     | Dependencies | Status |
| --- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------ |
| M1  | Core Integration Implementation              | All integration modules (`manifest.json`, `const.py`, `models.py`, `__init__.py`, `config_flow.py`, `client.py`, `strategy.py`, `conversation.py`) and unit/adversarial tests in `tests/` | none         | DONE   |
| M2  | Requirements-Driven E2E Test Suite           | Comprehensive opaque-box test suite (Tiers 1-4) in `tests/test_e2e_conversation.py`, publish `TEST_READY.md`                                                                              | M1           | DONE   |
| M3  | Full Integration Verification & Quality Gate | Pass 100% of E2E test suite, `./script/test` 100% pass, `./script/lint` clean pass                                                                                                        | M1, M2       | DONE   |
| M4  | Adversarial Coverage Hardening               | Tier 5 adversarial testing: edge cases, timeouts, ambiguous entities, error recovery                                                                                                      | M3           | DONE   |

## Interface Contracts

### Configuration & Data Model (`models.py`, `const.py`)

- `CONF_API_KEY = "api_key"`
- `CONF_MODEL = "model"`
- `CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"`
- `CONF_FALLBACK_AGENT = "fallback_agent"`
- `DEFAULT_MODEL = "jev-latest"`
- `DEFAULT_CONFIDENCE_THRESHOLD = 0.7`
- `type TypeSafeConfigEntry = ConfigEntry[TypeSafeData]`
- `dataclass TypeSafeData`: holds `client: TypeSafeClient`, `strategy: DecisionStrategy`

### TypeSafe Client (`client.py`)

- `class TypeSafeClient`:
  - `async def async_validate_key() -> bool`: GET `/v1/models`, returns True if 200 OK, raises `AuthenticationError` on 401.
  - `async def async_evaluate(state: str | dict, questions: dict[str, dict]) -> dict[str, Any]`: POST `/v1/systemone`, returns dict containing `answers`.

### Decision Strategy (`strategy.py`)

- `class DecisionStrategy`:
  - `async def async_decide(client: TypeSafeClient, utterance: str, context: DecisionContext) -> DecisionResult`
- `dataclass DecisionResult`:
  - `intent_name: str | None`
  - `entity_id: str | None`
  - `confidence: float`
  - `should_escalate: bool`

### Conversation Entity (`conversation.py`)

- Subclasses `(conversation.ConversationEntity, conversation.AbstractConversationAgent)`
- `async def _async_handle_message(user_input: ConversationInput, chat_log: ChatLog) -> ConversationResult`

## Code Layout

- `custom_components/typesafe/__init__.py`: Component lifecycle, config entry setup, platform forwarding, reload listener.
- `custom_components/typesafe/manifest.json`: Integration metadata, dependencies (`["conversation", "intent"]`).
- `custom_components/typesafe/const.py`: Constants, domain, configuration keys, defaults.
- `custom_components/typesafe/models.py`: Dataclasses, type aliases (`TypeSafeData`, `TypeSafeConfigEntry`).
- `custom_components/typesafe/config_flow.py`: `TypeSafeConfigFlow` and `TypeSafeOptionsFlowHandler`.
- `custom_components/typesafe/client.py`: Asynchronous TypeSafe System One client.
- `custom_components/typesafe/strategy.py`: Typed questions builder, entity/intent discovery, decision parsing.
- `custom_components/typesafe/conversation.py`: `TypeSafeConversationEntity` implementation with ChatLog integration.
- `tests/conftest.py`: Fixtures, fake client, mock fallback agent, dummy intent handlers.
- `tests/test_config_flow.py`: Unit tests for config flow and options flow.
- `tests/test_init.py`: Setup and unload lifecycle tests.
- `tests/test_conversation.py`: Unit tests for conversation entity, routing, and fallback.
- `tests/test_client.py`: Unit tests for TypeSafeClient HTTP handling.
- `tests/test_adversarial.py`: Adversarial edge case tests.
- `tests/test_adversarial_m1.py`: Defensive error handling tests.
- `tests/test_iter2_challenger_stress.py`: Empirical stress tests across API and options reload.
- `tests/test_e2e_conversation.py`: Full requirements-driven opaque-box E2E test suite (Tiers 1-4).
