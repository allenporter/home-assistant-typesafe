# E2E Test Infra: home-assistant-typesafe

## Test Philosophy

- Opaque-box, requirement-driven. No dependency on internal implementation details.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.
- 100% hermetic: No live network calls, no external API keys required.

## Feature Inventory & Mapping

| #   | Feature                                             | Source (Requirement) | Tier 1 | Tier 2 | Tier 3 |
| --- | --------------------------------------------------- | -------------------- | :----: | :----: | :----: |
| 1   | Component Manifest & Lifecycle                      | ORIGINAL_REQUEST §R1 |   5    |   5    |   ✓    |
| 2   | Credentials Validation (`GET /v1/models`)           | ORIGINAL_REQUEST §AC |   5    |   5    |   ✓    |
| 3   | Config Flow (api_key, model)                        | ORIGINAL_REQUEST §R1 |   5    |   5    |   ✓    |
| 4   | Options Flow (threshold, fallback)                  | ORIGINAL_REQUEST §R1 |   5    |   5    |   ✓    |
| 5   | Dynamic Reload on Options Update                    | ORIGINAL_REQUEST §R1 |   5    |   5    |   ✓    |
| 6   | TypeSafe System One HTTP Client                     | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 7   | Typed Primitives (Choice/Noul)                      | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 8   | Exposed Entity Discovery                            | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 9   | Dynamic Intent Schema Discovery                     | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 10  | Speculative Fan-out Query Builder                   | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 11  | Conversation Entity Registration                    | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 12  | Confidence Threshold Gating                         | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 13  | Entity Target Resolution                            | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 14  | HA Intent Execution (`intent.async_handle`)         | ORIGINAL_REQUEST §R2 |   5    |   5    |   ✓    |
| 15  | Fallback Escalation (`conversation.async_converse`) | ORIGINAL_REQUEST §R3 |   5    |   5    |   ✓    |
| 16  | Unmatched Error Handling (`NO_INTENT_MATCH`)        | ORIGINAL_REQUEST §R3 |   5    |   5    |   ✓    |

## Test Architecture

- Test runner: `./script/test` (`uv run --no-project pytest "$@"`)
- Test suites:
  - `tests/test_config_flow.py`: Unit and integration tests for Config Flow and Options Flow.
  - `tests/test_init.py`: Integration setup, lifecycle, unload, reload on options update.
  - `tests/test_conversation.py`: Conversation entity processing, intent resolution, execution, fallback.
  - `tests/test_e2e_conversation.py`: End-to-end multi-tier conversational and routing test scenarios.
- Hermetic fixtures in `tests/conftest.py`:
  - `MockTypeSafeClient` / responses for `/v1/models` and `/v1/systemone`.
  - Registered dummy intent handlers (`HassTurnOn`, `HassTurnOff`, `HassLightSet`).
  - Mock secondary conversation agent for fallback testing.

## Real-World Application Scenarios (Tier 4)

| #   | Scenario                                                                           | Features Exercised                              | Complexity |
| --- | ---------------------------------------------------------------------------------- | ----------------------------------------------- | ---------- |
| 1   | Standard device control with high confidence                                       | F1, F3, F6, F7, F8, F9, F10, F11, F12, F13, F14 | Medium     |
| 2   | Low confidence utterance escalates to fallback agent                               | F1, F4, F6, F11, F12, F15                       | Medium     |
| 3   | Out-of-domain utterance with fallback agent configured                             | F6, F10, F11, F15                               | Medium     |
| 4   | Out-of-domain utterance without fallback agent (returns NO_INTENT_MATCH)           | F6, F10, F11, F16                               | Medium     |
| 5   | Multi-room entity disambiguation (kitchen vs bedroom light)                        | F8, F10, F12, F13, F14                          | High       |
| 6   | Options flow live update changes confidence threshold, altering subsequent routing | F4, F5, F11, F12, F15                           | High       |
| 7   | Invalid credentials during config flow setup step rejected with error              | F2, F3                                          | Low        |
| 8   | Unexposed entities are ignored during entity resolution                            | F8, F10, F13                                    | Medium     |

## Coverage Thresholds

- Tier 1: ≥5 per feature (80 cases)
- Tier 2: ≥5 boundary/edge cases per feature (80 cases)
- Tier 3: Pairwise combinations across major features (≥16 cases)
- Tier 4: Real-world application workloads (≥8 cases)
- Total: ≥184 test assertions across test suite
