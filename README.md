# Home Assistant TypeSafe Integration

A Home Assistant custom conversation integration powered by **TypeSafe AI System One** (Jev model family) for fast, structured intent routing and device control.

## Overview

- **Speculative Fan-out**: Evaluates intent and entity candidates in parallel using structured TypeSafe primitives (`Choice`, `Noul`, `Score`).
- **Dynamic Discovery**: Discovers exposed entities and registered intent schemas in Home Assistant, performing candidate pruning and lexical ranking against the current home context.
- **Intent Execution**: Dispatches high-confidence matched actions directly to Home Assistant's native `intent.async_handle`.
- **Confidence-Gated Fallback**: Gracefully delegates low-confidence (< threshold) or out-of-domain utterances to a secondary fallback conversation agent (e.g., Assist / LLM).

## Configuration

Set up the integration through **Settings > Devices & Services > Add Integration > TypeSafe**:

- **API Key** (`api_key`): Your TypeSafe API key (validated against `/v1/models`).
- **Model** (`model`): TypeSafe model to query (defaults to `jev-latest`).

### Options

Access integration options to tune runtime behavior:

- **Confidence Threshold** (`confidence_threshold`): Confidence cutoff (0.0 – 1.0, default `0.7`) below which requests are escalated to fallback.
- **Fallback Agent** (`fallback_agent`): Secondary conversation agent ID to handle queries that cannot be resolved with high confidence.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local environment setup, running tests, and linting guidelines.
