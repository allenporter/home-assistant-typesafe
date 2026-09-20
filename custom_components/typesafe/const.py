"""Constants for the TypeSafe integration."""

from typing import Final

DOMAIN: Final = "typesafe"

# Configuration options
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_CONFIDENCE_THRESHOLD: Final = "confidence_threshold"
CONF_FALLBACK_AGENT: Final = "fallback_agent"

# Defaults
DEFAULT_MODEL: Final = "jev-latest"
DEFAULT_CONFIDENCE_THRESHOLD: Final = 0.7
DEFAULT_NAME: Final = "TypeSafe"

# API endpoints
API_BASE_URL: Final = "https://api.typesafe.ai"
DEFAULT_TIMEOUT: Final = 10.0
