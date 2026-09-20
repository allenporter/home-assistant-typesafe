"""Stage 5: Target Resolution & Decision Generation.

This stage takes the prediction answers, evaluates confidence thresholds and compound
detection gating, resolves target entities vs. room broadcasts, and binds extracted
numeric literals (e.g. percentages, temperatures) to the appropriate Home Assistant slots.

Principles:
1. Gated Intent Resolution: Immediately escalates compound commands or low-confidence predictions to fallback agents.
2. Target Disambiguation: Resolves whether an action targets an individual entity, an area broadcast, or a domain group.
3. Domain-Conditioned Slot Binding: Binds syntactic numbers (e.g. 50%) to intent-specific slots (e.g. brightness for light, percentage for fan, humidity for humidifier) only after the target domain is resolved.
"""
