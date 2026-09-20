"""Speculative decision architecture for Home Assistant.

This package orchestrates the 5-stage decision flow for Home Assistant voice
and text commands inspired by modern retrieval-augmented generation (RAG) pipelines:

- request: Syntactic preprocessing, token normalization, extracting raw numbers/temperatures/percentages
  without premature semantic binding, and extracting originating satellite room context (e.g., parsing
  "Set kitchen light to 50%" into tokens and raw value 50% before deciding whether it means brightness or fan speed).
- retrieval: Reading state and registries from Home Assistant (entity registry, area registry, and live device states)
  to rapidly filter candidate intents, target entities, and areas from the entire home into a focused shortlist
  (e.g., pruning 100+ entities across the home down to candidate lights and media players matching "kitchen").
- hydration: Enriching the shortlisted candidates with readable Home Assistant metadata and formulating
  canonical TypeSafe System One questions (e.g., formatting raw entity IDs like `light.kitchen_ceiling`
  into human-readable choices like "Kitchen Ceiling (light) in Kitchen" so the model has clear semantic context).
- scoring: Forward evaluation pass through the TypeSafe DecisionEngine (e.g., executing the System One inference
  call to score candidate choices, confidences, and probabilities).
- resolution: Reviewing the raw scores, picking the winner of which intent, entity, or area to execute,
  gating by confidence thresholds, deciding whether to escalate to fallback agents, and binding domain-specific
  slots (e.g., selecting HassLightSet and light.kitchen_ceiling, then binding the raw 50% into brightness: 50).

Components and consumers should import directly from the specific stage or subpackage modules
(e.g., flow, request, retrieval, hydration, scoring, or resolution).
"""
