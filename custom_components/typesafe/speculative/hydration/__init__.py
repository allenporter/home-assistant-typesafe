"""Stage 3: Fetch Content & Hydration.

This stage takes the top-N retrieved candidate IDs, fetches and formats their
Home Assistant semantic attributes and friendly descriptions, and constructs
the canonical System One question primitives (ChoiceQuestion, NoulQuestion,
ScoreQuestion) for model evaluation.

Principles:
1. Content Fetch & Hydration: Inflates candidate identifiers into human-readable descriptions and semantic criteria.
2. Canonical Type Construction: Forms structured System One questions with unambiguous task instructions.
3. Decoupled Model Payloads: Formulates the questions and state payload independently of how scoring or inference is executed.
"""
