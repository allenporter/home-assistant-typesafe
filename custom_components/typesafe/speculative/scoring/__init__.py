"""Stage 4: Scoring & Engine Evaluation.

This stage takes the HydratedPayload containing the canonical System One questions
and state dictionary, evaluates them against a DecisionEngine in a parallel forward pass,
and returns the typed PredictionResult.

Principles:
1. Model & Engine Decoupling: Scoring evaluates questions against the DecisionEngine protocol, supporting live models (Jev), mock adapters, or deterministic test doubles.
2. Direct Probability Output: Produces calibrated probabilities and typed answers without post-processing or business logic.
3. Observability: Captures model metadata and token usage statistics alongside raw answers.
"""
