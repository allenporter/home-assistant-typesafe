"""Stage 2: Candidate Retrieval.

This stage is responsible for searching and pruning Home Assistant's components
(intents, areas, and controllable entities) based on the ParsedRequest.

Principles:
1. Pure Search & Space Reduction: Bounds the search space from hundreds of entities down to a ranked candidate set without formatting prompts.
2. High Recall: Ensures the true target device, area, and intent are included in the top-N candidate lists.
3. Decoupled Strategies: Supports swapping retrieval implementations (e.g. Lexical with area boosting vs. Exhaustive 'return everything' vs. future semantic vector search) with identical downstream contracts.
"""
