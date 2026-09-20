"""Stage 1: Request Processing and Normalization.

This stage is responsible for ingesting the user's raw utterance and device context,
normalizing the input text, extracting lexical tokens, deterministically parsing raw
numeric tokens (percentages, temperatures, numbers), and resolving any satellite
device room priors.

Principles:
1. Separation of Concerns: Analyzes syntax and device context before touching smart home entity catalogs.
2. Deterministic Syntax Parsing: Parses explicit numbers/percentages directly, leaving semantic slot binding to later stages.
3. Zero Domain Bias: Avoids hardcoded domain filtering or premature intent assumptions.
"""
