# Changelog

[0.1.1] - 2025-06-05

## Added
- TokenLevelProbabilityProvider: Uses LLM token-level probabilities for grammar rule scoring.
- TokenLevelViterbiParser: Viterbi parser that interpolates between PCFG and LLM token-level probabilities.
- Redundancy penalty in LLM probability calculation to discourage unary chains (e.g., NP -> NP).
- Precomputation of span probabilities for all possible text spans before parsing.
- Integration with LocalLLM for efficient local inference.

## Changed
- Viterbi parsing logic to interpolate between grammar and LLM probabilities using a configurable theta parameter.

## Fixed
- Reduced excessive unary and nested constituents in output trees.
