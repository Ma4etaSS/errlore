"""errlore.trust -- adaptive per-model trust weights for LLM ensembles."""

from errlore.trust.engine import (
    DEFAULT_WEIGHT,
    BetaPrior,
    FeedbackSignal,
    TrustEngine,
    enforce_entropy,
)

__all__ = [
    "DEFAULT_WEIGHT",
    "BetaPrior",
    "FeedbackSignal",
    "TrustEngine",
    "enforce_entropy",
]
