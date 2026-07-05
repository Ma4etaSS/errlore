"""TrustEngine -- adaptive per-model trust weights with Bayesian cold start.

Maintains dynamic trust weights for an ensemble of models (or any scored
entities) using logit-space updates with adaptive learning rate, volatility
damping, domain-specific EMA bias, Beta-prior cold start, entropy
enforcement, and temporal decay towards neutral.

Key formulas:
    logit_delta = lr * ((outcome - 0.5) * 2 + disagreement) + domain_bias
    w_new = sigmoid(logit(w_old) + logit_delta)

Outcome is centered around 0.5 so that values below 0.5 DECREASE the weight
and values above 0.5 INCREASE it. This prevents the monotonic-growth bug
where all models converge to the cap regardless of actual quality.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("errlore.trust")

DEFAULT_WEIGHT: float = 0.5
"""Default weight assigned to a model with no history."""


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-7) -> float:
    """Inverse sigmoid (log-odds)."""
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Public dataclass: FeedbackSignal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeedbackSignal:
    """Generic quality signal for a model observation.

    Attributes:
        outcome: Quality score in [0, 1]. Values below 0.5 decrease trust,
                 above 0.5 increase it.
        domain: Task domain (e.g. "code_generation", "research").
        weight: Importance multiplier for this observation (default 1.0).
        disagreement: Signed disagreement bonus/penalty in [-1, 1].
                      Positive = model was contrarian and correct (boost).
                      Negative = model deviated and was wrong (penalty).
    """

    outcome: float
    domain: str = "general"
    weight: float = 1.0
    disagreement: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.outcome <= 1.0:
            raise ValueError(f"outcome must be in [0, 1], got {self.outcome}")
        if not -1.0 <= self.disagreement <= 1.0:
            raise ValueError(f"disagreement must be in [-1, 1], got {self.disagreement}")
        if self.weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")


# ---------------------------------------------------------------------------
# BetaPrior for cold start
# ---------------------------------------------------------------------------


@dataclass
class BetaPrior:
    """Bayesian Beta prior for cold-start blending."""

    alpha: float = 1.0
    beta_param: float = 1.0

    def update(self, success: float) -> None:
        """Record an observation (success in [0, 1])."""
        self.alpha += success
        self.beta_param += 1.0 - success

    @property
    def mean(self) -> float:
        """Posterior mean."""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def observations(self) -> float:
        """Number of effective observations (excluding the uniform prior)."""
        return (self.alpha + self.beta_param) - 2.0

    def is_cold(self, threshold: int = 15) -> bool:
        """True if insufficient observations to trust the logit update alone."""
        return self.observations < threshold

    def to_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta_param": self.beta_param}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BetaPrior:
        return cls(alpha=d["alpha"], beta_param=d["beta_param"])


def _init_prior(static_weight: float) -> BetaPrior:
    """Create a prior skewed towards the given static weight."""
    pseudo_obs = 5.0
    return BetaPrior(
        alpha=1.0 + pseudo_obs * static_weight,
        beta_param=1.0 + pseudo_obs * (1.0 - static_weight),
    )


# ---------------------------------------------------------------------------
# Adaptive learning rate
# ---------------------------------------------------------------------------


def _compute_adaptive_lr(
    task_count: int,
    weight_history: list[float],
    initial_lr: float = 0.15,
    min_lr: float = 0.02,
) -> float:
    """LR decays with log(task_count), damped by recent volatility."""
    k = 0.3
    decayed_lr = initial_lr / (1.0 + k * math.log(1.0 + task_count))
    if len(weight_history) >= 3:
        recent = weight_history[-10:]
        mean = sum(recent) / len(recent)
        variance = sum((w - mean) ** 2 for w in recent) / len(recent)
        volatility_factor = 1.0 / (1.0 + 5.0 * math.sqrt(variance))
    else:
        volatility_factor = 1.0
    return max(min_lr, decayed_lr * volatility_factor)


# ---------------------------------------------------------------------------
# Domain bias (EMA)
# ---------------------------------------------------------------------------


def _compute_domain_bias(
    model_id: str,
    domain: str,
    history: list[dict[str, Any]],
    half_life_hours: float = 24.0,
) -> float:
    """Time-weighted domain bias from historical outcome quality.

    Computes a recency-weighted average outcome for the model in the given
    domain, then converts deviation from 0.5 into a small bias term.
    """
    now = time.time()
    half_life_sec = half_life_hours * 3600.0
    relevant = [
        h for h in history
        if h.get("model") == model_id and h.get("domain", "general") == domain
    ]
    if len(relevant) < 3:
        return 0.0
    total_weight = 0.0
    weighted_outcome = 0.0
    for h in relevant:
        ts = h.get("timestamp", now)
        recency = math.exp(-max(0.0, now - ts) * math.log(2) / half_life_sec)
        outcome = h.get("outcome", 0.5)
        weighted_outcome += recency * outcome
        total_weight += recency
    if total_weight < 1e-9:
        return 0.0
    avg_outcome = weighted_outcome / total_weight
    bias = (avg_outcome - 0.5) * 0.20
    return max(-0.10, min(0.10, bias))


# ---------------------------------------------------------------------------
# Entropy enforcement
# ---------------------------------------------------------------------------


def enforce_entropy(
    weights: dict[str, float],
    min_weight: float = 0.1,
    entropy_threshold: float = 0.7,
) -> dict[str, float]:
    """Prevent collapse to a single model by enforcing minimum entropy.

    When normalized entropy drops below the threshold, weights are blended
    towards uniform distribution.
    """
    n = len(weights)
    if n < 2:
        return dict(weights)
    total = sum(weights.values())
    if total < 1e-9:
        return dict.fromkeys(weights, 1.0 / n)
    probs = {m: w / total for m, w in weights.items()}
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 1e-12)
    max_entropy = math.log(n)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    result = dict(weights)
    for mid in result:
        if result[mid] < min_weight:
            result[mid] = min_weight

    if norm_entropy < entropy_threshold:
        uniform = 1.0 / n
        alpha = 0.3 * (1.0 - norm_entropy / entropy_threshold)
        for mid in result:
            result[mid] = (1.0 - alpha) * result[mid] + alpha * uniform

    return result


# ---------------------------------------------------------------------------
# TrustEngine
# ---------------------------------------------------------------------------


@dataclass
class TrustEngine:
    """Adaptive trust-weight engine for model ensembles.

    Args:
        domains: Tuple of recognized domain names.
        initial_lr: Starting learning rate for logit updates.
        min_lr: Floor for the adaptive learning rate.
        cap: Maximum allowed weight (hard ceiling).
        floor: Minimum allowed weight (hard floor).
        entropy_threshold: Normalized entropy below which regularization kicks in.
        cold_start_threshold: Number of observations before prior blending fades.
        state_path: Path to JSON persistence file. None = in-memory only.
        lr_multipliers: Per-model LR multiplier (e.g. {"deepseek-r1": 0.6} to
                        slow down rate-limited models). Default empty = 1.0 for all.
        decay_rate: Per-update mean-reversion strength towards 0.5.
    """

    domains: tuple[str, ...] = ("general",)
    initial_lr: float = 0.15
    min_lr: float = 0.02
    cap: float = 0.92
    floor: float = 0.1
    entropy_threshold: float = 0.7
    cold_start_threshold: int = 15
    state_path: Path | None = None
    lr_multipliers: dict[str, float] = field(default_factory=dict)
    decay_rate: float = 0.005

    # Internal state
    _models: dict[str, dict[str, float]] = field(default_factory=dict)
    _priors: dict[str, dict[str, BetaPrior]] = field(default_factory=dict)
    _weight_history: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)),
    )
    _task_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)),
    )
    _bias_history: list[dict[str, Any]] = field(default_factory=list)
    _update_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def register_model(self, model_id: str, initial_weight: float = DEFAULT_WEIGHT) -> None:
        """Register a model with an optional initial weight hint.

        Safe to call multiple times -- only initializes missing domains.
        """
        with self._lock:
            self._ensure_model(model_id, initial_weight)

    def _ensure_model(self, model_id: str, static_weight: float = DEFAULT_WEIGHT) -> None:
        """Lazily initialize model state for all known domains."""
        if model_id not in self._models:
            self._models[model_id] = {}
            logger.debug("Registered model %s (initial=%.2f)", model_id, static_weight)
        if model_id not in self._priors:
            self._priors[model_id] = {}
        for domain in self.domains:
            if domain not in self._models[model_id]:
                self._models[model_id][domain] = static_weight
                self._priors[model_id][domain] = _init_prior(static_weight)
                self._weight_history[model_id][domain] = [static_weight]
                self._task_counts[model_id][domain] = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, model_id: str, signal: FeedbackSignal) -> float:
        """Update trust weight for a model given a feedback signal.

        Returns the new weight after update.
        """
        with self._lock:
            return self._update_impl(model_id, signal)

    def _update_impl(self, model_id: str, signal: FeedbackSignal) -> float:
        """Internal update (caller holds _lock)."""
        self._ensure_model(model_id)
        domain = signal.domain if signal.domain in self.domains else "general"

        w_old = self._models[model_id].get(domain, DEFAULT_WEIGHT)
        prior = self._priors[model_id].get(domain, BetaPrior())
        task_count = self._task_counts[model_id].get(domain, 0)
        history = self._weight_history[model_id].get(domain, [w_old])

        # Adaptive LR with per-model multiplier
        base_lr = _compute_adaptive_lr(task_count, history, self.initial_lr, self.min_lr)
        multiplier = self.lr_multipliers.get(model_id, 1.0)
        lr = base_lr * multiplier * signal.weight

        # Domain bias
        domain_bias = _compute_domain_bias(model_id, domain, self._bias_history)

        # Core logit update: CENTERED outcome so <0.5 decreases, >0.5 increases
        centered_outcome = (signal.outcome - 0.5) * 2.0  # in [-1, 1]
        logit_delta = lr * (centered_outcome + signal.disagreement) + domain_bias

        if prior.is_cold(self.cold_start_threshold):
            prior.update(signal.outcome)
            cold_ratio = max(0.0, min(
                1.0, 1.0 - prior.observations / self.cold_start_threshold
            ))
            logit_new = _logit(w_old) + logit_delta
            w_new = cold_ratio * prior.mean + (1.0 - cold_ratio) * _sigmoid(logit_new)
        else:
            prior.update(signal.outcome)
            w_new = _sigmoid(_logit(w_old) + logit_delta)

        # Temporal decay: mean-reversion towards 0.5
        w_new = w_new * (1.0 - self.decay_rate) + 0.5 * self.decay_rate

        # Per-model rate limiting via lr_multipliers affects max change rate
        if multiplier < 1.0 and w_old > 1e-9:
            max_change = 1.0 + (1.0 - multiplier) * 0.5  # e.g. mult=0.6 -> max 1.2x
            min_change = 1.0 - (1.0 - multiplier) * 0.5  # e.g. mult=0.6 -> min 0.8x
            ratio = w_new / w_old
            ratio = max(min_change, min(max_change, ratio))
            w_new = w_old * ratio

        # Hard cap/floor
        w_new = max(self.floor, min(self.cap, w_new))

        # Store state
        self._models[model_id][domain] = w_new
        self._priors[model_id][domain] = prior
        self._weight_history[model_id][domain].append(w_new)
        if len(self._weight_history[model_id][domain]) > 100:
            self._weight_history[model_id][domain].pop(0)
        self._task_counts[model_id][domain] = task_count + 1

        # Bias history for domain EMA
        self._bias_history.append({
            "model": model_id,
            "domain": domain,
            "outcome": signal.outcome,
            "timestamp": time.time(),
        })
        if len(self._bias_history) > 10000:
            self._bias_history = self._bias_history[-8000:]

        logger.debug(
            "update: %s [%s] %.4f -> %.4f (outcome=%.2f, lr=%.4f, cold=%s)",
            model_id, domain, w_old, w_new, signal.outcome, lr,
            prior.is_cold(self.cold_start_threshold),
        )
        self._maybe_persist()
        return w_new

    # ------------------------------------------------------------------
    # Convenience: thumbs up/down
    # ------------------------------------------------------------------

    def update_from_feedback(
        self,
        model_id: str,
        positive: bool,
        domain: str = "general",
    ) -> float:
        """Simple thumbs-up/thumbs-down update.

        Translates boolean feedback to outcome (1.0 or 0.0) and delegates
        to the core update method.
        """
        signal = FeedbackSignal(
            outcome=1.0 if positive else 0.0,
            domain=domain,
        )
        return self.update(model_id, signal)

    # ------------------------------------------------------------------
    # Batch update
    # ------------------------------------------------------------------

    def update_batch(
        self,
        model_ids: list[str],
        signal: FeedbackSignal,
        per_model_disagreement: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Atomically update multiple models with the same base signal.

        per_model_disagreement overrides signal.disagreement per model.
        """
        with self._lock:
            results: dict[str, float] = {}
            for mid in model_ids:
                if per_model_disagreement and mid in per_model_disagreement:
                    s = FeedbackSignal(
                        outcome=signal.outcome,
                        domain=signal.domain,
                        weight=signal.weight,
                        disagreement=per_model_disagreement[mid],
                    )
                else:
                    s = signal
                results[mid] = self._update_impl(mid, s)
            return results

    # ------------------------------------------------------------------
    # Weight retrieval
    # ------------------------------------------------------------------

    def get_weights(self, domain: str = "general") -> dict[str, float]:
        """Return entropy-enforced weights for all registered models."""
        with self._lock:
            raw = {
                mid: domains.get(domain, domains.get("general", DEFAULT_WEIGHT))
                for mid, domains in self._models.items()
            }
        return enforce_entropy(raw, self.floor, self.entropy_threshold)

    def get_weight(self, model_id: str, domain: str = "general") -> float:
        """Return raw weight for a single model (no entropy enforcement)."""
        with self._lock:
            if model_id not in self._models:
                return DEFAULT_WEIGHT
            return self._models[model_id].get(domain, DEFAULT_WEIGHT)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist full engine state to disk (atomic write)."""
        if self.state_path is None:
            return
        try:
            with self._lock:
                state = self._serialize()
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(state, indent=2, ensure_ascii=False)
            fd, tmp = tempfile.mkstemp(
                dir=self.state_path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                os.replace(tmp, str(self.state_path))
            except (OSError, TypeError, ValueError):
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            logger.debug("State saved to %s", self.state_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Save failed: %s", exc, exc_info=True)

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> TrustEngine:
        """Load engine from a persisted state file.

        Extra kwargs are forwarded to the constructor (e.g. domains, cap).
        """
        engine = cls(state_path=path, **kwargs)
        if path.exists():
            try:
                engine._restore(path)
                logger.info("Restored state from %s (%d models)", path, len(engine._models))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Restore failed, using defaults: %s", exc, exc_info=True)
        return engine

    def _serialize(self) -> dict[str, Any]:
        """Snapshot internal state for JSON serialization."""
        return {
            "models": {m: dict(d) for m, d in self._models.items()},
            "task_counts": {m: dict(d) for m, d in self._task_counts.items()},
            "priors": {
                m: {dom: p.to_dict() for dom, p in doms.items()}
                for m, doms in self._priors.items()
            },
            "bias_history": list(self._bias_history[-2000:]),
            "weight_history": {
                m: {d: list(h) for d, h in doms.items()}
                for m, doms in self._weight_history.items()
            },
        }

    def _restore(self, path: Path) -> None:
        """Restore state from a JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._models = raw.get("models", {})
        self._bias_history = raw.get("bias_history", [])
        for m, domains in raw.get("task_counts", {}).items():
            for domain, count in domains.items():
                self._task_counts[m][domain] = count
        for m, domains in raw.get("weight_history", {}).items():
            for domain, hist in domains.items():
                self._weight_history[m][domain] = hist
        for m, domains in raw.get("priors", {}).items():
            self._priors[m] = {}
            for domain, p in domains.items():
                self._priors[m][domain] = BetaPrior.from_dict(p)

    def _maybe_persist(self) -> None:
        """Auto-persist every 10 updates."""
        self._update_count += 1
        if self._update_count % 10 == 0:
            # Release lock before disk I/O -- save() acquires its own lock
            # We call _persist_weights directly to avoid double-locking
            self._persist_no_lock()

    def _persist_no_lock(self) -> None:
        """Lightweight persist (models + timestamp only), no lock needed."""
        if self.state_path is None:
            return
        try:
            state = {
                "models": {m: dict(d) for m, d in self._models.items()},
                "updated_at": time.time(),
            }
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Periodic persist failed: %s", exc, exc_info=True)
