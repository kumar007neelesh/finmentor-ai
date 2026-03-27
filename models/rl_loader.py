"""
models/rl_loader.py — FinMentor AI RL Model Loader
====================================================
Wraps a pretrained Reinforcement Learning model (Stable-Baselines3 PPO,
RLlib, or a mock) as a standardised callable tool:

    predict_return(state, context) -> PredictionResult

Responsibilities:
  - Load & cache a pretrained SB3 / RLlib model from disk
  - Accept a UserFinancialState, convert to obs vector, run inference
  - Compute a confidence score from the policy's action-probability distribution
  - Raise a low-confidence signal so the tool registry can invoke ML fallback
  - Expose async-safe inference (thread-pool executor wrapping blocking calls)
  - Provide a deterministic MockRLModel for unit tests (no GPU / model file needed)

Architecture position:
    UserFinancialState
         │  .to_observation_vector()  →  np.ndarray(20,)
         ▼
    RLModelLoader.predict()
         │
         ├─ confidence ≥ threshold  →  PredictionResult (from RL)
         └─ confidence <  threshold  →  raises LowConfidenceError
                                        → ML fallback takes over (rl_loader.py → ml_loader.py)

Usage:
    from models.rl_loader import build_rl_loader, PredictionResult

    loader = build_rl_loader()          # reads config automatically
    result = loader.predict(state)

    print(result.predicted_return)      # e.g. 0.143  (14.3% annual return)
    print(result.recommended_action)    # e.g. "increase_equity"
    print(result.confidence)            # e.g. 0.81
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from config import settings, ModelBackend
from environment.state import OBS_DIM

from environment.state import UserFinancialState, OBSERVATION_FEATURES
from logger import get_logger, metrics, timed

log = get_logger(__name__)

# Thread pool for running blocking model inference without blocking the event loop
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rl_inference")


# ══════════════════════════════════════════════════════════════════════════════
# Action space — what the RL agent can recommend
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioAction(str, Enum):
    """
    Discrete action space for the RL model.
    Maps to index 0–4 in the model's output.
    Must match the action space the model was trained with.
    """
    INCREASE_EQUITY    = "increase_equity"      # 0 — shift more to equity MFs
    REDUCE_EQUITY      = "reduce_equity"        # 1 — de-risk, move to debt
    INCREASE_SIP       = "increase_sip"         # 2 — raise monthly SIP amount
    BUILD_EMERGENCY    = "build_emergency_fund" # 3 — prioritise liquid savings
    OPTIMIZE_TAX       = "optimize_tax"         # 4 — use 80C/NPS/HRA headroom


ACTION_INDEX: Dict[int, PortfolioAction] = {
    0: PortfolioAction.INCREASE_EQUITY,
    1: PortfolioAction.REDUCE_EQUITY,
    2: PortfolioAction.INCREASE_SIP,
    3: PortfolioAction.BUILD_EMERGENCY,
    4: PortfolioAction.OPTIMIZE_TAX,
}

# Expected annual return delta per action (used to compute predicted_return)
ACTION_RETURN_DELTA: Dict[PortfolioAction, float] = {
    PortfolioAction.INCREASE_EQUITY:    0.02,   # +2% CAGR from higher equity
    PortfolioAction.REDUCE_EQUITY:     -0.01,   # -1% CAGR (safer, lower return)
    PortfolioAction.INCREASE_SIP:       0.015,  # compounding effect
    PortfolioAction.BUILD_EMERGENCY:   -0.005,  # opportunity cost
    PortfolioAction.OPTIMIZE_TAX:       0.01,   # effective post-tax return gain
}


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """
    Standardised output from any RL / ML model call.
    The Tool Registry and Planner consume this object — never raw model output.
    """

    # Core prediction
    recommended_action: PortfolioAction
    predicted_return: float          # Expected annual portfolio return (fraction, e.g. 0.143)
    confidence: float                # 0.0–1.0  (derived from policy softmax)

    # Ranked alternatives (action, probability)
    action_probabilities: List[Tuple[str, float]] = field(default_factory=list)

    # Metadata
    model_backend: str = "unknown"
    inference_time_ms: float = 0.0
    obs_vector: Optional[List[float]] = None   # Store for audit / debugging
    is_fallback: bool = False                  # True if ML fallback was used

    # Human-readable rationale (filled in by the caller, not the model)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_action":   self.recommended_action.value,
            "predicted_return_pct": round(self.predicted_return * 100, 2),
            "confidence":           round(self.confidence, 4),
            "action_probabilities": self.action_probabilities,
            "model_backend":        self.model_backend,
            "inference_time_ms":    round(self.inference_time_ms, 2),
            "is_fallback":          self.is_fallback,
            "rationale":            self.rationale,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Custom exceptions
# ══════════════════════════════════════════════════════════════════════════════

class ModelLoadError(RuntimeError):
    """Raised when the model file cannot be loaded from disk."""


class LowConfidenceError(RuntimeError):
    """
    Raised when model confidence falls below the configured threshold.
    The tool registry catches this and routes to the ML fallback.
    """
    def __init__(self, confidence: float, threshold: float) -> None:
        self.confidence = confidence
        self.threshold  = threshold
        super().__init__(
            f"RL model confidence {confidence:.3f} < threshold {threshold:.3f}. "
            "Routing to ML fallback."
        )


class InferenceError(RuntimeError):
    """Raised when model inference fails unexpectedly."""


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base — every backend implements this
# ══════════════════════════════════════════════════════════════════════════════

class BaseRLModel(ABC):
    """
    Interface that all RL backends must implement.
    Keeps the RLModelLoader backend-agnostic.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights from disk into memory."""
        ...

    @abstractmethod
    def _raw_predict(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Run a single forward pass.
        Returns:
            action_index (int): index into ACTION_INDEX
            action_probs (np.ndarray): softmax probabilities, shape (n_actions,)
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if model weights are in memory."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Backend 1: Stable-Baselines3 (PPO / A2C / DQN)
# ══════════════════════════════════════════════════════════════════════════════

class SB3RLModel(BaseRLModel):
    """
    Wraps a Stable-Baselines3 PPO (or A2C / DQN) model.

    Training assumption:
        The model was trained on a custom Gym environment where:
          - obs_space  = Box(low=0, high=1, shape=(20,), dtype=float32)
          - action_space = Discrete(5)   # matches ACTION_INDEX above

    To train and save:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500_000)
        model.save("saved_models/finmentor_ppo")
    """

    def __init__(self, model_path: Path, algorithm: str = "PPO") -> None:
        self._model_path = model_path
        self._algorithm  = algorithm.upper()
        self._model      = None
        self._loaded     = False

    def load(self) -> None:
        try:
            # Lazy import — SB3 is heavy; only load if backend is sb3
            if self._algorithm == "PPO":
                from stable_baselines3 import PPO as Algo
            elif self._algorithm == "A2C":
                from stable_baselines3 import A2C as Algo
            elif self._algorithm == "DQN":
                from stable_baselines3 import DQN as Algo
            else:
                raise ModelLoadError(f"Unsupported SB3 algorithm: {self._algorithm}")

            self._model = Algo.load(
                str(self._model_path),
                device=settings.rl_model.device,
            )
            self._loaded = True
            log.info(
                "SB3 model loaded",
                algorithm=self._algorithm,
                path=str(self._model_path),
                device=settings.rl_model.device,
            )
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"SB3 model file not found: {self._model_path}"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load SB3 model: {exc}") from exc

    def _raw_predict(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Run SB3 policy forward pass.
        For PPO/A2C: extract action probabilities from the policy distribution.
        For DQN:     use Q-values converted to pseudo-probabilities via softmax.
        """
        if not self._loaded or self._model is None:
            raise InferenceError("Model not loaded. Call .load() first.")

        # SB3 predict returns (action, state)
        action, _ = self._model.predict(obs, deterministic=True)
        action_index = int(action)

        # Extract action probabilities from the policy
        try:
            import torch
            obs_tensor = self._model.policy.obs_to_tensor(obs)[0]

            if self._algorithm in ("PPO", "A2C"):
                with torch.no_grad():
                    distribution = self._model.policy.get_distribution(obs_tensor)
                    log_probs = distribution.distribution.logits
                    probs = torch.softmax(log_probs, dim=-1).cpu().numpy().flatten()
            elif self._algorithm == "DQN":
                with torch.no_grad():
                    q_values = self._model.policy.q_net(obs_tensor)
                    probs = torch.softmax(q_values, dim=-1).cpu().numpy().flatten()
            else:
                # Uniform fallback
                n = settings.rl_model.action_space_size
                probs = np.ones(n, dtype=np.float32) / n

        except Exception as exc:
            log.warning(
                "Could not extract action probabilities from policy",
                error=str(exc),
                fallback="uniform distribution",
            )
            n = settings.rl_model.action_space_size
            probs = np.ones(n, dtype=np.float32) / n
            probs[action_index] = 0.5  # Give chosen action higher weight
            probs /= probs.sum()

        return action_index, probs

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return f"sb3_{self._algorithm.lower()}"


# ══════════════════════════════════════════════════════════════════════════════
# Backend 2: Mock (deterministic, no model file needed)
# ══════════════════════════════════════════════════════════════════════════════

class MockRLModel(BaseRLModel):
    """
    Deterministic rule-based mock that mimics RL model outputs.
    Used in:
      - Unit tests (no file I/O or GPU needed)
      - Development environments without a trained model
      - CI/CD pipelines

    Logic mirrors what a well-trained RL agent would learn:
      - Low emergency fund  → BUILD_EMERGENCY
      - High FOIR           → REDUCE_EQUITY
      - Low savings rate    → INCREASE_SIP
      - Tax headroom left   → OPTIMIZE_TAX
      - Default             → INCREASE_EQUITY
    """

    _loaded = True

    def load(self) -> None:
        log.info("MockRLModel loaded (no file I/O — deterministic rules)")

    def _raw_predict(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        obs indices (from OBSERVATION_FEATURES):
          2  = savings_rate
          3  = foir
          4  = emergency_fund_ratio
          15 = tax_bracket_normalized
        """
        savings_rate_norm      = float(obs[2])
        foir_norm              = float(obs[3])
        emergency_fund_ratio   = float(obs[4])
        tax_bracket            = float(obs[15])

        # Rule priority chain
        if emergency_fund_ratio < 0.5:          # < 3 months covered
            chosen = 3   # BUILD_EMERGENCY
            base_probs = [0.05, 0.05, 0.10, 0.75, 0.05]
        elif foir_norm > 0.6:                   # FOIR > 60%  (very high debt)
            chosen = 1   # REDUCE_EQUITY
            base_probs = [0.05, 0.75, 0.10, 0.05, 0.05]
        elif savings_rate_norm < 0.15:          # < 15% savings rate
            chosen = 2   # INCREASE_SIP
            base_probs = [0.10, 0.05, 0.70, 0.10, 0.05]
        elif tax_bracket > 0.5:                 # 30% tax bracket
            chosen = 4   # OPTIMIZE_TAX
            base_probs = [0.10, 0.05, 0.10, 0.05, 0.70]
        else:
            chosen = 0   # INCREASE_EQUITY  (default wealth-building action)
            base_probs = [0.65, 0.10, 0.15, 0.05, 0.05]

        probs = np.array(base_probs, dtype=np.float32)
        probs /= probs.sum()    # Ensure sums to 1.0
        return chosen, probs

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "mock"


# ══════════════════════════════════════════════════════════════════════════════
# RLModelLoader — the public interface
# ══════════════════════════════════════════════════════════════════════════════

class RLModelLoader:
    """
    Public facade for RL inference.

    Handles:
      - Backend selection (SB3 | Mock)
      - Confidence scoring
      - LowConfidenceError signalling
      - Async-safe blocking inference via thread pool
      - Metrics emission for every prediction

    This is registered as a tool in tools/registry.py:
        "rl_predict": loader.predict
    """

    def __init__(self, model: BaseRLModel) -> None:
        self._model = model
        self._conf_threshold = settings.rl_model.confidence_threshold
        self._action_space_size = settings.rl_model.action_space_size
        self._base_return = settings.finance.default_equity_return_pct / 100.0

    # ── Sync predict ──────────────────────────────────────────────────────────

    @timed("rl_model.predict", tags={"layer": "rl"})
    def predict(
        self,
        state: UserFinancialState,
        raise_on_low_confidence: bool = True,
    ) -> PredictionResult:
        """
        Run RL inference on the given state.

        Args:
            state: The full user financial state.
            raise_on_low_confidence: If True, raises LowConfidenceError when
                confidence < threshold (allowing fallback to ML model).
                If False, returns the result regardless of confidence.

        Returns:
            PredictionResult with action, return prediction, and confidence.

        Raises:
            LowConfidenceError: When confidence < config threshold (if raise_on_low_confidence).
            InferenceError: On model forward-pass failure.
        """
        if not self._model.is_loaded:
            self._model.load()

        t0 = time.perf_counter()

        # Step 1: Get observation vector
        obs = state.to_observation_vector()

        # Step 2: Validate shape matches what model expects
        expected_dim = settings.rl_model.observation_dim
        if obs.shape[0] != expected_dim:
            raise InferenceError(
                f"Observation dimension mismatch: got {obs.shape[0]}, "
                f"expected {expected_dim}."
            )

        # Step 3: Run forward pass
        try:
            action_index, action_probs = self._model._raw_predict(obs)
        except Exception as exc:
            metrics.increment("rl_model.inference_error")
            raise InferenceError(f"Model forward pass failed: {exc}") from exc

        # Step 4: Map action index → PortfolioAction (clamp to valid range)
        action_index = int(np.clip(action_index, 0, len(ACTION_INDEX) - 1))
        recommended_action = ACTION_INDEX[action_index]

        # Step 5: Compute confidence = probability of the chosen action
        confidence = float(action_probs[action_index])

        # Step 6: Compute predicted annual return
        #   base_return (e.g. 12%) + delta from recommended action
        return_delta = ACTION_RETURN_DELTA.get(recommended_action, 0.0)
        predicted_return = self._base_return + return_delta

        # Adjust for risk profile
        risk_adj = (state.risk_score - 5.0) * 0.005  # ±0.5% per risk point
        predicted_return = max(0.0, predicted_return + risk_adj)

        # Step 7: Build sorted probability list for transparency
        sorted_probs = sorted(
            [
                (ACTION_INDEX[i].value, round(float(action_probs[i]), 4))
                for i in range(len(action_probs))
            ],
            key=lambda x: -x[1],
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = PredictionResult(
            recommended_action=recommended_action,
            predicted_return=round(predicted_return, 6),
            confidence=round(confidence, 4),
            action_probabilities=sorted_probs,
            model_backend=self._model.backend_name,
            inference_time_ms=round(elapsed_ms, 2),
            obs_vector=obs.tolist(),
            rationale=self._build_rationale(recommended_action, state, confidence),
        )

        # Step 8: Emit metrics
        metrics.increment(
            "rl_model.predictions",
            tags={"action": recommended_action.value, "backend": self._model.backend_name},
        )
        metrics.observe("rl_model.confidence", confidence)
        metrics.observe("rl_model.predicted_return", predicted_return)

        log.info(
            "RL prediction complete",
            action=recommended_action.value,
            confidence=round(confidence, 3),
            predicted_return_pct=round(predicted_return * 100, 2),
            backend=self._model.backend_name,
            latency_ms=round(elapsed_ms, 2),
            user_id=state.user_id,
        )

        # Step 9: Low-confidence guard
        if raise_on_low_confidence and confidence < self._conf_threshold:
            log.warning(
                "Low confidence — signalling ML fallback",
                confidence=round(confidence, 3),
                threshold=self._conf_threshold,
                user_id=state.user_id,
            )
            metrics.increment("rl_model.low_confidence")
            raise LowConfidenceError(confidence, self._conf_threshold)

        return result

    # ── Async predict ─────────────────────────────────────────────────────────

    async def predict_async(
        self,
        state: UserFinancialState,
        raise_on_low_confidence: bool = True,
    ) -> PredictionResult:
        """
        Async-safe wrapper around predict().
        Runs the blocking inference in a thread pool so it doesn't block
        the FastAPI / asyncio event loop.

        Usage in async agent executor:
            result = await loader.predict_async(state)
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _THREAD_POOL,
            lambda: self.predict(state, raise_on_low_confidence),
        )
        return result

    # ── Batch predict (for offline evaluation / backtesting) ─────────────────

    def predict_batch(
        self,
        states: List[UserFinancialState],
        raise_on_low_confidence: bool = False,
    ) -> List[PredictionResult]:
        """
        Run inference on a list of states.
        Used by the Evaluator for offline batch scoring.
        """
        results = []
        for i, state in enumerate(states):
            try:
                result = self.predict(state, raise_on_low_confidence)
                results.append(result)
            except LowConfidenceError as exc:
                log.warning(
                    f"Low confidence on batch item {i}, skipping",
                    confidence=exc.confidence,
                )
                # Append a low-confidence placeholder
                results.append(
                    PredictionResult(
                        recommended_action=PortfolioAction.BUILD_EMERGENCY,
                        predicted_return=self._base_return,
                        confidence=exc.confidence,
                        model_backend=self._model.backend_name,
                        is_fallback=True,
                        rationale="Low confidence — defaulting to conservative action.",
                    )
                )
            except InferenceError as exc:
                log.error(f"Inference error on batch item {i}: {exc}")
                raise
        return results

    # ── Warm-up ───────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Run a dummy prediction to pre-load model weights into memory
        and JIT-compile any Torch graphs.
        Call during application startup to avoid cold-start latency.
        """
        log.info("Warming up RL model...")
        from environment.state import minimal_state
        from config import RiskProfile
        dummy = minimal_state(30, 80_000, 50_000, RiskProfile.MODERATE)
        try:
            self.predict(dummy, raise_on_low_confidence=False)
            log.info("RL model warm-up complete")
        except Exception as exc:
            log.warning(f"RL model warm-up failed (non-fatal): {exc}")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_rationale(
        action: PortfolioAction,
        state: UserFinancialState,
        confidence: float,
    ) -> str:
        """
        Generate a one-sentence rationale for the recommended action.
        Consumed by the LLM planner to explain tool output to the user.
        """
        templates = {
            PortfolioAction.INCREASE_EQUITY: (
                f"With a {state.investment_horizon_years}-year horizon and "
                f"{state.risk_profile.value} risk profile, increasing equity "
                f"allocation can improve long-term CAGR."
            ),
            PortfolioAction.REDUCE_EQUITY: (
                f"FOIR of {state.foir:.1%} indicates debt load — reducing equity "
                f"exposure lowers portfolio volatility risk."
            ),
            PortfolioAction.INCREASE_SIP: (
                f"Savings rate of {state.savings_rate:.1%} leaves room to increase "
                f"SIP for accelerated goal achievement."
            ),
            PortfolioAction.BUILD_EMERGENCY: (
                f"Emergency fund covers only {state.emergency_fund_months:.1f} months "
                f"(target: 6). Prioritise liquidity before investing."
            ),
            PortfolioAction.OPTIMIZE_TAX: (
                f"Effective tax rate of {state.effective_tax_rate:.1%} — "
                f"80C/NPS/HRA optimisation can recover post-tax returns."
            ),
        }
        base = templates.get(action, "Model recommendation based on portfolio analysis.")
        conf_note = f" (Model confidence: {confidence:.0%})"
        return base + conf_note


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_rl_loader() -> RLModelLoader:
    """
    Factory — reads config and returns a ready-to-use RLModelLoader.

    Selection logic:
      config.rl_model.backend == "mock" → MockRLModel (no file needed)
      config.rl_model.backend == "sb3"  → SB3RLModel  (loads from model_path)

    Called once at application startup; result is shared across all requests.
    """
    backend = settings.rl_model.backend

    if backend == ModelBackend.MOCK:
        model = MockRLModel()
        model.load()
        log.info("RL loader initialised with MockRLModel")

    elif backend == ModelBackend.STABLE_BASELINES3:
        model_path = settings.rl_model.model_path
        if model_path is None:
            raise ModelLoadError(
                "config.rl_model.model_path must be set when backend='sb3'. "
                "Set FINMENTOR__RL_MODEL__MODEL_PATH in your .env file."
            )
        model = SB3RLModel(
            model_path=Path(model_path),
            algorithm=settings.rl_model.algorithm,
        )
        model.load()
        log.info(
            "RL loader initialised with SB3RLModel",
            algorithm=settings.rl_model.algorithm,
            path=str(model_path),
        )

    else:
        raise ModelLoadError(
            f"Unsupported RL backend: {backend}. "
            "Supported: 'mock', 'sb3'."
        )

    return RLModelLoader(model)


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json
    from environment.state import (
        UserFinancialState, InsuranceCoverage, DebtProfile,
        InvestmentPortfolio, EmploymentType, CityTier,
    )
    from config import RiskProfile

    print("=== RLModelLoader Self-Test (MockRLModel) ===\n")

    loader = build_rl_loader()   # Uses MOCK by default in dev

    # ── Test Case 1: Critical emergency fund ─────────────────────────────────
    state_1 = UserFinancialState(
        age=28,
        monthly_income=60_000,
        monthly_expense=52_000,
        emergency_fund_amount=50_000,   # Only ~1 month
        risk_profile=RiskProfile.MODERATE,
        risk_score=5.0,
        investment_horizon_years=30,
    )
    result_1 = loader.predict(state_1, raise_on_low_confidence=False)
    print("── Test 1: Low Emergency Fund ──")
    print(f"  Action     : {result_1.recommended_action.value}")
    print(f"  Confidence : {result_1.confidence:.2%}")
    print(f"  Return est : {result_1.predicted_return*100:.2f}% p.a.")
    print(f"  Rationale  : {result_1.rationale}")
    print(f"  Probs      : {result_1.action_probabilities}\n")

    # ── Test Case 2: Aggressive investor, high income ─────────────────────────
    state_2 = UserFinancialState(
        age=35,
        monthly_income=200_000,
        monthly_expense=80_000,
        emergency_fund_amount=700_000,  # 8.75 months ✓
        risk_profile=RiskProfile.AGGRESSIVE,
        risk_score=8.5,
        investment_horizon_years=25,
        debts=DebtProfile(home_loan_emi=30_000),
        portfolio=InvestmentPortfolio(equity_mf=2_000_000, debt_mf=500_000),
    )
    result_2 = loader.predict(state_2, raise_on_low_confidence=False)
    print("── Test 2: Aggressive Investor ──")
    print(f"  Action     : {result_2.recommended_action.value}")
    print(f"  Confidence : {result_2.confidence:.2%}")
    print(f"  Return est : {result_2.predicted_return*100:.2f}% p.a.")
    print(f"  Rationale  : {result_2.rationale}\n")

    # ── Test Case 3: High FOIR scenario ──────────────────────────────────────
    state_3 = UserFinancialState(
        age=40,
        monthly_income=100_000,
        monthly_expense=40_000,
        emergency_fund_amount=400_000,
        risk_profile=RiskProfile.CONSERVATIVE,
        risk_score=3.0,
        investment_horizon_years=20,
        debts=DebtProfile(
            home_loan_emi=35_000,
            car_loan_emi=12_000,
            personal_loan_emi=8_000,
        ),
    )
    result_3 = loader.predict(state_3, raise_on_low_confidence=False)
    print("── Test 3: High FOIR (55%) ──")
    print(f"  FOIR       : {state_3.foir:.1%}")
    print(f"  Action     : {result_3.recommended_action.value}")
    print(f"  Confidence : {result_3.confidence:.2%}")
    print(f"  Return est : {result_3.predicted_return*100:.2f}% p.a.\n")

    # ── Test Case 4: LowConfidenceError ──────────────────────────────────────
    print("── Test 4: LowConfidenceError (forced via low threshold override) ──")
    loader._conf_threshold = 0.99   # Force low confidence on any prediction
    try:
        loader.predict(state_2, raise_on_low_confidence=True)
    except LowConfidenceError as exc:
        print(f"  ✓ LowConfidenceError raised as expected: {exc}\n")
    finally:
        loader._conf_threshold = settings.rl_model.confidence_threshold  # Reset

    # ── Test Case 5: Batch predict ────────────────────────────────────────────
    print("── Test 5: Batch Predict ──")
    results = loader.predict_batch([state_1, state_2, state_3])
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r.recommended_action.value:<25}  conf={r.confidence:.2%}  return={r.predicted_return*100:.1f}%")

    # ── Metrics snapshot ──────────────────────────────────────────────────────
    print("\n── Metrics Snapshot ──")
    snap = metrics.snapshot()
    for c in snap["counters"]:
        print(f"  {c['name']:<40} {c['value']}")
    for h in snap["histograms"]:
        print(f"  {h['name']:<40} mean={h['mean_ms']:.1f}ms  p99={h['p99_ms']:.1f}ms")
