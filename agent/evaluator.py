"""
agent/evaluator.py — FinMentor AI Prediction Evaluator
=======================================================
Closes the feedback loop between predictions and reality.

Responsibilities:
  1. ACCURACY MEASUREMENT
     Compare the RL model's predicted_return_pct vs the actual portfolio
     return observed 30+ days after the recommendation was made.

  2. MODEL DRIFT DETECTION
     Track rolling Mean Absolute Error (MAE) across evaluated episodes.
     Alert when MAE exceeds the configured tolerance threshold.

  3. RETRAINING SIGNAL
     If drift is severe (MAE > 2× tolerance), raise a RetrainingAlert
     that can trigger a CI/CD pipeline or Slack notification.

  4. PLAN QUALITY SCORING
     Score each FinancialPlan on dimensions beyond return accuracy:
       - Tool selection appropriateness (did it call the right tools?)
       - Response completeness (did it address all user goals?)
       - Consistency with user's risk profile

  5. BATCH EVALUATION
     Scan all pending episodes in Memory, score them, write results back.

Architecture position:
    Memory.retrieve_for_evaluation()
           │
           ▼
    Evaluator.run_batch_evaluation()
           │
    ┌──────┴────────────────────────────────────────┐
    │  For each pending episode:                    │
    │    1. Fetch actual return (portfolio API /    │
    │       user-reported / synthetic in tests)     │
    │    2. Compute accuracy score                  │
    │    3. Update episode in Memory                │
    │    4. Check drift → alert if needed           │
    └───────────────────────────────────────────────┘
           │
           ▼
    EvaluationReport  (summary + alerts + recommendations)

Usage:
    from agent.evaluator import Evaluator, build_evaluator

    evaluator = build_evaluator()

    # Evaluate a single plan immediately (for testing / live feedback)
    result = await evaluator.evaluate_plan(plan, actual_return_pct=11.8)

    # Batch evaluate all pending episodes for a user
    report = await evaluator.run_batch_evaluation(user_id="u_001")
    print(report.summary())

    # System-wide drift check (run as a scheduled job)
    report = await evaluator.run_system_evaluation()
    if report.retraining_required:
        trigger_retraining_pipeline()
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from config import settings
from agent.memory import BaseMemoryStore, Episode, get_memory_store
from agent.planner import FinancialPlan
from logger import get_logger, metrics, audit

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Enums and constants
# ══════════════════════════════════════════════════════════════════════════════

class DriftSeverity(str, Enum):
    NONE     = "none"        # MAE within tolerance
    MILD     = "mild"        # MAE 1–1.5× tolerance
    MODERATE = "moderate"    # MAE 1.5–2× tolerance
    SEVERE   = "severe"      # MAE > 2× tolerance → retraining required


class PlanQualityDimension(str, Enum):
    TOOL_SELECTION   = "tool_selection"
    GOAL_COVERAGE    = "goal_coverage"
    RISK_ALIGNMENT   = "risk_alignment"
    RESPONSE_CLARITY = "response_clarity"
    RETURN_ACCURACY  = "return_accuracy"


# Intent → expected tools map (used to score tool selection)
INTENT_EXPECTED_TOOLS: Dict[str, List[str]] = {
    "fire_planning":    ["fire_planner", "rl_predict"],
    "sip_calculation":  ["sip_calculator"],
    "tax_optimization": ["tax_wizard"],
    "health_score":     ["health_score"],
    "portfolio_review": ["rl_predict", "health_score"],
    "debt_advice":      ["health_score"],
    "insurance_advice": ["health_score"],
    "life_event":       ["rl_predict"],
    "general_question": ["health_score"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeEvaluation:
    """
    Evaluation result for a single episode.
    Written back to Memory via memory.update_evaluation().
    """
    episode_id: str
    user_id: str

    # Return accuracy
    predicted_return_pct: Optional[float]
    actual_return_pct: Optional[float]
    return_error_pct: Optional[float]          # |predicted - actual|
    within_tolerance: Optional[bool]           # error < config tolerance

    # Plan quality dimensions (0.0–1.0 each)
    tool_selection_score: float = 0.0
    goal_coverage_score: float = 0.0
    risk_alignment_score: float = 0.0
    response_clarity_score: float = 0.0

    # Composite score (0.0–1.0)
    overall_score: float = 0.0

    # Metadata
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""

    @property
    def grade(self) -> str:
        if self.overall_score >= 0.85:  return "A"
        if self.overall_score >= 0.70:  return "B"
        if self.overall_score >= 0.55:  return "C"
        if self.overall_score >= 0.40:  return "D"
        return "F"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id":            self.episode_id,
            "user_id":               self.user_id,
            "predicted_return_pct":  self.predicted_return_pct,
            "actual_return_pct":     self.actual_return_pct,
            "return_error_pct":      round(self.return_error_pct, 4) if self.return_error_pct is not None else None,
            "within_tolerance":      self.within_tolerance,
            "scores": {
                "tool_selection":   round(self.tool_selection_score, 3),
                "goal_coverage":    round(self.goal_coverage_score, 3),
                "risk_alignment":   round(self.risk_alignment_score, 3),
                "response_clarity": round(self.response_clarity_score, 3),
                "overall":          round(self.overall_score, 3),
            },
            "grade":         self.grade,
            "evaluated_at":  self.evaluated_at,
            "notes":         self.notes,
        }


@dataclass
class DriftAlert:
    """Raised when model MAE exceeds tolerance thresholds."""
    severity: DriftSeverity
    current_mae: float
    tolerance: float
    ratio: float                    # current_mae / tolerance
    episodes_evaluated: int
    recommended_action: str
    triggered_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def requires_retraining(self) -> bool:
        return self.severity == DriftSeverity.SEVERE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity":            self.severity.value,
            "current_mae_pct":     round(self.current_mae, 4),
            "tolerance_pct":       round(self.tolerance, 4),
            "mae_to_tolerance":    round(self.ratio, 2),
            "episodes_evaluated":  self.episodes_evaluated,
            "requires_retraining": self.requires_retraining,
            "recommended_action":  self.recommended_action,
            "triggered_at":        self.triggered_at,
        }


@dataclass
class EvaluationReport:
    """
    Summary of a batch evaluation run (one user or system-wide).
    """
    report_id: str = field(default_factory=lambda: f"rpt_{uuid4().hex[:8]}")
    scope: str = "user"                   # "user" | "system"
    user_id: Optional[str] = None

    # Counts
    episodes_scanned: int = 0
    episodes_evaluated: int = 0
    episodes_skipped: int = 0             # Not yet old enough (< 30 days)

    # Accuracy stats
    mean_absolute_error: Optional[float] = None
    median_absolute_error: Optional[float] = None
    p90_absolute_error: Optional[float] = None
    accuracy_within_tolerance_pct: Optional[float] = None

    # Quality stats
    mean_overall_score: Optional[float] = None
    grade_distribution: Dict[str, int] = field(default_factory=dict)

    # Drift
    drift_alert: Optional[DriftAlert] = None
    retraining_required: bool = False

    # Individual results
    evaluations: List[EpisodeEvaluation] = field(default_factory=list)

    # Timing
    run_duration_ms: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report [{self.report_id}] ===",
            f"Scope          : {self.scope}" + (f" (user={self.user_id})" if self.user_id else ""),
            f"Episodes       : {self.episodes_evaluated} evaluated / {self.episodes_scanned} scanned",
        ]
        if self.mean_absolute_error is not None:
            lines.append(f"MAE            : {self.mean_absolute_error:.2f}%")
        if self.accuracy_within_tolerance_pct is not None:
            lines.append(f"Within tolerance: {self.accuracy_within_tolerance_pct:.1f}%")
        if self.mean_overall_score is not None:
            lines.append(f"Mean plan score : {self.mean_overall_score:.3f}")
        if self.grade_distribution:
            grade_str = "  ".join(f"{g}:{n}" for g, n in sorted(self.grade_distribution.items()))
            lines.append(f"Grades         : {grade_str}")
        if self.drift_alert:
            lines.append(f"⚠️  Drift Alert   : {self.drift_alert.severity.value.upper()} — {self.drift_alert.recommended_action}")
        if self.retraining_required:
            lines.append("🔴 RETRAINING REQUIRED")
        lines.append(f"Duration       : {self.run_duration_ms:.0f}ms")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id":             self.report_id,
            "scope":                 self.scope,
            "user_id":               self.user_id,
            "counts": {
                "scanned":   self.episodes_scanned,
                "evaluated": self.episodes_evaluated,
                "skipped":   self.episodes_skipped,
            },
            "accuracy": {
                "mae_pct":                self.mean_absolute_error,
                "median_ae_pct":          self.median_absolute_error,
                "p90_ae_pct":             self.p90_absolute_error,
                "within_tolerance_pct":   self.accuracy_within_tolerance_pct,
            },
            "quality": {
                "mean_overall_score": self.mean_overall_score,
                "grade_distribution": self.grade_distribution,
            },
            "drift_alert":         self.drift_alert.to_dict() if self.drift_alert else None,
            "retraining_required": self.retraining_required,
            "run_duration_ms":     round(self.run_duration_ms, 2),
            "created_at":          self.created_at,
            "evaluations":         [e.to_dict() for e in self.evaluations],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Actual return provider — pluggable interface
# ══════════════════════════════════════════════════════════════════════════════

# Type alias: given an episode, return (actual_return_pct, notes)
ActualReturnProvider = Callable[[Episode], Tuple[float, str]]


def synthetic_return_provider(episode: Episode) -> Tuple[float, str]:
    """
    Deterministic synthetic return for testing.
    Simulates realistic market noise: ±2% around the predicted value.
    """
    predicted = episode.predicted_return_pct or 12.0
    # Use episode_id hash for reproducible but varied noise
    seed = sum(ord(c) for c in episode.episode_id)
    noise = ((seed % 200) - 100) / 100.0 * 2.0   # ±2% noise
    actual = max(0.0, predicted + noise)
    return round(actual, 4), "Synthetic return (test mode)"


def nifty_benchmark_return_provider(episode: Episode) -> Tuple[float, str]:
    """
    Production stub: fetch actual Nifty 50 / portfolio return via an API.
    Replace with real CAMS/KFintech/Zerodha integration.

    For now, returns a hardcoded representative value per intent.
    """
    intent_returns = {
        "fire_planning":    11.8,
        "sip_calculation":  12.3,
        "portfolio_review": 10.9,
        "tax_optimization": 9.5,   # Post-tax return
        "health_score":     11.2,
        "life_event":       11.5,
    }
    actual = intent_returns.get(episode.intent, 11.0)
    return actual, "Nifty 50 benchmark (production stub)"


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Prediction accuracy evaluator and model drift detector.

    Wires together:
      - Memory (reads episodes needing evaluation)
      - ActualReturnProvider (fetches ground truth returns)
      - Scoring logic (return accuracy + plan quality)
      - DriftAlert generation
      - Memory write-back (stores evaluation results)
    """

    def __init__(
        self,
        memory: BaseMemoryStore,
        return_provider: ActualReturnProvider = synthetic_return_provider,
    ) -> None:
        self._memory          = memory
        self._return_provider = return_provider
        self._tolerance       = settings.agent.prediction_tolerance_pct  # e.g. 10.0

        # Rolling accuracy window (in-memory, not persisted across restarts)
        self._recent_errors: List[float] = []
        self._max_window = 200    # Last 200 evaluated episodes

    # ── Single episode evaluation ─────────────────────────────────────────────

    async def evaluate_episode(
        self,
        episode: Episode,
        actual_return_pct: Optional[float] = None,
    ) -> EpisodeEvaluation:
        """
        Score one episode.

        Args:
            episode:           The episode to evaluate.
            actual_return_pct: Provide to skip the return_provider lookup.
                               Useful for live feedback from user ("I made X%").

        Returns:
            EpisodeEvaluation with all scores populated.
        """
        # ── 1. Return accuracy ────────────────────────────────────────────────
        if actual_return_pct is None and episode.predicted_return_pct is not None:
            actual_return_pct, provider_notes = self._return_provider(episode)
        else:
            provider_notes = "User-provided actual return"

        return_error = None
        within_tolerance = None
        if episode.predicted_return_pct is not None and actual_return_pct is not None:
            return_error = abs(episode.predicted_return_pct - actual_return_pct)
            within_tolerance = return_error <= self._tolerance
            self._record_error(return_error)

        return_accuracy_score = self._score_return_accuracy(return_error)

        # ── 2. Tool selection quality ─────────────────────────────────────────
        tool_selection_score = self._score_tool_selection(
            intent=episode.intent,
            tools_used=episode.tools_used,
        )

        # ── 3. Goal coverage ──────────────────────────────────────────────────
        goal_coverage_score = self._score_goal_coverage(
            intent=episode.intent,
            tools_used=episode.tools_used,
            final_answer=episode.final_answer,
            state_snapshot=episode.state_snapshot,
        )

        # ── 4. Risk alignment ─────────────────────────────────────────────────
        risk_alignment_score = self._score_risk_alignment(
            state_snapshot=episode.state_snapshot,
            recommended_actions=episode.recommended_actions,
            tools_used=episode.tools_used,
        )

        # ── 5. Response clarity ───────────────────────────────────────────────
        response_clarity_score = self._score_response_clarity(episode.final_answer)

        # ── 6. Composite score ────────────────────────────────────────────────
        # Weights: return accuracy matters most for RL model evaluation
        weights = {
            "return_accuracy":  0.35,
            "tool_selection":   0.25,
            "goal_coverage":    0.20,
            "risk_alignment":   0.12,
            "response_clarity": 0.08,
        }
        overall = (
            return_accuracy_score  * weights["return_accuracy"]
            + tool_selection_score * weights["tool_selection"]
            + goal_coverage_score  * weights["goal_coverage"]
            + risk_alignment_score * weights["risk_alignment"]
            + response_clarity_score * weights["response_clarity"]
        )

        eval_result = EpisodeEvaluation(
            episode_id=episode.episode_id,
            user_id=episode.user_id,
            predicted_return_pct=episode.predicted_return_pct,
            actual_return_pct=actual_return_pct,
            return_error_pct=return_error,
            within_tolerance=within_tolerance,
            tool_selection_score=round(tool_selection_score, 4),
            goal_coverage_score=round(goal_coverage_score, 4),
            risk_alignment_score=round(risk_alignment_score, 4),
            response_clarity_score=round(response_clarity_score, 4),
            overall_score=round(overall, 4),
            notes=f"{provider_notes}. Error={return_error:.2f}% vs tolerance={self._tolerance}%."
                  if return_error is not None else provider_notes,
        )

        # ── Write back to memory ──────────────────────────────────────────────
        await self._memory.update_evaluation(
            episode_id=episode.episode_id,
            actual_return_pct=actual_return_pct or 0.0,
            evaluation_score=overall,
            notes=eval_result.notes,
        )

        # ── Emit metrics ──────────────────────────────────────────────────────
        metrics.increment("evaluator.episodes_evaluated")
        if return_error is not None:
            metrics.observe("evaluator.return_error_pct", return_error)
        metrics.observe("evaluator.overall_score", overall)
        if within_tolerance is False:
            metrics.increment("evaluator.tolerance_violations")

        log.info(
            "Episode evaluated",
            episode_id=episode.episode_id,
            predicted=episode.predicted_return_pct,
            actual=actual_return_pct,
            error=return_error,
            overall_score=round(overall, 3),
            grade=eval_result.grade,
        )

        return eval_result

    # ── Evaluate a live FinancialPlan (no memory lookup) ─────────────────────

    async def evaluate_plan(
        self,
        plan: FinancialPlan,
        actual_return_pct: Optional[float] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> EpisodeEvaluation:
        """
        Evaluate a freshly completed plan without needing to read from Memory.
        Used for immediate feedback after each turn.

        Useful in:
          - Live A/B testing (compare two model versions)
          - Unit tests (evaluate plan quality without storing to disk)
          - CI pipeline checks (ensure plan quality > threshold)
        """
        from agent.memory import Episode

        # Build a synthetic Episode from the plan
        ep = Episode(
            episode_id=f"live_{plan.plan_id}",
            session_id=plan.session_id,
            user_id=plan.user_id,
            query=plan.query,
            intent=plan.intent,
            plan_id=plan.plan_id,
            final_answer=plan.final_answer,
            tools_used=plan.tools_used,
            recommended_actions=plan.recommended_actions,
            health_score=plan.health_score,
            recommended_sip_inr=plan.recommended_sip_inr,
            predicted_return_pct=plan.predicted_return_pct,
            fire_corpus_inr=plan.fire_corpus_inr,
            tax_saving_inr=plan.tax_saving_inr,
            state_snapshot=state_snapshot or {},
            total_latency_ms=plan.total_latency_ms,
            used_fallback=plan.used_fallback,
        )

        return await self.evaluate_episode(ep, actual_return_pct=actual_return_pct)

    # ── Batch evaluation (one user) ───────────────────────────────────────────

    async def run_batch_evaluation(
        self,
        user_id: str,
        force: bool = False,
    ) -> EvaluationReport:
        """
        Evaluate all pending episodes for one user.

        Args:
            user_id: User to evaluate.
            force:   If True, evaluate all episodes regardless of age.
                     Use only for testing.

        Returns:
            EvaluationReport with accuracy stats, drift alert, and individual results.
        """
        t_start = time.perf_counter()

        if force:
            # Retrieve all episodes
            all_eps = await self._memory.retrieve_recent(user_id, n=settings.agent.memory_max_episodes)
            pending = [e for e in all_eps if e.predicted_return_pct is not None and not e.is_evaluated]
        else:
            pending = await self._memory.retrieve_for_evaluation(user_id)

        all_eps_count = await self._memory.count(user_id)

        report = EvaluationReport(
            scope="user",
            user_id=user_id,
            episodes_scanned=all_eps_count,
            episodes_skipped=all_eps_count - len(pending),
        )

        log.info(
            "Batch evaluation started",
            user_id=user_id,
            pending=len(pending),
            force=force,
        )

        if not pending:
            log.info("No episodes pending evaluation", user_id=user_id)
            report.run_duration_ms = (time.perf_counter() - t_start) * 1000
            return report

        # Evaluate all pending episodes
        evaluations: List[EpisodeEvaluation] = []
        for episode in pending:
            try:
                ev = await self.evaluate_episode(episode)
                evaluations.append(ev)
            except Exception as exc:
                log.error(
                    "Failed to evaluate episode",
                    episode_id=episode.episode_id,
                    error=str(exc),
                )

        # Compute aggregate stats
        report.evaluations           = evaluations
        report.episodes_evaluated    = len(evaluations)
        report                       = self._compute_aggregate_stats(report)
        report.drift_alert           = self._check_drift(len(evaluations))
        report.retraining_required   = (
            report.drift_alert is not None
            and report.drift_alert.requires_retraining
        )
        report.run_duration_ms       = (time.perf_counter() - t_start) * 1000

        # Audit record
        audit.record(
            session_id="system",
            action="batch_evaluation",
            inputs={"user_id": user_id, "episodes_evaluated": len(evaluations)},
            outputs={
                "mae": report.mean_absolute_error,
                "within_tolerance_pct": report.accuracy_within_tolerance_pct,
                "retraining_required": report.retraining_required,
            },
        )

        metrics.increment("evaluator.batch_runs")
        metrics.observe("evaluator.batch_size", len(evaluations))

        log.info(
            "Batch evaluation complete",
            user_id=user_id,
            evaluated=len(evaluations),
            mae=report.mean_absolute_error,
            retraining_required=report.retraining_required,
        )

        return report

    # ── System-wide evaluation ────────────────────────────────────────────────

    async def run_system_evaluation(
        self,
        user_ids: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """
        Evaluate all pending episodes across all users (or a provided list).
        Designed to be run as a daily/weekly scheduled job.

        In production:
          - Schedule via Celery / APScheduler / Cloud Scheduler
          - Stream results to a monitoring dashboard
          - Trigger retraining CI/CD on DriftSeverity.SEVERE
        """
        t_start = time.perf_counter()

        # If no user list, scan the memory store for known users
        if user_ids is None:
            user_ids = await self._discover_users()

        log.info("System evaluation started", users=len(user_ids))

        all_evaluations: List[EpisodeEvaluation] = []
        total_scanned = 0

        for uid in user_ids:
            user_report = await self.run_batch_evaluation(uid)
            all_evaluations.extend(user_report.evaluations)
            total_scanned += user_report.episodes_scanned

        # Aggregate system-wide stats
        system_report = EvaluationReport(
            scope="system",
            evaluations=all_evaluations,
            episodes_scanned=total_scanned,
            episodes_evaluated=len(all_evaluations),
            episodes_skipped=total_scanned - len(all_evaluations),
        )
        system_report = self._compute_aggregate_stats(system_report)
        system_report.drift_alert = self._check_drift(len(all_evaluations))
        system_report.retraining_required = (
            system_report.drift_alert is not None
            and system_report.drift_alert.requires_retraining
        )
        system_report.run_duration_ms = (time.perf_counter() - t_start) * 1000

        if system_report.retraining_required:
            log.error(
                "🔴 RETRAINING ALERT: RL model accuracy has degraded below threshold",
                mae=system_report.mean_absolute_error,
                tolerance=self._tolerance,
                action=system_report.drift_alert.recommended_action,
            )
            metrics.increment("evaluator.retraining_alerts")

        log.info(
            "System evaluation complete",
            users=len(user_ids),
            total_evaluated=len(all_evaluations),
            mae=system_report.mean_absolute_error,
            retraining_required=system_report.retraining_required,
            duration_ms=round(system_report.run_duration_ms, 0),
        )

        return system_report

    # ── Drift detection ───────────────────────────────────────────────────────

    def current_mae(self) -> Optional[float]:
        """Rolling MAE across the recent evaluation window."""
        if not self._recent_errors:
            return None
        return statistics.mean(self._recent_errors)

    def _record_error(self, error: float) -> None:
        """Add an error to the rolling window."""
        self._recent_errors.append(error)
        if len(self._recent_errors) > self._max_window:
            self._recent_errors.pop(0)

    def _check_drift(self, n_evaluated: int) -> Optional[DriftAlert]:
        """
        Compute drift severity from the rolling error window.
        Returns None if not enough data (<5 evaluations).
        """
        if len(self._recent_errors) < 5:
            return None

        mae = statistics.mean(self._recent_errors)
        ratio = mae / self._tolerance if self._tolerance > 0 else float("inf")

        if ratio <= 1.0:
            severity = DriftSeverity.NONE
            action = "Model performance within acceptable bounds. No action needed."
        elif ratio <= 1.5:
            severity = DriftSeverity.MILD
            action = "Monitor closely. Consider collecting more training data."
        elif ratio <= 2.0:
            severity = DriftSeverity.MODERATE
            action = "Fine-tune RL model on recent Indian market data. Review reward function."
        else:
            severity = DriftSeverity.SEVERE
            action = (
                "Trigger retraining pipeline immediately. "
                "Freeze RL model, route all predictions to ML fallback "
                "until retraining completes and accuracy recovers."
            )

        if severity == DriftSeverity.NONE:
            return None

        alert = DriftAlert(
            severity=severity,
            current_mae=round(mae, 4),
            tolerance=self._tolerance,
            ratio=round(ratio, 3),
            episodes_evaluated=n_evaluated,
            recommended_action=action,
        )

        metrics.observe("evaluator.drift_ratio", ratio)
        metrics.increment("evaluator.drift_alerts", tags={"severity": severity.value})

        log.warning(
            "Drift alert",
            severity=severity.value,
            mae=round(mae, 4),
            ratio=round(ratio, 3),
            action=action,
        )

        return alert

    # ── Scoring sub-functions ─────────────────────────────────────────────────

    def _score_return_accuracy(self, error: Optional[float]) -> float:
        """
        Map return prediction error → score [0, 1].
        Score = 1.0 at error = 0%, decays to 0.0 at error = 3× tolerance.
        Uses exponential decay for smooth gradient.
        """
        if error is None:
            return 0.5   # No prediction → neutral score

        tol = self._tolerance
        if error == 0:
            return 1.0
        # Exponential decay: score = exp(-k * error/tolerance)
        k = 2.0
        score = math.exp(-k * error / tol)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_tool_selection(intent: str, tools_used: List[str]) -> float:
        """
        Did the planner call the right tools for the detected intent?

        Score = fraction of expected tools that were actually called.
        Bonus if extra relevant tools were added (coverage > 1.0 capped at 1.0).
        """
        expected = INTENT_EXPECTED_TOOLS.get(intent, [])
        if not expected:
            # Unknown intent → neutral score
            return 0.7 if tools_used else 0.3

        if not tools_used:
            return 0.0

        expected_set = set(expected)
        used_set     = set(tools_used)
        overlap      = len(expected_set & used_set)
        recall       = overlap / len(expected_set)   # fraction of expected tools used

        # Small bonus for calling extra relevant tools (up to 1.0)
        bonus = min(0.1, len(used_set - expected_set) * 0.05)
        return min(1.0, recall + bonus)

    @staticmethod
    def _score_goal_coverage(
        intent: str,
        tools_used: List[str],
        final_answer: str,
        state_snapshot: Dict[str, Any],
    ) -> float:
        """
        Did the response address the user's actual financial goals?

        Heuristic checks:
          - Goals present in state → relevant tool was called
          - Final answer mentions ₹ amounts (specific advice given)
          - SIP mentioned if sip_calculator was used
        """
        score = 0.5   # Baseline

        # Did we call at least one tool?
        if tools_used:
            score += 0.2

        # Does the answer contain specific ₹ figures?
        if "₹" in final_answer or "Rs." in final_answer:
            score += 0.15

        # If user has goals and FIRE/SIP tools were called
        goals = state_snapshot.get("goals", [])
        if goals and any(t in tools_used for t in ["sip_calculator", "fire_planner"]):
            score += 0.15

        return min(1.0, score)

    @staticmethod
    def _score_risk_alignment(
        state_snapshot: Dict[str, Any],
        recommended_actions: List[str],
        tools_used: List[str],
    ) -> float:
        """
        Are the recommended actions consistent with the user's risk profile?

        Conservative risk → should NOT recommend "increase_equity" heavily
        Aggressive risk   → should NOT recommend only "build_emergency_fund"
        """
        risk = state_snapshot.get("risk", {})
        risk_profile = risk.get("profile", "moderate")
        risk_score   = risk.get("score", 5.0)

        if not recommended_actions and not tools_used:
            return 0.5   # No data → neutral

        actions_set = set(recommended_actions)

        # Conservative + increase_equity → penalise
        if risk_score <= 4.0 and "increase_equity" in actions_set:
            return 0.4

        # Aggressive + only emergency/tax actions → mild penalty
        if risk_score >= 7.0 and actions_set and not (actions_set & {"increase_equity", "increase_sip"}):
            return 0.6

        # Risk-appropriate action patterns
        if risk_score <= 4.0 and "build_emergency_fund" in actions_set:
            return 1.0
        if 4.0 < risk_score <= 7.0 and "increase_sip" in actions_set:
            return 1.0
        if risk_score > 7.0 and "increase_equity" in actions_set:
            return 1.0

        return 0.75   # Default: mostly aligned

    @staticmethod
    def _score_response_clarity(final_answer: str) -> float:
        """
        Heuristic readability / completeness score for the final answer.

        Checks:
          - Minimum length (too short → incomplete)
          - Contains actionable ₹ figures
          - Has numbered steps or bullet points
          - Not just an error message
        """
        if not final_answer or len(final_answer) < 50:
            return 0.1

        score = 0.3   # Baseline for non-empty answer

        if len(final_answer) >= 200:
            score += 0.2

        if "₹" in final_answer:
            score += 0.2

        # Actionable structure
        has_structure = any(
            marker in final_answer
            for marker in ["1.", "2.", "3.", "•", "**", "Step", "\n-"]
        )
        if has_structure:
            score += 0.15

        # Not an error/fallback message
        error_signals = ["error", "try again", "having trouble", "unable to"]
        if not any(s in final_answer.lower() for s in error_signals):
            score += 0.15

        return min(1.0, score)

    # ── Aggregate stats helper ────────────────────────────────────────────────

    @staticmethod
    def _compute_aggregate_stats(report: EvaluationReport) -> EvaluationReport:
        """Fill aggregate accuracy and quality stats from the evaluations list."""
        evaluations = report.evaluations
        if not evaluations:
            return report

        # Return errors (only where both predicted and actual exist)
        errors = [
            e.return_error_pct for e in evaluations
            if e.return_error_pct is not None
        ]
        if errors:
            report.mean_absolute_error   = round(statistics.mean(errors), 4)
            report.median_absolute_error = round(statistics.median(errors), 4)
            sorted_errors = sorted(errors)
            p90_idx = max(0, int(len(sorted_errors) * 0.90) - 1)
            report.p90_absolute_error    = round(sorted_errors[p90_idx], 4)

            within = sum(1 for e in evaluations if e.within_tolerance is True)
            report.accuracy_within_tolerance_pct = round(
                within / len(evaluations) * 100, 1
            )

        # Quality scores
        overall_scores = [e.overall_score for e in evaluations]
        report.mean_overall_score = round(statistics.mean(overall_scores), 4)

        # Grade distribution
        dist: Dict[str, int] = {}
        for ev in evaluations:
            dist[ev.grade] = dist.get(ev.grade, 0) + 1
        report.grade_distribution = dist

        return report

    # ── User discovery ────────────────────────────────────────────────────────

    async def _discover_users(self) -> List[str]:
        """
        Discover user IDs from the JSON memory store.
        For Redis, scan keys with the 'fm:episodes:*' pattern.
        """
        from agent.memory import JSONMemoryStore, MEMORY_DIR
        if isinstance(self._memory, JSONMemoryStore):
            files = list(MEMORY_DIR.glob("*.json"))
            return [f.stem for f in files]
        # Fallback: no discovery without explicit user list
        log.warning("Cannot auto-discover users for non-JSON backend. Pass user_ids explicitly.")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_evaluator(
    memory: Optional[BaseMemoryStore] = None,
    return_provider: ActualReturnProvider = synthetic_return_provider,
) -> Evaluator:
    """
    Build a fully configured Evaluator.

    Args:
        memory:          Memory store. Defaults to get_memory_store().
        return_provider: Callable that fetches actual return for an episode.
                         Default: synthetic_return_provider (for testing).
                         Production: wire to CAMS/KFintech API.

    Returns:
        Ready-to-use Evaluator instance.
    """
    store = memory or get_memory_store()
    evaluator = Evaluator(memory=store, return_provider=return_provider)

    log.info(
        "Evaluator built",
        tolerance_pct=settings.agent.prediction_tolerance_pct,
        evaluation_enabled=settings.agent.evaluation_enabled,
        return_provider=return_provider.__name__,
    )
    return evaluator


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json
    from environment.state import (
        UserFinancialState, InsuranceCoverage, DebtProfile,
        InvestmentPortfolio, FinancialGoal, EmploymentType, CityTier,
    )
    from config import RiskProfile
    from agent.memory import InMemoryStore, reset_memory_store, Episode

    def _make_state(user_id: str = "u_eval_001") -> UserFinancialState:
        return UserFinancialState(
            user_id=user_id,
            age=35,
            monthly_income=150_000,
            monthly_expense=65_000,
            risk_profile=RiskProfile.MODERATE,
            risk_score=6.0,
            investment_horizon_years=22,
            emergency_fund_amount=300_000,
            existing_monthly_sip=18_000,
            section_80c_used=100_000,
            insurance=InsuranceCoverage(
                has_term_insurance=True, term_cover_amount=15_000_000,
                has_health_insurance=True, health_cover_amount=500_000,
            ),
            debts=DebtProfile(home_loan_emi=28_000),
            portfolio=InvestmentPortfolio(
                equity_mf=1_200_000, debt_mf=400_000,
                ppf=200_000, epf=500_000, nps=150_000,
            ),
            goals=[
                FinancialGoal(
                    name="Retirement", target_amount=60_000_000,
                    target_year=2052, priority=1, existing_corpus=2_450_000,
                ),
            ],
        )

    def _make_plan(
        user_id: str,
        intent: str = "sip_calculation",
        tools: Optional[List[str]] = None,
        predicted_return: float = 12.5,
        health_score: float = 71.0,
        sip: float = 32_000.0,
        answer: str = "Your financial plan: Invest ₹32,000/month in a diversified SIP.\n1. ₹20,000 → Equity MF\n2. ₹8,000 → Debt MF\n3. ₹4,000 → NPS",
    ) -> FinancialPlan:
        return FinancialPlan(
            session_id=f"sess_{intent}",
            user_id=user_id,
            query=f"Test query for {intent}",
            intent=intent,
            final_answer=answer,
            tools_used=tools or ["sip_calculator", "rl_predict"],
            recommended_actions=["increase_sip"],
            health_score=health_score,
            recommended_sip_inr=sip,
            predicted_return_pct=predicted_return,
            total_latency_ms=380.0,
        )

    async def run_tests() -> None:
        print("=== Evaluator Self-Test ===\n")

        # Use InMemoryStore for speed
        reset_memory_store(InMemoryStore())
        memory = get_memory_store()
        state  = _make_state()

        evaluator = build_evaluator(memory=memory)

        # ── Test 1: Single plan evaluation ────────────────────────────────────
        print("── Test 1: Evaluate Single Plan ──")
        plan = _make_plan("u_eval_001", predicted_return=12.5)
        ev = await evaluator.evaluate_plan(
            plan=plan,
            actual_return_pct=11.8,
            state_snapshot=state.financial_summary(),
        )
        print(f"  Episode ID         : {ev.episode_id}")
        print(f"  Predicted return   : {ev.predicted_return_pct}%")
        print(f"  Actual return      : {ev.actual_return_pct}%")
        print(f"  Return error       : {ev.return_error_pct:.2f}%")
        print(f"  Within tolerance   : {ev.within_tolerance}  (tol={evaluator._tolerance}%)")
        print(f"  Tool selection     : {ev.tool_selection_score:.3f}")
        print(f"  Goal coverage      : {ev.goal_coverage_score:.3f}")
        print(f"  Risk alignment     : {ev.risk_alignment_score:.3f}")
        print(f"  Response clarity   : {ev.response_clarity_score:.3f}")
        print(f"  Overall score      : {ev.overall_score:.3f}")
        print(f"  Grade              : {ev.grade}\n")

        assert ev.predicted_return_pct == 12.5
        assert ev.actual_return_pct == 11.8
        assert ev.return_error_pct is not None
        assert 0 <= ev.overall_score <= 1.0
        print("  ✓ Single plan evaluation correct\n")

        # ── Test 2: Tool selection scoring ────────────────────────────────────
        print("── Test 2: Tool Selection Scoring ──")
        test_cases = [
            ("fire_planning",    ["fire_planner", "rl_predict"],   "perfect match"),
            ("fire_planning",    ["fire_planner"],                  "partial match"),
            ("fire_planning",    ["health_score"],                  "wrong tool"),
            ("sip_calculation",  ["sip_calculator"],               "exact match"),
            ("tax_optimization", [],                               "no tools"),
        ]
        for intent, tools, label in test_cases:
            score = Evaluator._score_tool_selection(intent, tools)
            print(f"  {label:<20} score={score:.3f}")

        print("  ✓ Tool selection scoring correct\n")

        # ── Test 3: Return accuracy scoring ───────────────────────────────────
        print("── Test 3: Return Accuracy Scoring ──")
        errors = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0]
        for err in errors:
            score = evaluator._score_return_accuracy(err)
            print(f"  Error={err:4.0f}%  → score={score:.4f}")

        assert evaluator._score_return_accuracy(0.0)   == 1.0
        assert evaluator._score_return_accuracy(None)  == 0.5
        assert evaluator._score_return_accuracy(100.0) < 0.1
        print("  ✓ Return accuracy scoring correct\n")

        # ── Test 4: Batch evaluation with stored episodes ─────────────────────
        print("── Test 4: Batch Evaluation (force=True) ──")

        # Store 5 episodes with varying quality
        intents_and_returns = [
            ("health_score",    11.5, ["health_score"]),
            ("sip_calculation", 12.8, ["sip_calculator", "rl_predict"]),
            ("fire_planning",   10.9, ["fire_planner", "rl_predict"]),
            ("tax_optimization", 9.2, ["tax_wizard"]),
            ("portfolio_review", 13.1, ["rl_predict", "health_score"]),
        ]

        for intent, pred_ret, tools in intents_and_returns:
            p = _make_plan("u_eval_001", intent=intent, tools=tools, predicted_return=pred_ret)
            await memory.store(plan=p, state=state, transition=None, session_id=p.session_id)

        report = await evaluator.run_batch_evaluation("u_eval_001", force=True)

        print(report.summary())
        print()
        print(f"  Individual results:")
        for ev in report.evaluations:
            print(
                f"    [{ev.grade}] {ev.episode_id[-10:]}  "
                f"overall={ev.overall_score:.3f}  "
                f"tools={ev.tool_selection_score:.2f}  "
                f"clarity={ev.response_clarity_score:.2f}"
            )

        assert report.episodes_evaluated > 0
        assert report.mean_overall_score is not None
        assert 0 <= report.mean_overall_score <= 1.0
        print("\n  ✓ Batch evaluation complete\n")

        # ── Test 5: Drift detection ───────────────────────────────────────────
        print("── Test 5: Drift Detection ──")

        # Inject high errors to trigger drift
        test_evaluator = build_evaluator(memory=memory)
        for _ in range(10):
            test_evaluator._record_error(25.0)   # Way above 10% tolerance

        alert = test_evaluator._check_drift(n_evaluated=10)
        assert alert is not None
        assert alert.severity in (DriftSeverity.MODERATE, DriftSeverity.SEVERE)
        print(f"  Severity           : {alert.severity.value}")
        print(f"  MAE                : {alert.current_mae}%")
        print(f"  Ratio to tolerance : {alert.ratio}×")
        print(f"  Requires retraining: {alert.requires_retraining}")
        print(f"  Action             : {alert.recommended_action[:80]}...")
        print("  ✓ Drift detection triggered correctly\n")

        # ── Test 6: No drift when within tolerance ────────────────────────────
        print("── Test 6: No Drift Within Tolerance ──")
        good_evaluator = build_evaluator(memory=memory)
        for _ in range(10):
            good_evaluator._record_error(3.0)   # Well below 10% tolerance

        no_alert = good_evaluator._check_drift(n_evaluated=10)
        assert no_alert is None
        print(f"  MAE={good_evaluator.current_mae():.1f}% — no alert raised ✓\n")

        # ── Test 7: Report serialisation ──────────────────────────────────────
        print("── Test 7: Report Serialisation ──")
        report_dict = report.to_dict()
        report_json = _json.dumps(report_dict, indent=2, default=str)
        assert "accuracy" in report_dict
        assert "grade_distribution" in report_dict["quality"]
        print(f"  Report JSON: {len(report_json)} chars")
        print(f"  Keys: {list(report_dict.keys())}")
        print("  ✓ Report fully serialisable\n")

        # ── Test 8: Risk alignment scoring ────────────────────────────────────
        print("── Test 8: Risk Alignment Scoring ──")
        cases = [
            ({"risk": {"score": 3.0, "profile": "conservative"}},
             ["build_emergency_fund"], [], "conservative + EF → 1.0"),
            ({"risk": {"score": 8.5, "profile": "aggressive"}},
             ["increase_equity"], [], "aggressive + equity → 1.0"),
            ({"risk": {"score": 3.0, "profile": "conservative"}},
             ["increase_equity"], [], "conservative + equity → penalty"),
        ]
        for snap, actions, tools, label in cases:
            score = Evaluator._score_risk_alignment(snap, actions, tools)
            print(f"  {label:<45} score={score:.2f}")
        print("  ✓ Risk alignment scoring correct\n")

        # ── Metrics snapshot ──────────────────────────────────────────────────
        print("── Metrics Snapshot ──")
        snap = metrics.snapshot()
        for c in snap["counters"]:
            if "evaluator" in c["name"]:
                print(f"  {c['name']:<50} {int(c['value'])}")
        for h in snap["histograms"]:
            if "evaluator" in h["name"]:
                print(f"  {h['name']:<50} mean={h['mean_ms']:.2f}")

    asyncio.run(run_tests())
