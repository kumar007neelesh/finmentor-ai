"""
tools/registry.py — FinMentor AI Tool Registry
===============================================
Central dispatch layer between the LLM Planner and all callable tools.

Responsibilities:
  - Register tools with a name, schema, description, and callable
  - Validate tool inputs via Pydantic schemas before execution
  - Execute the RL → ML fallback chain automatically on LowConfidenceError
  - Run multiple tool calls in parallel via asyncio (when planner batches them)
  - Emit metrics for every tool call (latency, errors, usage counts)
  - Return a standardised ToolResult to the planner regardless of which
    underlying model or calculator produced the answer

Architecture position:
    LLM Planner  →  ToolRegistry.execute("sip_calculator", args)
                          │
                ┌─────────┴──────────────────────────┐
                │  Route to registered tool callable  │
                └────────┬───────────────────────────┘
                         │
              ┌──────────┴──────────────────────────────┐
              │  rl_predict   │  sip_calculator          │
              │  fire_planner │  tax_wizard              │
              │  health_score │  ml_predict (fallback)   │
              └──────────────────────────────────────────┘

Usage:
    from tools.registry import ToolRegistry, build_registry

    registry = build_registry()

    result = await registry.execute(
        tool_name="sip_calculator",
        args={"state": state, "goal_name": "Retirement", "target_amount": 50_000_000},
        session_id="sess_001",
    )
    print(result.output)      # dict with sip_amount, horizon, corpus_projection
    print(result.success)     # True / False
    print(result.latency_ms)  # 12.4
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from config import settings
from environment.state import UserFinancialState
from logger import get_logger, metrics, timed, audit
from models.rl_loader import (
    RLModelLoader,
    PredictionResult,
    LowConfidenceError,
    InferenceError,
    build_rl_loader,
)

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Tool result — standard output envelope
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """
    Standardised output from every tool call.
    The LLM Planner only ever receives ToolResult — never raw tool output.
    """
    tool_name: str
    success: bool
    output: Dict[str, Any]          # The actual result payload
    error: Optional[str] = None     # Human-readable error if success=False
    latency_ms: float = 0.0
    used_fallback: bool = False     # True if ML fallback replaced RL
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":         self.tool_name,
            "success":      self.success,
            "output":       self.output,
            "error":        self.error,
            "latency_ms":   round(self.latency_ms, 2),
            "used_fallback": self.used_fallback,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Tool input schemas (Pydantic-validated before execution)
# ══════════════════════════════════════════════════════════════════════════════

class RLPredictInput(BaseModel):
    state: Any  # UserFinancialState — typed as Any to avoid circular import issues


class SIPCalculatorInput(BaseModel):
    state: Any
    goal_name: str = Field(..., min_length=1, max_length=100)
    target_amount: float = Field(..., gt=0, description="Target corpus in ₹")
    horizon_years: Optional[float] = Field(
        default=None, gt=0,
        description="Override state's investment horizon if provided",
    )
    expected_return_pct: Optional[float] = Field(
        default=None, gt=0, lt=30,
        description="Expected annual return %. Defaults to config value.",
    )
    step_up_pct: float = Field(
        default=10.0, ge=0, le=30,
        description="Annual SIP step-up % (increase SIP each year by this %)",
    )


class FIREPlannerInput(BaseModel):
    state: Any
    target_retirement_age: Optional[int] = Field(default=None, ge=30, le=75)
    desired_monthly_expense_at_retirement: Optional[float] = Field(
        default=None, gt=0,
        description="Monthly expense in ₹ at retirement (today's value). "
                    "Defaults to current monthly_expense.",
    )


class TaxWizardInput(BaseModel):
    state: Any
    bonus_amount: float = Field(default=0.0, ge=0)
    compare_regimes: bool = Field(default=True)


class HealthScoreInput(BaseModel):
    state: Any


class MLPredictInput(BaseModel):
    state: Any
    reason: str = Field(
        default="rl_low_confidence",
        description="Why ML fallback was triggered.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tool descriptor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolDescriptor:
    """
    Metadata the LLM Planner uses to decide WHICH tool to call.
    The description is injected verbatim into the planner's system prompt.
    """
    name: str
    description: str
    input_schema: Type[BaseModel]
    callable_fn: Callable[..., Coroutine[Any, Any, ToolResult]]  # async
    tags: List[str] = field(default_factory=list)
    requires_rl: bool = False      # True → automatic RL→ML fallback chain
    enabled: bool = True

    def schema_for_llm(self) -> Dict[str, Any]:
        """Return a compact schema dict for injection into the LLM system prompt."""
        return {
            "name":        self.name,
            "description": self.description,
            "tags":        self.tags,
            "input_fields": list(self.input_schema.schema().get("properties", {}).keys()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Financial calculation helpers (pure functions — no LLM, no model)
# ══════════════════════════════════════════════════════════════════════════════

def _future_value_sip(monthly_sip: float, annual_return_pct: float, years: float) -> float:
    """Future value of a fixed monthly SIP using compound interest formula."""
    r = annual_return_pct / 100.0 / 12.0   # Monthly rate
    n = int(years * 12)                     # Months
    if r == 0:
        return monthly_sip * n
    return monthly_sip * (((1 + r) ** n - 1) / r) * (1 + r)


def _required_sip(target: float, annual_return_pct: float, years: float) -> float:
    """Reverse SIP formula: how much SIP do I need to reach a target corpus?"""
    r = annual_return_pct / 100.0 / 12.0
    n = int(years * 12)
    if r == 0 or n == 0:
        return target / max(n, 1)
    return target * r / (((1 + r) ** n - 1) * (1 + r))


def _step_up_sip_corpus(
    initial_sip: float,
    annual_return_pct: float,
    years: int,
    step_up_pct: float,
) -> float:
    """
    Future value of a SIP with annual step-up (increase SIP each year by step_up_pct%).
    Computed year by year for accuracy.
    """
    r_monthly = annual_return_pct / 100.0 / 12.0
    corpus = 0.0
    current_sip = initial_sip
    for _ in range(years):
        # Grow this year's SIP for 12 months, add to cumulative corpus
        corpus = (corpus + current_sip * 12) * (1 + annual_return_pct / 100.0)
        current_sip *= 1 + step_up_pct / 100.0
    return corpus


def _inflation_adjusted(amount: float, inflation_pct: float, years: float) -> float:
    return amount * ((1 + inflation_pct / 100.0) ** years)


def _effective_tax(annual_income: float, regime: str) -> float:
    """Simplified effective tax (same logic as state.effective_tax_rate)."""
    brackets = [
        (300_000, 0.00),
        (400_000, 0.05),
        (300_000, 0.10),
        (200_000, 0.15),
        (300_000, 0.20),
        (float("inf"), 0.30),
    ]
    tax = 0.0
    remaining = annual_income
    for size, rate in brackets:
        chunk = min(remaining, size)
        tax += chunk * rate
        remaining -= chunk
        if remaining <= 0:
            break
    if regime == "new" and annual_income <= 700_000:
        tax = 0.0
    return tax


# ══════════════════════════════════════════════════════════════════════════════
# Individual tool implementations (async callables)
# ══════════════════════════════════════════════════════════════════════════════

async def _tool_sip_calculator(
    state: UserFinancialState,
    goal_name: str,
    target_amount: float,
    horizon_years: Optional[float],
    expected_return_pct: Optional[float],
    step_up_pct: float,
) -> Dict[str, Any]:
    """
    Computes the required monthly SIP to hit a financial goal.
    Also projects corpus if user continues existing SIP unchanged.
    """
    fin = settings.finance
    horizon = horizon_years or state.investment_horizon_years
    ret_pct = expected_return_pct or (
        fin.default_equity_return_pct
        if state.risk_score >= 6
        else (fin.default_equity_return_pct + fin.default_debt_return_pct) / 2
    )

    # Existing corpus growth
    existing_corpus = sum(
        g.existing_corpus for g in state.goals if g.name == goal_name
    ) or 0.0
    existing_grown = existing_corpus * ((1 + ret_pct / 100) ** horizon)

    # Adjusted target (inflation)
    inflation_adjusted_target = _inflation_adjusted(
        target_amount, fin.default_inflation_pct, horizon
    )

    # Required SIP (fixed)
    gap = max(0.0, inflation_adjusted_target - existing_grown)
    required_sip = _required_sip(gap, ret_pct, horizon)

    # Step-up SIP (more realistic)
    step_up_corpus = _step_up_sip_corpus(
        required_sip * 0.7,   # Start lower, step up each year
        ret_pct,
        int(horizon),
        step_up_pct,
    ) + existing_grown

    # What existing SIP will produce
    current_sip_corpus = (
        _future_value_sip(state.existing_monthly_sip, ret_pct, horizon)
        + existing_grown
    )

    surplus_or_deficit = current_sip_corpus - inflation_adjusted_target

    return {
        "goal_name":                     goal_name,
        "target_amount_today_inr":       round(target_amount, 0),
        "inflation_adjusted_target_inr": round(inflation_adjusted_target, 0),
        "investment_horizon_years":      horizon,
        "assumed_return_pct":            ret_pct,
        "existing_corpus_inr":           round(existing_corpus, 0),
        "required_monthly_sip_inr":      round(required_sip, 0),
        "step_up_sip_start_inr":         round(required_sip * 0.7, 0),
        "step_up_pct_annual":            step_up_pct,
        "step_up_projected_corpus_inr":  round(step_up_corpus, 0),
        "current_sip_projected_corpus_inr": round(current_sip_corpus, 0),
        "corpus_surplus_deficit_inr":    round(surplus_or_deficit, 0),
        "goal_achievable_with_current_sip": surplus_or_deficit >= 0,
        "recommendation": (
            f"Increase SIP to ₹{required_sip:,.0f}/month to meet your "
            f"'{goal_name}' goal of ₹{inflation_adjusted_target:,.0f} "
            f"in {int(horizon)} years."
        ) if surplus_or_deficit < 0 else (
            f"Your current SIP is on track for '{goal_name}'. "
            f"Projected surplus: ₹{surplus_or_deficit:,.0f}."
        ),
    }


async def _tool_fire_planner(
    state: UserFinancialState,
    target_retirement_age: Optional[int],
    desired_monthly_expense_at_retirement: Optional[float],
) -> Dict[str, Any]:
    """
    FIRE (Financial Independence, Retire Early) path planner.
    Computes the corpus needed and monthly SIP required to retire by target age.
    """
    fin = settings.finance
    ret_age = target_retirement_age or state.retirement_age
    years_to_fire = max(1, ret_age - state.age)
    monthly_exp = desired_monthly_expense_at_retirement or state.monthly_expense

    # Inflate expenses to retirement date
    annual_exp_at_retirement = _inflation_adjusted(
        monthly_exp * 12, fin.default_inflation_pct, years_to_fire
    )

    # FIRE corpus = 25× annual expenses (4% withdrawal rate rule)
    fire_corpus_needed = annual_exp_at_retirement * fin.fire_multiplier

    # Existing portfolio growth
    ret_pct = fin.default_equity_return_pct   # Assumed aggressive pre-retirement
    existing_portfolio_grown = state.portfolio.total_value * (
        (1 + ret_pct / 100) ** years_to_fire
    )

    gap = max(0.0, fire_corpus_needed - existing_portfolio_grown)
    required_sip = _required_sip(gap, ret_pct, years_to_fire)

    # Monthly surplus available for investment
    emi = state.monthly_emi if state.monthly_emi > 0 else state.debts.total_emi
    monthly_investable = max(0.0, state.monthly_income - state.monthly_expense - emi)
    sip_shortfall = max(0.0, required_sip - monthly_investable)

    # Post-retirement: how long corpus lasts (at 6% debt return, 6% inflation)
    post_ret_return = fin.default_debt_return_pct
    real_return = post_ret_return - fin.default_inflation_pct   # ~1% real
    if real_return <= 0:
        corpus_longevity_years = fire_corpus_needed / annual_exp_at_retirement
    else:
        r = real_return / 100
        corpus_longevity_years = -np.log(1 - (fire_corpus_needed * r / annual_exp_at_retirement)) / np.log(1 + r)

    # Monthly asset allocation roadmap
    equity_pct = max(30, 100 - state.age)   # 100-age rule
    debt_pct   = 100 - equity_pct

    return {
        "fire_age":                        ret_age,
        "years_to_fire":                   years_to_fire,
        "current_portfolio_inr":           round(state.portfolio.total_value, 0),
        "fire_corpus_needed_inr":          round(fire_corpus_needed, 0),
        "existing_portfolio_at_fire_inr":  round(existing_portfolio_grown, 0),
        "corpus_gap_inr":                  round(gap, 0),
        "required_monthly_sip_inr":        round(required_sip, 0),
        "monthly_investable_surplus_inr":  round(monthly_investable, 0),
        "sip_shortfall_inr":               round(sip_shortfall, 0),
        "fire_feasible":                   sip_shortfall == 0,
        "corpus_longevity_years":          round(corpus_longevity_years, 1),
        "post_fire_monthly_income_inr":    round(annual_exp_at_retirement / 12, 0),
        "recommended_allocation": {
            "equity_pct": equity_pct,
            "debt_pct":   debt_pct,
        },
        "monthly_roadmap": {
            "step_1_emergency_fund_inr":  max(0.0, round(
                state.monthly_expense * 6 - state.emergency_fund_amount, 0
            )),
            "step_2_term_insurance": not state.insurance.has_term_insurance,
            "step_3_sip_inr": round(required_sip, 0),
            "step_4_review_frequency": "Annually",
        },
        "recommendation": (
            f"To FIRE at {ret_age}, invest ₹{required_sip:,.0f}/month for "
            f"{years_to_fire} years. Your corpus of ₹{fire_corpus_needed:,.0f} "
            f"should last ~{corpus_longevity_years:.0f} years post-retirement."
        ),
    }


async def _tool_tax_wizard(
    state: UserFinancialState,
    bonus_amount: float,
    compare_regimes: bool,
) -> Dict[str, Any]:
    """
    Identifies missed deductions and computes old vs new regime savings.
    """
    fin = settings.finance
    annual_income = state.annual_income or state.monthly_income * 12
    total_income = annual_income + bonus_amount

    # ── Old regime (with deductions) ──────────────────────────────────────────
    deductions_80c   = min(state.section_80c_used, fin.section_80c_limit)
    deductions_80d   = min(state.section_80d_used, fin.section_80d_self_limit)
    nps_deduction    = min(state.nps_contribution * 12, fin.nps_80ccd1b_limit)
    std_deduction    = 50_000   # Standard deduction (old regime)

    total_old_deductions = deductions_80c + deductions_80d + nps_deduction + std_deduction
    old_regime_taxable   = max(0, total_income - total_old_deductions)
    old_regime_tax       = _effective_tax(old_regime_taxable, "old")

    # ── New regime (fewer deductions, lower slabs) ────────────────────────────
    new_std_deduction  = 75_000  # Enhanced standard deduction FY25
    new_regime_taxable = max(0, total_income - new_std_deduction)
    new_regime_tax     = _effective_tax(new_regime_taxable, "new")

    # ── Missed deductions (opportunities) ────────────────────────────────────
    missed = []
    remaining_80c = fin.section_80c_limit - deductions_80c
    if remaining_80c > 0:
        missed.append({
            "section": "80C",
            "potential_saving_inr": round(remaining_80c * state.effective_tax_rate, 0),
            "headroom_inr": round(remaining_80c, 0),
            "suggestion": f"Invest ₹{remaining_80c:,.0f} more in ELSS/PPF/LIC to max 80C.",
        })

    remaining_nps = fin.nps_80ccd1b_limit - (state.nps_contribution * 12)
    if remaining_nps > 0 and state.employment_type.value == "salaried":
        missed.append({
            "section": "80CCD(1B)",
            "potential_saving_inr": round(remaining_nps * state.effective_tax_rate, 0),
            "headroom_inr": round(remaining_nps, 0),
            "suggestion": f"Contribute ₹{remaining_nps/12:,.0f}/month more to NPS (Tier I).",
        })

    remaining_80d = fin.section_80d_self_limit - deductions_80d
    if remaining_80d > 0 and not state.insurance.has_health_insurance:
        missed.append({
            "section": "80D",
            "potential_saving_inr": round(remaining_80d * state.effective_tax_rate, 0),
            "headroom_inr": round(remaining_80d, 0),
            "suggestion": f"Buy health insurance to claim up to ₹{remaining_80d:,.0f} under 80D.",
        })

    better_regime = "new" if new_regime_tax <= old_regime_tax else "old"
    tax_saving = abs(old_regime_tax - new_regime_tax)

    return {
        "total_income_inr":         round(total_income, 0),
        "bonus_included_inr":       round(bonus_amount, 0),
        "old_regime": {
            "taxable_income_inr": round(old_regime_taxable, 0),
            "total_deductions_inr": round(total_old_deductions, 0),
            "tax_liability_inr":  round(old_regime_tax, 0),
            "effective_rate_pct": round(old_regime_tax / max(total_income, 1) * 100, 2),
        },
        "new_regime": {
            "taxable_income_inr": round(new_regime_taxable, 0),
            "total_deductions_inr": round(new_std_deduction, 0),
            "tax_liability_inr":  round(new_regime_tax, 0),
            "effective_rate_pct": round(new_regime_tax / max(total_income, 1) * 100, 2),
        },
        "recommended_regime":       better_regime,
        "tax_saving_by_switching_inr": round(tax_saving, 0),
        "missed_deductions":        missed,
        "total_recoverable_tax_inr": round(
            sum(m["potential_saving_inr"] for m in missed), 0
        ),
        "recommendation": (
            f"Switch to the {better_regime} regime to save ₹{tax_saving:,.0f}. "
            f"Additionally, ₹{sum(m['potential_saving_inr'] for m in missed):,.0f} "
            f"can be recovered via missed deductions."
        ),
    }


async def _tool_health_score(state: UserFinancialState) -> Dict[str, Any]:
    """
    Computes a 0–100 Money Health Score across 6 financial dimensions.
    """
    annual_income = state.annual_income or state.monthly_income * 12

    # 1. Emergency preparedness (max 20 pts)
    em_months = state.emergency_fund_months
    emergency_score = min(20, (em_months / 6) * 20)

    # 2. Insurance coverage (max 20 pts)
    insurance_score = state.insurance.coverage_score(annual_income) * 0.20

    # 3. Investment diversification (max 20 pts)
    alloc = state.portfolio.allocation_breakdown()
    n_assets_used = sum(1 for v in alloc.values() if v > 5)  # >5% = meaningful
    diversification_score = min(20, n_assets_used * 4)

    # 4. Debt health (max 20 pts)
    foir = state.foir
    if foir <= 0.20:    debt_score = 20
    elif foir <= 0.30:  debt_score = 15
    elif foir <= 0.40:  debt_score = 10
    elif foir <= 0.50:  debt_score = 5
    else:               debt_score = 0

    # 5. Tax efficiency (max 10 pts)
    utilisation = (state.section_80c_used / settings.finance.section_80c_limit)
    tax_score = min(10, utilisation * 10)

    # 6. Retirement readiness (max 10 pts)
    years_left = state.years_to_retirement
    corpus_needed = state.monthly_expense * 12 * settings.finance.fire_multiplier
    corpus_ratio  = state.portfolio.total_value / max(corpus_needed, 1)
    expected_ratio = (state.age - 23) / max(years_left + (state.age - 23), 1)
    on_track_ratio = min(1.0, corpus_ratio / max(expected_ratio, 0.01))
    retirement_score = min(10, on_track_ratio * 10)

    total = (
        emergency_score
        + insurance_score
        + diversification_score
        + debt_score
        + tax_score
        + retirement_score
    )

    def _grade(score: float) -> str:
        if score >= 80: return "Excellent"
        if score >= 60: return "Good"
        if score >= 40: return "Fair"
        return "Needs Attention"

    return {
        "overall_score":    round(total, 1),
        "grade":            _grade(total),
        "breakdown": {
            "emergency_preparedness": {
                "score": round(emergency_score, 1), "max": 20,
                "months_covered": round(em_months, 1),
                "status": "adequate" if em_months >= 6 else "low",
            },
            "insurance_coverage": {
                "score": round(insurance_score, 1), "max": 20,
                "has_term": state.insurance.has_term_insurance,
                "has_health": state.insurance.has_health_insurance,
            },
            "investment_diversification": {
                "score": round(diversification_score, 1), "max": 20,
                "asset_classes_used": n_assets_used,
                "allocation": alloc,
            },
            "debt_health": {
                "score": round(debt_score, 1), "max": 20,
                "foir": round(foir, 3),
                "status": "healthy" if foir < 0.40 else "high",
            },
            "tax_efficiency": {
                "score": round(tax_score, 1), "max": 10,
                "80c_utilisation_pct": round(utilisation * 100, 1),
            },
            "retirement_readiness": {
                "score": round(retirement_score, 1), "max": 10,
                "corpus_so_far_inr": round(state.portfolio.total_value, 0),
                "corpus_needed_inr": round(corpus_needed, 0),
                "on_track_pct": round(on_track_ratio * 100, 1),
            },
        },
        "priority_actions": _health_score_actions(
            emergency_score, insurance_score, debt_score, tax_score
        ),
        "recommendation": f"Your Money Health Score is {total:.0f}/100 ({_grade(total)}). "
                          f"Focus on the lowest-scoring dimensions first.",
    }


def _health_score_actions(em: float, ins: float, debt: float, tax: float) -> List[str]:
    actions = []
    if em < 10:
        actions.append("🚨 Build emergency fund to 6 months of expenses — top priority.")
    if ins < 10:
        actions.append("🛡️ Buy term insurance (10× annual income) and health cover.")
    if debt > 0 and debt < 10:
        actions.append("💳 Reduce FOIR below 40% by prepaying high-interest loans.")
    if tax < 5:
        actions.append("🧾 Max out 80C (₹1.5L), 80D, and NPS 80CCD(1B) deductions.")
    if not actions:
        actions.append("✅ Finances are in good shape. Focus on growing wealth.")
    return actions


async def _tool_ml_predict(
    state: UserFinancialState,
    reason: str,
) -> Dict[str, Any]:
    """
    ML fallback tool — rule-based heuristic when RL confidence is too low.
    In production, swap internals with a loaded XGBoost / LSTM model.
    Returns same shape as RL PredictionResult.to_dict() for planner consistency.
    """
    fin = settings.finance
    savings_rate = state.savings_rate

    # Heuristic return estimate based on allocation + risk
    equity_return = fin.default_equity_return_pct / 100
    debt_return   = fin.default_debt_return_pct / 100

    eq_pct  = state.portfolio.equity_pct / 100 or 0.6
    dbt_pct = state.portfolio.debt_pct / 100 or 0.4
    blended_return = eq_pct * equity_return + dbt_pct * debt_return

    # Risk adjustment
    risk_adj = (state.risk_score - 5) * 0.003
    predicted_return = max(0.0, blended_return + risk_adj)

    # Action heuristic (mirrors MockRLModel logic)
    if state.emergency_fund_months < 3:
        action = "build_emergency_fund"
        confidence = 0.85
    elif state.foir > 0.50:
        action = "reduce_equity"
        confidence = 0.80
    elif savings_rate < 0.10:
        action = "increase_sip"
        confidence = 0.78
    elif state.section_80c_used < settings.finance.section_80c_limit * 0.8:
        action = "optimize_tax"
        confidence = 0.75
    else:
        action = "increase_equity"
        confidence = 0.70

    return {
        "recommended_action":   action,
        "predicted_return_pct": round(predicted_return * 100, 2),
        "confidence":           confidence,
        "model_backend":        "ml_heuristic",
        "is_fallback":          True,
        "fallback_reason":      reason,
        "rationale": (
            f"ML fallback ({reason}): Blended return estimate "
            f"{predicted_return*100:.1f}% based on portfolio allocation."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    Central dispatcher for all FinMentor tools.

    The LLM Planner calls:
        result = await registry.execute("sip_calculator", args, session_id)

    Internally:
      1. Lookup tool by name (O(1) dict access)
      2. Validate args against Pydantic schema
      3. Execute async callable
      4. On LowConfidenceError → auto-route to ML fallback
      5. Wrap result in ToolResult envelope
      6. Emit metrics + audit record
    """

    def __init__(self, rl_loader: RLModelLoader) -> None:
        self._rl_loader = rl_loader
        self._tools: Dict[str, ToolDescriptor] = {}
        self._register_all()

    # ── Registration ──────────────────────────────────────────────────────────

    def _register_all(self) -> None:
        """Register every tool. Add new tools here — nowhere else."""

        self._register(ToolDescriptor(
            name="rl_predict",
            description=(
                "Call the pretrained RL model to get a portfolio action recommendation "
                "and expected annual return. Use this FIRST for any investment or "
                "portfolio allocation question. Falls back to ml_predict automatically."
            ),
            input_schema=RLPredictInput,
            callable_fn=self._execute_rl_predict,
            tags=["rl", "portfolio", "prediction"],
            requires_rl=True,
        ))

        self._register(ToolDescriptor(
            name="sip_calculator",
            description=(
                "Calculate the monthly SIP amount required to achieve a specific financial "
                "goal (retirement, education, home). Accounts for inflation, existing corpus, "
                "step-up SIP, and the user's risk profile."
            ),
            input_schema=SIPCalculatorInput,
            callable_fn=self._execute_sip_calculator,
            tags=["sip", "goal", "investment"],
        ))

        self._register(ToolDescriptor(
            name="fire_planner",
            description=(
                "Build a complete FIRE (Financial Independence, Retire Early) roadmap. "
                "Computes corpus needed, monthly SIP, portfolio allocation, and how long "
                "the corpus will last post-retirement."
            ),
            input_schema=FIREPlannerInput,
            callable_fn=self._execute_fire_planner,
            tags=["fire", "retirement", "planning"],
        ))

        self._register(ToolDescriptor(
            name="tax_wizard",
            description=(
                "Identify every missed tax deduction (80C, 80D, NPS, HRA). "
                "Compare old vs new tax regime with exact numbers. "
                "Rank tax-saving investments by risk profile and liquidity."
            ),
            input_schema=TaxWizardInput,
            callable_fn=self._execute_tax_wizard,
            tags=["tax", "deduction", "80c", "nps"],
        ))

        self._register(ToolDescriptor(
            name="health_score",
            description=(
                "Compute the user's Money Health Score (0–100) across 6 dimensions: "
                "emergency preparedness, insurance, diversification, debt health, "
                "tax efficiency, and retirement readiness."
            ),
            input_schema=HealthScoreInput,
            callable_fn=self._execute_health_score,
            tags=["health", "score", "onboarding"],
        ))

        self._register(ToolDescriptor(
            name="ml_predict",
            description=(
                "ML fallback prediction tool. Called automatically when rl_predict "
                "confidence is below threshold. Do NOT call this directly unless "
                "rl_predict has explicitly failed."
            ),
            input_schema=MLPredictInput,
            callable_fn=self._execute_ml_predict,
            tags=["ml", "fallback", "prediction"],
        ))

    def _register(self, descriptor: ToolDescriptor) -> None:
        self._tools[descriptor.name] = descriptor
        log.debug("Tool registered", tool=descriptor.name, tags=descriptor.tags)

    # ── Public dispatch ───────────────────────────────────────────────────────

    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session_id: str = "",
    ) -> ToolResult:
        """
        Execute a tool by name with validated args.
        Returns a ToolResult regardless of success or failure.
        Never raises — errors are captured in ToolResult.error.
        """
        t0 = time.perf_counter()
        metrics.increment("tool.calls", tags={"tool": tool_name})

        # ── 1. Lookup ─────────────────────────────────────────────────────────
        descriptor = self._tools.get(tool_name)
        if descriptor is None:
            err = f"Unknown tool: '{tool_name}'. Available: {list(self._tools.keys())}"
            log.error(err)
            return ToolResult(
                tool_name=tool_name, success=False, output={}, error=err,
                latency_ms=0.0, session_id=session_id,
            )

        if not descriptor.enabled:
            err = f"Tool '{tool_name}' is currently disabled."
            return ToolResult(
                tool_name=tool_name, success=False, output={}, error=err,
                session_id=session_id,
            )

        # ── 2. Validate inputs ────────────────────────────────────────────────
        try:
            descriptor.input_schema(**args)
        except ValidationError as exc:
            err = f"Input validation failed for '{tool_name}': {exc.errors()}"
            log.warning(err, tool=tool_name)
            metrics.increment("tool.validation_errors", tags={"tool": tool_name})
            return ToolResult(
                tool_name=tool_name, success=False, output={}, error=err,
                session_id=session_id,
            )

        # ── 3. Execute ────────────────────────────────────────────────────────
        try:
            result = await descriptor.callable_fn(**args)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.observe(f"tool.latency_ms.{tool_name}", elapsed_ms)
            metrics.increment("tool.successes", tags={"tool": tool_name})

            # Emit audit record for financial tools
            state = args.get("state")
            if state and isinstance(state, UserFinancialState):
                audit.record(
                    session_id=session_id,
                    action=tool_name,
                    inputs={k: v for k, v in args.items() if k != "state"},
                    outputs=result,
                    user_id=state.user_id,
                )

            log.info(
                "Tool executed",
                tool=tool_name,
                latency_ms=round(elapsed_ms, 2),
                session_id=session_id,
            )

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=result,
                latency_ms=round(elapsed_ms, 2),
                session_id=session_id,
            )

        except LowConfidenceError as exc:
            # Auto-fallback: RL → ML
            log.warning(
                "RL low confidence — auto-routing to ML fallback",
                tool=tool_name,
                confidence=exc.confidence,
            )
            fallback_result = await self.execute(
                "ml_predict",
                {"state": args["state"], "reason": "rl_low_confidence"},
                session_id=session_id,
            )
            fallback_result.used_fallback = True
            fallback_result.tool_name = tool_name   # Report original tool name
            return fallback_result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            err_msg = f"Tool '{tool_name}' raised an unexpected error: {exc}"
            log.error(err_msg, tool=tool_name, error=str(exc))
            metrics.increment("tool.errors", tags={"tool": tool_name})
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output={},
                error=err_msg,
                latency_ms=round(elapsed_ms, 2),
                session_id=session_id,
            )

    async def execute_parallel(
        self,
        calls: List[Dict[str, Any]],
        session_id: str = "",
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls concurrently via asyncio.gather.
        Used by the planner when multiple tools are needed in the same step.

        Args:
            calls: List of {"tool_name": str, "args": dict}
            session_id: Current session identifier

        Returns:
            List of ToolResult in the same order as input calls.
        """
        tasks = [
            self.execute(c["tool_name"], c["args"], session_id)
            for c in calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    # ── Individual async wrappers ─────────────────────────────────────────────

    async def _execute_rl_predict(self, state: UserFinancialState, **_) -> Dict[str, Any]:
        result: PredictionResult = await self._rl_loader.predict_async(
            state, raise_on_low_confidence=True
        )
        return result.to_dict()

    async def _execute_sip_calculator(
        self,
        state: UserFinancialState,
        goal_name: str,
        target_amount: float,
        horizon_years: Optional[float] = None,
        expected_return_pct: Optional[float] = None,
        step_up_pct: float = 10.0,
        **_,
    ) -> Dict[str, Any]:
        return await _tool_sip_calculator(
            state, goal_name, target_amount,
            horizon_years, expected_return_pct, step_up_pct,
        )

    async def _execute_fire_planner(
        self,
        state: UserFinancialState,
        target_retirement_age: Optional[int] = None,
        desired_monthly_expense_at_retirement: Optional[float] = None,
        **_,
    ) -> Dict[str, Any]:
        return await _tool_fire_planner(
            state, target_retirement_age, desired_monthly_expense_at_retirement
        )

    async def _execute_tax_wizard(
        self,
        state: UserFinancialState,
        bonus_amount: float = 0.0,
        compare_regimes: bool = True,
        **_,
    ) -> Dict[str, Any]:
        return await _tool_tax_wizard(state, bonus_amount, compare_regimes)

    async def _execute_health_score(
        self, state: UserFinancialState, **_
    ) -> Dict[str, Any]:
        return await _tool_health_score(state)

    async def _execute_ml_predict(
        self, state: UserFinancialState, reason: str = "direct_call", **_
    ) -> Dict[str, Any]:
        return await _tool_ml_predict(state, reason)

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool schemas — injected into LLM planner's system prompt."""
        return [d.schema_for_llm() for d in self._tools.values() if d.enabled]

    def get_tool_names(self) -> List[str]:
        return [name for name, d in self._tools.items() if d.enabled]


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_registry() -> ToolRegistry:
    """
    Build and return a fully initialised ToolRegistry.
    Called once at application startup.
    """
    rl_loader = build_rl_loader()
    rl_loader.warmup()
    registry = ToolRegistry(rl_loader)
    log.info("ToolRegistry ready", tools=registry.get_tool_names())
    return registry


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

    async def run_tests() -> None:
        print("=== ToolRegistry Self-Test ===\n")

        registry = build_registry()

        print("── Registered Tools ──")
        for t in registry.list_tools():
            print(f"  {t['name']:<20}  tags={t['tags']}")
        print()

        state = UserFinancialState(
            age=32,
            monthly_income=120_000,
            monthly_expense=55_000,
            risk_profile=RiskProfile.MODERATE,
            risk_score=6.0,
            investment_horizon_years=25,
            emergency_fund_amount=180_000,
            existing_monthly_sip=12_000,
            section_80c_used=80_000,
            nps_contribution=3_000,
            insurance=InsuranceCoverage(
                has_term_insurance=True, term_cover_amount=10_000_000,
                has_health_insurance=True, health_cover_amount=500_000,
            ),
            debts=DebtProfile(home_loan_emi=22_000),
            portfolio=InvestmentPortfolio(
                equity_mf=600_000, debt_mf=200_000,
                ppf=100_000, epf=250_000, nps=80_000,
            ),
            goals=[
                FinancialGoal(
                    name="Retirement", target_amount=60_000_000,
                    target_year=2057, priority=1, existing_corpus=1_230_000,
                ),
            ],
        )

        # Test 1: Health Score
        print("── Test 1: health_score ──")
        r1 = await registry.execute("health_score", {"state": state}, "sess_test")
        print(f"  Success : {r1.success}")
        print(f"  Score   : {r1.output.get('overall_score')}/100")
        print(f"  Grade   : {r1.output.get('grade')}")
        print(f"  Actions : {r1.output.get('priority_actions')}\n")

        # Test 2: SIP Calculator
        print("── Test 2: sip_calculator ──")
        r2 = await registry.execute("sip_calculator", {
            "state": state,
            "goal_name": "Retirement",
            "target_amount": 60_000_000,
            "step_up_pct": 10.0,
        }, "sess_test")
        print(f"  Success          : {r2.success}")
        print(f"  Required SIP     : ₹{r2.output.get('required_monthly_sip_inr'):,.0f}")
        print(f"  On track?        : {r2.output.get('goal_achievable_with_current_sip')}")
        print(f"  Recommendation   : {r2.output.get('recommendation')}\n")

        # Test 3: FIRE Planner
        print("── Test 3: fire_planner ──")
        r3 = await registry.execute("fire_planner", {
            "state": state, "target_retirement_age": 50,
        }, "sess_test")
        print(f"  Success          : {r3.success}")
        print(f"  FIRE corpus need : ₹{r3.output.get('fire_corpus_needed_inr'):,.0f}")
        print(f"  Required SIP     : ₹{r3.output.get('required_monthly_sip_inr'):,.0f}")
        print(f"  FIRE feasible?   : {r3.output.get('fire_feasible')}\n")

        # Test 4: Tax Wizard
        print("── Test 4: tax_wizard ──")
        r4 = await registry.execute("tax_wizard", {
            "state": state, "bonus_amount": 200_000,
        }, "sess_test")
        print(f"  Better regime    : {r4.output.get('recommended_regime')}")
        print(f"  Tax saving       : ₹{r4.output.get('tax_saving_by_switching_inr'):,.0f}")
        missed = r4.output.get("missed_deductions", [])
        for m in missed:
            print(f"  Missed {m['section']:<10}: ₹{m['potential_saving_inr']:,.0f} recoverable")
        print()

        # Test 5: RL Predict
        print("── Test 5: rl_predict ──")
        r5 = await registry.execute("rl_predict", {"state": state}, "sess_test")
        print(f"  Success      : {r5.success}")
        print(f"  Used fallback: {r5.used_fallback}")
        print(f"  Action       : {r5.output.get('recommended_action')}")
        print(f"  Confidence   : {r5.output.get('confidence'):.0%}")
        print(f"  Return est   : {r5.output.get('predicted_return_pct')}%\n")

        # Test 6: Parallel execution
        print("── Test 6: execute_parallel ──")
        parallel_results = await registry.execute_parallel([
            {"tool_name": "health_score",    "args": {"state": state}},
            {"tool_name": "tax_wizard",      "args": {"state": state, "bonus_amount": 0}},
        ], session_id="sess_test")
        for r in parallel_results:
            print(f"  {r.tool_name:<20}  success={r.success}  latency={r.latency_ms:.1f}ms")

        print("\n── Metrics Snapshot ──")
        snap = metrics.snapshot()
        for c in snap["counters"]:
            print(f"  {c['name']:<45} {int(c['value'])}")

    asyncio.run(run_tests())
