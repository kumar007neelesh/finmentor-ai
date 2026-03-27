"""
environment/state.py — FinMentor AI User Financial State
=========================================================
Defines the canonical `UserFinancialState` — the single object that:
  1. Holds all user financial data (Pydantic-validated at ingestion)
  2. Serializes to a fixed-length NumPy observation vector for RL/ML models
  3. Tracks state transitions as the user's situation evolves
  4. Computes derived financial ratios used across all tools

This is the "environment" in the RL sense:
  observation = state.to_observation_vector()
  → fed into RL model → action (portfolio allocation) → reward

Usage:
    from environment.state import UserFinancialState, StateTransition, LifeEvent

    state = UserFinancialState(
        user_id="u_001",
        age=30,
        monthly_income=80_000,
        monthly_expense=45_000,
        ...
    )
    obs = state.to_observation_vector()   # np.ndarray shape (20,)
    summary = state.financial_summary()   # human-readable dict
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel,Field, field_validator, model_validator, ValidationInfo
from pydantic import ConfigDict
from config import settings, RiskProfile
from logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Supporting Enums
# ══════════════════════════════════════════════════════════════════════════════

class EmploymentType(str, Enum):
    SALARIED       = "salaried"
    SELF_EMPLOYED  = "self_employed"
    BUSINESS       = "business"
    RETIRED        = "retired"
    STUDENT        = "student"


class CityTier(str, Enum):
    METRO   = "metro"    # Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata
    TIER2   = "tier2"    # Pune, Ahmedabad, Jaipur, etc.
    TIER3   = "tier3"    # Smaller cities


class LifeEvent(str, Enum):
    """Trigger types that cause state transitions."""
    SALARY_HIKE     = "salary_hike"
    MARRIAGE        = "marriage"
    NEW_BABY        = "new_baby"
    HOME_PURCHASE   = "home_purchase"
    JOB_LOSS        = "job_loss"
    INHERITANCE     = "inheritance"
    BONUS           = "bonus"
    MEDICAL_EXPENSE = "medical_expense"
    RETIREMENT      = "retirement"
    NONE            = "none"


class AssetClass(str, Enum):
    EQUITY        = "equity"
    DEBT          = "debt"
    GOLD          = "gold"
    REAL_ESTATE   = "real_estate"
    CASH          = "cash"
    CRYPTO        = "crypto"
    PPF           = "ppf"
    EPF           = "epf"
    NPS           = "nps"


# ══════════════════════════════════════════════════════════════════════════════
# Sub-models
# ══════════════════════════════════════════════════════════════════════════════

class InsuranceCoverage(BaseModel):
    """Insurance details for health score computation."""

    has_term_insurance: bool = False
    term_cover_amount: float = Field(default=0.0, ge=0)        # ₹
    has_health_insurance: bool = False
    health_cover_amount: float = Field(default=0.0, ge=0)      # ₹ per year
    has_critical_illness: bool = False
    has_disability_cover: bool = False

    @property
    def term_cover_adequacy_ratio(self) -> float:
        """Ideal: 10–15× annual income. Returns actual ratio or 0."""
        return 0.0  # Computed externally once we have annual income

    def coverage_score(self, annual_income: float) -> float:
        """
        0–100 insurance adequacy score.
        Weights: term(40) + health(35) + critical(15) + disability(10)
        """
        score = 0.0
        # Term insurance: ideal = 10× annual income
        if self.has_term_insurance and annual_income > 0:
            ratio = self.term_cover_amount / (10 * annual_income)
            score += min(40.0, 40.0 * ratio)
        # Health insurance: ideal ≥ ₹5 lakh per person
        if self.has_health_insurance:
            ratio = min(1.0, self.health_cover_amount / 500_000)
            score += 35.0 * ratio
        if self.has_critical_illness:
            score += 15.0
        if self.has_disability_cover:
            score += 10.0
        return round(score, 2)


class DebtProfile(BaseModel):
    """Outstanding liabilities."""

    home_loan_emi: float = Field(default=0.0, ge=0)
    car_loan_emi: float = Field(default=0.0, ge=0)
    personal_loan_emi: float = Field(default=0.0, ge=0)
    credit_card_outstanding: float = Field(default=0.0, ge=0)
    education_loan_emi: float = Field(default=0.0, ge=0)
    other_emi: float = Field(default=0.0, ge=0)

    @property
    def total_emi(self) -> float:
        return (
            self.home_loan_emi
            + self.car_loan_emi
            + self.personal_loan_emi
            + self.education_loan_emi
            + self.other_emi
        )

    @property
    def total_outstanding_debt(self) -> float:
        """Approximate total outstanding (EMIs are monthly snapshots)."""
        return self.total_emi + self.credit_card_outstanding

    def foir(self, monthly_income: float) -> float:
        """Fixed Obligation to Income Ratio. Ideal < 0.40."""
        if monthly_income <= 0:
            return 1.0
        return round(self.total_emi / monthly_income, 4)


class InvestmentPortfolio(BaseModel):
    """Current investment holdings by asset class (₹ current value)."""

    equity_mf: float = Field(default=0.0, ge=0)      # Equity mutual funds
    debt_mf: float = Field(default=0.0, ge=0)         # Debt mutual funds
    direct_equity: float = Field(default=0.0, ge=0)   # Stocks
    ppf: float = Field(default=0.0, ge=0)
    epf: float = Field(default=0.0, ge=0)
    nps: float = Field(default=0.0, ge=0)
    fd: float = Field(default=0.0, ge=0)              # Fixed deposits
    gold: float = Field(default=0.0, ge=0)
    real_estate: float = Field(default=0.0, ge=0)
    savings_account: float = Field(default=0.0, ge=0)
    crypto: float = Field(default=0.0, ge=0)
    other: float = Field(default=0.0, ge=0)

    @property
    def total_value(self) -> float:
        return sum(self.dict().values())

    @property
    def equity_pct(self) -> float:
        total = self.total_value
        if total == 0:
            return 0.0
        equity = self.equity_mf + self.direct_equity
        return round(equity / total * 100, 2)

    @property
    def debt_pct(self) -> float:
        total = self.total_value
        if total == 0:
            return 0.0
        debt = self.debt_mf + self.ppf + self.epf + self.nps + self.fd
        return round(debt / total * 100, 2)

    def allocation_breakdown(self) -> Dict[str, float]:
        total = self.total_value or 1.0
        return {
            AssetClass.EQUITY:      round((self.equity_mf + self.direct_equity) / total * 100, 1),
            AssetClass.DEBT:        round((self.debt_mf + self.ppf + self.epf + self.fd) / total * 100, 1),
            AssetClass.NPS:         round(self.nps / total * 100, 1),
            AssetClass.GOLD:        round(self.gold / total * 100, 1),
            AssetClass.REAL_ESTATE: round(self.real_estate / total * 100, 1),
            AssetClass.CASH:        round(self.savings_account / total * 100, 1),
            AssetClass.CRYPTO:      round(self.crypto / total * 100, 1),
        }


class FinancialGoal(BaseModel):
    """A single user financial goal."""

    goal_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str                                    # "Child education", "Retirement", etc.
    target_amount: float = Field(ge=0)           # ₹ in today's value
    target_year: int                             # Calendar year
    priority: int = Field(ge=1, le=5)           # 1=highest, 5=lowest
    existing_corpus: float = Field(default=0.0, ge=0)
    monthly_sip_allocated: float = Field(default=0.0, ge=0)

    @property
    def years_remaining(self) -> float:
        today = date.today()
        return max(0.0, self.target_year - today.year)

    @property
    def shortfall(self) -> float:
        return max(0.0, self.target_amount - self.existing_corpus)


# ══════════════════════════════════════════════════════════════════════════════
# Core State Model
# ══════════════════════════════════════════════════════════════════════════════

# Observation vector feature names — ORDER IS FIXED (RL model depends on this)
OBSERVATION_FEATURES: Tuple[str, ...] = (
    "age_normalized",               # 0  age / 60
    "income_log",                   # 1  log10(monthly_income / 10000)
    "savings_rate",                 # 2  (income - expense) / income
    "foir",                         # 3  total_emi / income
    "emergency_fund_ratio",         # 4  emergency_fund / (6 * monthly_expense)
    "investment_to_income_ratio",   # 5  total_portfolio / (12 * income)
    "equity_allocation_pct",        # 6  equity % of portfolio (0–1)
    "debt_allocation_pct",          # 7  debt % of portfolio (0–1)
    "risk_score_normalized",        # 8  risk_score / 10
    "insurance_score_normalized",   # 9  insurance_score / 100
    "has_term_insurance",           # 10 binary
    "has_health_insurance",         # 11 binary
    "horizon_normalized",           # 12 investment_horizon / 40
    "num_goals_normalized",         # 13 len(goals) / 10
    "goal_shortfall_ratio",         # 14 total_shortfall / (12 * annual_income)
    "tax_bracket_normalized",       # 15 effective_tax_rate / 0.30
    "existing_sip_to_income",       # 16 monthly_sip / monthly_income
    "debt_to_income_annual",        # 17 total_debt / annual_income
    "city_tier_encoded",            # 18 metro=1.0, tier2=0.5, tier3=0.0
    "employment_encoded",           # 19 salaried=1.0, self_employed=0.7, other=0.4
)

OBS_DIM = len(OBSERVATION_FEATURES)  # == 20, must match config.rl_model.observation_dim


class UserFinancialState(BaseModel):
    """
    Canonical user financial state.

    This is the single source of truth flowing through the entire system:
      - Planner reads it for context
      - Tools compute recommendations from it
      - RL model receives it as a fixed-length observation vector
      - Memory stores snapshots of it over time
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    user_id: str = Field(default_factory=lambda: f"u_{uuid4().hex[:8]}")
    session_id: str = Field(default_factory=lambda: f"sess_{uuid4().hex[:8]}")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # ── Demographics ──────────────────────────────────────────────────────────
    age: int = Field(ge=18, le=80, description="User's current age")
    retirement_age: int = Field(default=60, ge=40, le=75)
    employment_type: EmploymentType = EmploymentType.SALARIED
    city_tier: CityTier = CityTier.METRO
    dependents: int = Field(default=0, ge=0, le=10)
    is_married: bool = False

    # ── Income & Expenses (monthly, ₹) ───────────────────────────────────────
    monthly_income: float = Field(gt=0, description="Gross monthly income in ₹")
    monthly_expense: float = Field(gt=0, description="Monthly living expenses in ₹")
    monthly_emi: float = Field(default=0.0, ge=0, description="Total monthly EMIs (overrides debt_profile sum if set)")

    # ── Tax ───────────────────────────────────────────────────────────────────
    annual_income: float = Field(default=0.0, ge=0)
    tax_regime: str = Field(default="new", pattern="^(old|new)$")
    section_80c_used: float = Field(default=0.0, ge=0)
    section_80d_used: float = Field(default=0.0, ge=0)
    nps_contribution: float = Field(default=0.0, ge=0)   # monthly ₹ to NPS

    # ── Risk ──────────────────────────────────────────────────────────────────
    risk_profile: RiskProfile = RiskProfile.MODERATE
    risk_score: float = Field(
        default=5.0, ge=1.0, le=10.0,
        description="1=very conservative, 10=very aggressive"
    )
    investment_horizon_years: int = Field(
        default=20, ge=1, le=40,
        description="Primary investment horizon (years)"
    )

    # ── Emergency Fund ────────────────────────────────────────────────────────
    emergency_fund_amount: float = Field(default=0.0, ge=0)  # ₹ in liquid form

    # ── Existing SIP ─────────────────────────────────────────────────────────
    existing_monthly_sip: float = Field(default=0.0, ge=0)   # ₹/month total SIP

    # ── Sub-models ────────────────────────────────────────────────────────────
    insurance: InsuranceCoverage = Field(default_factory=InsuranceCoverage)
    debts: DebtProfile = Field(default_factory=DebtProfile)
    portfolio: InvestmentPortfolio = Field(default_factory=InvestmentPortfolio)
    goals: List[FinancialGoal] = Field(default_factory=list)

    # ── Last life event ───────────────────────────────────────────────────────
    last_life_event: LifeEvent = LifeEvent.NONE
    last_life_event_amount: float = Field(default=0.0, ge=0)  # ₹ amount (bonus, inheritance, etc.)

    # ── State hash (for change detection) ────────────────────────────────────
    _state_hash: str = ""

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("annual_income", mode="before")
    @classmethod
    def compute_annual_income(cls, v: float, info: ValidationInfo) -> float:
        """Auto-derive annual_income from monthly if not explicitly set."""
        if (v is None or v == 0.0) and info.data.get("monthly_income"):
            return info.data["monthly_income"] * 12
        return v or 0.0

    @field_validator("monthly_expense", mode="after")
    @classmethod
    def expense_lt_income(cls, v: float, info: ValidationInfo) -> float:
        income = info.data.get("monthly_income", 0)
        if income > 0 and v >= income:
            log.warning(
                "Monthly expense >= monthly income — user may be in deficit",
                expense=v,
                income=income,
            )
        return v
    
    @field_validator("risk_score", mode="after")
    @classmethod
    def sync_risk_profile(cls, v: float, info: ValidationInfo) -> float:
        profile = info.data.get("risk_profile", RiskProfile.MODERATE)
        expected = {
            RiskProfile.CONSERVATIVE: (1.0, 4.0),
            RiskProfile.MODERATE:     (4.1, 7.0),
            RiskProfile.AGGRESSIVE:   (7.1, 10.0),
        }
        lo, hi = expected[profile]
        if not (lo <= v <= hi):
            log.warning(
                "risk_score out of range for risk_profile — clamping",
                risk_score=v,
                risk_profile=profile,
                expected_range=(lo, hi),
            )
            return max(lo, min(hi, v))
        return v

    @model_validator(mode="after")
    def validate_retirement_age(self) -> "UserFinancialState":
        if self.retirement_age <= self.age:
            raise ValueError(
                f"retirement_age ({self.retirement_age}) must be greater than current age ({self.age})."
            )
        return self

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def monthly_savings(self) -> float:
        """Net monthly surplus after expenses and EMIs."""
        emi = self.monthly_emi if self.monthly_emi > 0 else self.debts.total_emi
        return max(0.0, self.monthly_income - self.monthly_expense - emi)

    @property
    def savings_rate(self) -> float:
        """Savings as fraction of income. Ideal ≥ 0.20."""
        if self.monthly_income <= 0:
            return 0.0
        return round(self.monthly_savings / self.monthly_income, 4)

    @property
    def emergency_fund_months(self) -> float:
        """How many months of expenses the emergency fund covers."""
        if self.monthly_expense <= 0:
            return 0.0
        return round(self.emergency_fund_amount / self.monthly_expense, 2)

    @property
    def effective_tax_rate(self) -> float:
        """Approximate effective tax rate based on annual income and regime."""
        income = self.annual_income
        slabs = settings.finance.tax_slabs_new_regime
        tax = 0.0
        remaining = income

        bracket_map = [
            (300_000, 0.00),
            (400_000, 0.05),
            (300_000, 0.10),
            (200_000, 0.15),
            (300_000, 0.20),
            (float("inf"), 0.30),
        ]
        for bracket_size, rate in bracket_map:
            taxable = min(remaining, bracket_size)
            tax += taxable * rate
            remaining -= taxable
            if remaining <= 0:
                break

        # Section 87A rebate: no tax if income ≤ ₹7 lakh (new regime)
        if income <= 700_000 and self.tax_regime == "new":
            tax = 0.0

        return round(tax / income, 4) if income > 0 else 0.0

    @property
    def foir(self) -> float:
        """Fixed Obligation to Income Ratio."""
        emi = self.monthly_emi if self.monthly_emi > 0 else self.debts.total_emi
        return round(emi / self.monthly_income, 4) if self.monthly_income > 0 else 0.0

    @property
    def total_goal_shortfall(self) -> float:
        return sum(g.shortfall for g in self.goals)

    @property
    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.age)

    # ── Observation vector ────────────────────────────────────────────────────

    def to_observation_vector(self) -> np.ndarray:
        """
        Serialize state to a fixed-length float32 NumPy array for RL/ML models.
        Shape: (OBS_DIM,) == (20,)

        All features are normalized to [0, 1] range.
        Missing/zero values are represented as 0.0 (not NaN).
        """
        annual_income = self.annual_income or (self.monthly_income * 12)
        insurance_score = self.insurance.coverage_score(annual_income)

        obs = np.array(
            [
                # 0: age normalized
                np.clip(self.age / 60.0, 0.0, 1.0),
                # 1: income (log-scaled, anchored at ₹10k/month)
                np.clip(np.log10(max(self.monthly_income, 1) / 10_000) / 2.0 + 0.5, 0.0, 1.0),
                # 2: savings rate
                np.clip(self.savings_rate, 0.0, 1.0),
                # 3: FOIR (inverted: lower is better)
                np.clip(self.foir, 0.0, 1.0),
                # 4: emergency fund ratio (target = 6 months)
                np.clip(self.emergency_fund_months / 6.0, 0.0, 1.0),
                # 5: portfolio to annual income ratio (capped at 10×)
                np.clip(self.portfolio.total_value / max(annual_income, 1) / 10.0, 0.0, 1.0),
                # 6: equity allocation %
                np.clip(self.portfolio.equity_pct / 100.0, 0.0, 1.0),
                # 7: debt allocation %
                np.clip(self.portfolio.debt_pct / 100.0, 0.0, 1.0),
                # 8: risk score (1–10 → 0–1)
                np.clip((self.risk_score - 1.0) / 9.0, 0.0, 1.0),
                # 9: insurance score (0–100 → 0–1)
                np.clip(insurance_score / 100.0, 0.0, 1.0),
                # 10: has term insurance (binary)
                float(self.insurance.has_term_insurance),
                # 11: has health insurance (binary)
                float(self.insurance.has_health_insurance),
                # 12: investment horizon (years, normalized to 40)
                np.clip(self.investment_horizon_years / 40.0, 0.0, 1.0),
                # 13: number of goals (capped at 10)
                np.clip(len(self.goals) / 10.0, 0.0, 1.0),
                # 14: goal shortfall to annual income ratio (capped at 20×)
                np.clip(self.total_goal_shortfall / max(annual_income, 1) / 20.0, 0.0, 1.0),
                # 15: effective tax rate (max 30%)
                np.clip(self.effective_tax_rate / 0.30, 0.0, 1.0),
                # 16: existing SIP to income ratio
                np.clip(self.existing_monthly_sip / max(self.monthly_income, 1), 0.0, 1.0),
                # 17: total outstanding debt to annual income
                np.clip(self.debts.total_outstanding_debt / max(annual_income, 1), 0.0, 1.0),
                # 18: city tier (metro=1.0, tier2=0.5, tier3=0.0)
                {"metro": 1.0, "tier2": 0.5, "tier3": 0.0}[self.city_tier.value],
                # 19: employment type
                {"salaried": 1.0, "self_employed": 0.7, "business": 0.6,
                 "retired": 0.3, "student": 0.1}[self.employment_type.value],
            ],
            dtype=np.float32,
        )

        assert obs.shape == (OBS_DIM,), f"Observation shape mismatch: {obs.shape}"
        assert not np.any(np.isnan(obs)), "NaN detected in observation vector"

        log.debug(
            "Observation vector built",
            shape=obs.shape,
            user_id=self.user_id,
            savings_rate=float(obs[2]),
            foir=float(obs[3]),
        )
        return obs

    # ── Fingerprint ───────────────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """SHA-256 hash of the state for change detection and memory dedup."""
        snapshot = {
            "age": self.age,
            "monthly_income": self.monthly_income,
            "monthly_expense": self.monthly_expense,
            "emergency_fund": self.emergency_fund_amount,
            "portfolio_total": self.portfolio.total_value,
            "existing_sip": self.existing_monthly_sip,
            "risk_score": self.risk_score,
            "goals_count": len(self.goals),
        }
        raw = json.dumps(snapshot, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Human-readable summary ────────────────────────────────────────────────

    def financial_summary(self) -> Dict[str, Any]:
        """
        Dict consumed by the LLM planner as context.
        Deliberately concise — avoids token bloat.
        """
        annual_income = self.annual_income or self.monthly_income * 12
        return {
            "profile": {
                "age": self.age,
                "employment": self.employment_type.value,
                "city": self.city_tier.value,
                "is_married": self.is_married,
                "dependents": self.dependents,
                "retirement_age": self.retirement_age,
                "years_to_retirement": self.years_to_retirement,
            },
            "income_and_savings": {
                "monthly_income_inr": self.monthly_income,
                "monthly_expense_inr": self.monthly_expense,
                "monthly_savings_inr": round(self.monthly_savings, 2),
                "savings_rate_pct": round(self.savings_rate * 100, 1),
                "annual_income_inr": annual_income,
                "effective_tax_rate_pct": round(self.effective_tax_rate * 100, 1),
                "tax_regime": self.tax_regime,
            },
            "debt": {
                "total_monthly_emi_inr": round(self.debts.total_emi, 2),
                "foir": round(self.foir, 3),
                "foir_status": "healthy" if self.foir < 0.40 else "high",
                "credit_card_outstanding_inr": self.debts.credit_card_outstanding,
            },
            "emergency_fund": {
                "amount_inr": self.emergency_fund_amount,
                "months_covered": self.emergency_fund_months,
                "status": (
                    "adequate" if self.emergency_fund_months >= 6
                    else "low" if self.emergency_fund_months >= 3
                    else "critical"
                ),
            },
            "insurance": {
                "has_term": self.insurance.has_term_insurance,
                "term_cover_inr": self.insurance.term_cover_amount,
                "has_health": self.insurance.has_health_insurance,
                "health_cover_inr": self.insurance.health_cover_amount,
                "coverage_score": self.insurance.coverage_score(annual_income),
            },
            "portfolio": {
                "total_value_inr": self.portfolio.total_value,
                "equity_pct": self.portfolio.equity_pct,
                "debt_pct": self.portfolio.debt_pct,
                "allocation_breakdown": self.portfolio.allocation_breakdown(),
                "existing_monthly_sip_inr": self.existing_monthly_sip,
            },
            "goals": [
                {
                    "name": g.name,
                    "target_inr": g.target_amount,
                    "target_year": g.target_year,
                    "years_remaining": g.years_remaining,
                    "existing_corpus_inr": g.existing_corpus,
                    "shortfall_inr": round(g.shortfall, 2),
                    "priority": g.priority,
                }
                for g in sorted(self.goals, key=lambda g: g.priority)
            ],
            "risk": {
                "profile": self.risk_profile.value,
                "score": self.risk_score,
                "investment_horizon_years": self.investment_horizon_years,
            },
            "last_life_event": {
                "event": self.last_life_event.value,
                "amount_inr": self.last_life_event_amount,
            },
        }

    class Config:
        use_enum_values = False
        validate_assignment = True
        json_encoders = {datetime: lambda v: v.isoformat(), np.ndarray: lambda v: v.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# State Transition
# ══════════════════════════════════════════════════════════════════════════════

class StateTransition(BaseModel):
    """
    Records a before/after state change triggered by a life event or agent action.
    Stored in memory for the Evaluator to measure prediction accuracy.
    """

    transition_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    user_id: str
    session_id: str
    life_event: LifeEvent
    event_amount: float = 0.0

    # Serialized state snapshots (use .dict() to avoid circular ref)
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]

    # Observation vectors as lists (JSON-serializable)
    obs_before: List[float]
    obs_after: List[float]

    # Agent action taken (e.g. "increase_sip", "rebalance_equity")
    agent_action: str = ""
    predicted_reward: float = 0.0
    actual_reward: float = 0.0

    @classmethod
    def create(
        cls,
        state_before: UserFinancialState,
        state_after: UserFinancialState,
        life_event: LifeEvent,
        event_amount: float = 0.0,
        agent_action: str = "",
        predicted_reward: float = 0.0,
    ) -> "StateTransition":
        return cls(
            user_id=state_before.user_id,
            session_id=state_before.session_id,
            life_event=life_event,
            event_amount=event_amount,
            state_before=state_before.financial_summary(),
            state_after=state_after.financial_summary(),
            obs_before=state_before.to_observation_vector().tolist(),
            obs_after=state_after.to_observation_vector().tolist(),
            agent_action=agent_action,
            predicted_reward=predicted_reward,
        )

    def reward_error(self) -> float:
        """Absolute prediction error."""
        return abs(self.predicted_reward - self.actual_reward)


# ══════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def state_from_dict(data: Dict[str, Any]) -> UserFinancialState:
    """Construct a UserFinancialState from a raw dict (e.g. from API payload)."""
    return UserFinancialState(**data)


def minimal_state(
    age: int,
    monthly_income: float,
    monthly_expense: float,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
) -> UserFinancialState:
    """
    Quick factory for tests and demos — creates a valid state with minimal fields.
    """
    return UserFinancialState(
        age=age,
        monthly_income=monthly_income,
        monthly_expense=monthly_expense,
        risk_profile=risk_profile,
        risk_score={
            RiskProfile.CONSERVATIVE: 3.0,
            RiskProfile.MODERATE:     5.5,
            RiskProfile.AGGRESSIVE:   8.5,
        }[risk_profile],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json

    print("=== UserFinancialState Self-Test ===\n")

    state = UserFinancialState(
        age=32,
        monthly_income=120_000,
        monthly_expense=55_000,
        risk_profile=RiskProfile.MODERATE,
        risk_score=6.0,
        investment_horizon_years=25,
        emergency_fund_amount=200_000,
        existing_monthly_sip=15_000,
        employment_type=EmploymentType.SALARIED,
        city_tier=CityTier.METRO,
        is_married=True,
        dependents=1,
        section_80c_used=100_000,
        nps_contribution=5_000,
        insurance=InsuranceCoverage(
            has_term_insurance=True,
            term_cover_amount=10_000_000,
            has_health_insurance=True,
            health_cover_amount=500_000,
        ),
        debts=DebtProfile(
            home_loan_emi=25_000,
            car_loan_emi=8_000,
        ),
        portfolio=InvestmentPortfolio(
            equity_mf=800_000,
            debt_mf=200_000,
            ppf=150_000,
            epf=300_000,
            nps=100_000,
            fd=200_000,
            gold=100_000,
        ),
        goals=[
            FinancialGoal(
                name="Child Higher Education",
                target_amount=3_000_000,
                target_year=2042,
                priority=1,
                existing_corpus=200_000,
            ),
            FinancialGoal(
                name="Retirement Corpus",
                target_amount=60_000_000,
                target_year=2055,
                priority=2,
                existing_corpus=1_550_000,
            ),
        ],
    )

    print("── Derived Properties ──")
    print(f"  Monthly savings     : ₹{state.monthly_savings:,.0f}")
    print(f"  Savings rate        : {state.savings_rate*100:.1f}%")
    print(f"  Emergency fund      : {state.emergency_fund_months:.1f} months")
    print(f"  FOIR                : {state.foir:.3f} ({'healthy' if state.foir < 0.4 else 'high'})")
    print(f"  Effective tax rate  : {state.effective_tax_rate*100:.1f}%")
    print(f"  Portfolio total     : ₹{state.portfolio.total_value:,.0f}")
    print(f"  Equity allocation   : {state.portfolio.equity_pct:.1f}%")
    print(f"  Total goal shortfall: ₹{state.total_goal_shortfall:,.0f}")
    print(f"  State fingerprint   : {state.fingerprint()}")
    print(f"  Years to retirement : {state.years_to_retirement}")

    print("\n── Observation Vector ──")
    obs = state.to_observation_vector()
    print(f"  Shape: {obs.shape}   dtype: {obs.dtype}")
    for i, (name, val) in enumerate(zip(OBSERVATION_FEATURES, obs)):
        print(f"  [{i:02d}] {name:<35} {val:.4f}")

    print("\n── Financial Summary (for LLM planner) ──")
    summary = state.financial_summary()
    print(_json.dumps(summary, indent=2, default=str))

    print("\n── StateTransition ──")
    state2 = state.model_copy(
        update={"monthly_income": 140_000, "last_life_event": LifeEvent.SALARY_HIKE}
    )
    transition = StateTransition.create(
        state_before=state,
        state_after=state2,
        life_event=LifeEvent.SALARY_HIKE,
        event_amount=20_000,
        agent_action="increase_sip",
        predicted_reward=0.72,
    )
    print(f"  Transition ID  : {transition.transition_id}")
    print(f"  Life event     : {transition.life_event}")
    print(f"  obs_before[1]  : {transition.obs_before[1]:.4f}  (income)")
    print(f"  obs_after[1]   : {transition.obs_after[1]:.4f}  (income after hike)")
