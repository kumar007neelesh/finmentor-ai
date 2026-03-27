"""
main.py — FinMentor AI Application Entry Point
===============================================
FastAPI application that exposes the entire FinMentor AI agent stack
over HTTP. This is the deployment unit — what runs inside Docker.

Endpoints:
  POST /sessions                → Create a new user session
  POST /sessions/{id}/chat      → Send a message, get a financial plan
  POST /sessions/{id}/onboard   → Run the 3-step onboarding sequence
  POST /sessions/{id}/life-event → Apply a life event + get advice
  GET  /sessions/{id}           → Get session summary
  DELETE /sessions/{id}         → End and clean up a session
  GET  /health                  → Liveness probe (k8s / ECS)
  GET  /ready                   → Readiness probe (is agent stack warm?)
  GET  /metrics                 → In-process metrics snapshot (Prometheus-ready)
  POST /evaluate/{user_id}      → Trigger batch evaluation for a user
  GET  /evaluate/report/{id}    → Fetch a stored evaluation report

Startup sequence (lifespan):
  1. Validate config (API key present in production)
  2. Build ToolRegistry → warms up RL model
  3. Build LLMWrapper   → constructs system prompt with tool schemas
  4. Build Planner      → wires classifier
  5. Build Executor     → ready to handle requests
  6. Build Evaluator    → ready for background scoring
  7. Log "System ready"

CLI usage (development):
  python main.py                 # Start API server (uvicorn)
  python main.py --demo          # Run interactive demo in terminal
  python main.py --evaluate      # Run system evaluation and print report
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator, field_validator

from config import settings, RiskProfile
from agent.evaluator import build_evaluator, EvaluationReport, synthetic_return_provider
from agent.executor import Executor, LifeEventInput, build_executor
from agent.memory import get_memory_store, reset_memory_store
from agent.planner import FinancialPlan
from environment.state import (
    UserFinancialState,
    InsuranceCoverage,
    DebtProfile,
    InvestmentPortfolio,
    FinancialGoal,
    LifeEvent,
    EmploymentType,
    CityTier,
    minimal_state,
)
from logger import get_logger, metrics, set_context, clear_context

log = get_logger(__name__)

# ── Global singletons (initialised during lifespan) ───────────────────────────
_executor:  Optional[Executor]        = None
_evaluator: Optional[Any]             = None   # Evaluator type


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — startup & shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Startup: build and warm up the full agent stack.
    Shutdown: clean up resources.
    """
    global _executor, _evaluator

    log.info("=" * 60)
    log.info("FinMentor AI — Starting up")
    log.info(f"  Environment : {settings.environment}")
    log.info(f"  LLM Model   : {settings.llm.model}")
    log.info(f"  RL Backend  : {settings.rl_model.backend}")
    log.info(f"  Memory      : {settings.agent.memory_backend}")
    log.info("=" * 60)

    # Validate API key presence
    if settings.is_production and not settings.anthropic_api_key:
        log.error("ANTHROPIC_API_KEY is not set in production. Aborting.")
        sys.exit(1)

    try:
        # Build full stack (each step logs its own startup message)
        _executor  = build_executor()
        _evaluator = build_evaluator(
            memory=get_memory_store(),
            return_provider=synthetic_return_provider,   # Swap for prod API
        )
        metrics.increment("system.startups")
        log.info("✅ FinMentor AI is ready to serve requests")

    except Exception as exc:
        log.error(f"Startup failed: {exc}")
        if settings.is_production:
            sys.exit(1)

    yield   # ← Application serves requests here

    # Shutdown
    log.info("FinMentor AI — Shutting down")
    metrics.increment("system.shutdowns")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="FinMentor AI",
    description=(
        "AI-powered personal finance mentor for India. "
        "Provides SIP planning, FIRE roadmaps, tax optimisation, "
        "health scoring, and life event advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Serve static files (frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
# ── Request ID middleware ──────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject a unique request_id into every request for tracing."""
    request_id = request.headers.get("X-Request-ID", f"req_{uuid4().hex[:8]}")
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Dependency: require executor ──────────────────────────────────────────────

def get_executor() -> Executor:
    if _executor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialised. Try again in a moment.",
        )
    return _executor


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class InsuranceRequest(BaseModel):
    has_term_insurance: bool = False
    term_cover_amount: float = 0.0
    has_health_insurance: bool = False
    health_cover_amount: float = 0.0
    has_critical_illness: bool = False


class DebtRequest(BaseModel):
    home_loan_emi: float = 0.0
    car_loan_emi: float = 0.0
    personal_loan_emi: float = 0.0
    credit_card_outstanding: float = 0.0
    education_loan_emi: float = 0.0


class PortfolioRequest(BaseModel):
    equity_mf: float = 0.0
    debt_mf: float = 0.0
    direct_equity: float = 0.0
    ppf: float = 0.0
    epf: float = 0.0
    nps: float = 0.0
    fd: float = 0.0
    gold: float = 0.0
    real_estate: float = 0.0
    savings_account: float = 0.0


class GoalRequest(BaseModel):
    name: str
    target_amount: float = Field(gt=0)
    target_year: int = Field(gt=2024)
    priority: int = Field(ge=1, le=5, default=1)
    existing_corpus: float = 0.0


class CreateSessionRequest(BaseModel):
    """Full user profile to create a session with."""
    user_id: Optional[str] = None
    age: int = Field(ge=18, le=80)
    monthly_income: float = Field(gt=0)
    monthly_expense: float = Field(gt=0)
    risk_profile: str = Field(default="moderate", pattern="^(conservative|moderate|aggressive)$")
    risk_score: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    investment_horizon_years: int = Field(default=20, ge=1, le=40)
    retirement_age: int = Field(default=60, ge=40, le=75)
    employment_type: str = Field(default="salaried")
    city_tier: str = Field(default="metro")
    is_married: bool = False
    dependents: int = Field(default=0, ge=0)
    emergency_fund_amount: float = Field(default=0.0, ge=0)
    existing_monthly_sip: float = Field(default=0.0, ge=0)
    section_80c_used: float = Field(default=0.0, ge=0)
    nps_contribution: float = Field(default=0.0, ge=0)
    tax_regime: str = Field(default="new", pattern="^(old|new)$")
    insurance: InsuranceRequest = Field(default_factory=InsuranceRequest)
    debts: DebtRequest = Field(default_factory=DebtRequest)
    portfolio: PortfolioRequest = Field(default_factory=PortfolioRequest)
    goals: List[GoalRequest] = Field(default_factory=list)

    def to_state(self) -> UserFinancialState:
        """Convert API request to canonical UserFinancialState."""
        rp_map = {
            "conservative": RiskProfile.CONSERVATIVE,
            "moderate":     RiskProfile.MODERATE,
            "aggressive":   RiskProfile.AGGRESSIVE,
        }
        risk_profile = rp_map.get(self.risk_profile, RiskProfile.MODERATE)
        score_defaults = {
            RiskProfile.CONSERVATIVE: 3.0,
            RiskProfile.MODERATE:     5.5,
            RiskProfile.AGGRESSIVE:   8.5,
        }

        employment_map = {e.value: e for e in EmploymentType}
        city_map       = {c.value: c for c in CityTier}

        return UserFinancialState(
            user_id=self.user_id or f"u_{uuid4().hex[:8]}",
            age=self.age,
            monthly_income=self.monthly_income,
            monthly_expense=self.monthly_expense,
            risk_profile=risk_profile,
            risk_score=self.risk_score or score_defaults[risk_profile],
            investment_horizon_years=self.investment_horizon_years,
            retirement_age=self.retirement_age,
            employment_type=employment_map.get(self.employment_type, EmploymentType.SALARIED),
            city_tier=city_map.get(self.city_tier, CityTier.METRO),
            is_married=self.is_married,
            dependents=self.dependents,
            emergency_fund_amount=self.emergency_fund_amount,
            existing_monthly_sip=self.existing_monthly_sip,
            section_80c_used=self.section_80c_used,
            nps_contribution=self.nps_contribution,
            tax_regime=self.tax_regime,
            insurance=InsuranceCoverage(
                has_term_insurance=self.insurance.has_term_insurance,
                term_cover_amount=self.insurance.term_cover_amount,
                has_health_insurance=self.insurance.has_health_insurance,
                health_cover_amount=self.insurance.health_cover_amount,
                has_critical_illness=self.insurance.has_critical_illness,
            ),
            debts=DebtProfile(
                home_loan_emi=self.debts.home_loan_emi,
                car_loan_emi=self.debts.car_loan_emi,
                personal_loan_emi=self.debts.personal_loan_emi,
                credit_card_outstanding=self.debts.credit_card_outstanding,
                education_loan_emi=self.debts.education_loan_emi,
            ),
            portfolio=InvestmentPortfolio(
                equity_mf=self.portfolio.equity_mf,
                debt_mf=self.portfolio.debt_mf,
                direct_equity=self.portfolio.direct_equity,
                ppf=self.portfolio.ppf,
                epf=self.portfolio.epf,
                nps=self.portfolio.nps,
                fd=self.portfolio.fd,
                gold=self.portfolio.gold,
                real_estate=self.portfolio.real_estate,
                savings_account=self.portfolio.savings_account,
            ),
            goals=[
                FinancialGoal(
                    name=g.name,
                    target_amount=g.target_amount,
                    target_year=g.target_year,
                    priority=g.priority,
                    existing_corpus=g.existing_corpus,
                )
                for g in self.goals
            ],
        )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    stream: bool = Field(default=False, description="Stream the response token by token")


class LifeEventRequest(BaseModel):
    event: str = Field(..., description="Life event type")
    amount: float = Field(default=0.0, ge=0, description="₹ amount associated with event")
    description: str = Field(default="")
    message: Optional[str] = Field(default=None, description="Optional custom query")

    @field_validator("event")
    def valid_event(cls, v: str) -> str:
        valid = {e.value for e in LifeEvent}
        if v not in valid:
            raise ValueError(f"Invalid event. Choose from: {sorted(valid)}")
        return v


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    message: str
    created_at: str


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    answer: str
    intent: str
    tools_used: List[str]
    key_numbers: Dict[str, Any]
    latency_ms: float
    plan_id: str


class OnboardingResponse(BaseModel):
    session_id: str
    steps: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

# ── Health & readiness ────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness probe — always returns 200 if process is running."""
    return {
        "status": "ok",
        "service": "finmentor-ai",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["System"])
async def ready():
    """
    Readiness probe — returns 200 only when agent stack is fully initialised.
    Used by Kubernetes/ECS to gate traffic.
    """
    if _executor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent stack not yet initialised.",
        )
    return {
        "status": "ready",
        "model": settings.llm.model,
        "rl_backend": settings.rl_model.backend.value,
        "memory_backend": settings.agent.memory_backend,
        "active_sessions": len(_executor.active_sessions()),
    }


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """
    In-process metrics snapshot.
    Format compatible with Prometheus custom exporter or Datadog API.
    """
    return metrics.snapshot()


# ── Session management ────────────────────────────────────────────────────────

@app.post("/sessions", response_model=SessionResponse, tags=["Sessions"])
async def create_session(body: CreateSessionRequest):
    """
    Create a new financial planning session.
    Accepts the user's full financial profile and returns a session_id
    that must be included in all subsequent requests.

    This is the entry point for:
      - New user onboarding
      - Returning users with updated financial data
    """
    executor = get_executor()

    try:
        state = body.to_state()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid financial state: {exc}",
        )

    session_id = await executor.start_session(state)
    metrics.increment("api.sessions_created")

    log.info(
        "Session created via API",
        session_id=session_id,
        user_id=state.user_id,
        age=state.age,
        income=state.monthly_income,
    )

    return SessionResponse(
        session_id=session_id,
        user_id=state.user_id,
        message="Session created. Send your financial question to /sessions/{id}/chat",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Get the current session summary including state and last plan."""
    executor = get_executor()
    summary = executor.session_summary(session_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return summary


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def end_session(session_id: str):
    """End and clean up a session."""
    executor = get_executor()
    if executor.get_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    await executor.end_session(session_id)
    metrics.increment("api.sessions_ended")
    return {"status": "ended", "session_id": session_id}


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/chat", tags=["Chat"])
async def chat(session_id: str, body: ChatRequest, request: Request):
    """
    Send a financial query and receive an AI-generated plan.

    Supports streaming (set stream=true) for real-time token output.
    Without streaming, waits for the full response (typically 2–8 seconds).

    The agent will:
      1. Classify your intent (SIP / FIRE / Tax / Health / Portfolio)
      2. Call the appropriate financial tools
      3. Return a personalised, actionable recommendation
    """
    executor = get_executor()
    session  = executor.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Create one via POST /sessions.",
        )

    set_context(session_id=session_id, user_id=session.user_id)
    metrics.increment("api.chat_requests")

    try:
        if body.stream:
            # Streaming response — pipe LLM tokens directly to client
            return StreamingResponse(
                _stream_chat(executor, session_id, session.user_id, body.message),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Standard non-streaming response
        result = await executor.run_turn(
            session_id=session_id,
            user_id=session.user_id,
            query=body.message,
        )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error or "Planning failed.",
            )

        plan = result.plan
        return ChatResponse(
            session_id=session_id,
            turn=result.turn_number,
            answer=plan.final_answer,
            intent=plan.intent,
            tools_used=plan.tools_used,
            key_numbers={
                "health_score":          plan.health_score,
                "recommended_sip_inr":   plan.recommended_sip_inr,
                "predicted_return_pct":  plan.predicted_return_pct,
                "fire_corpus_inr":       plan.fire_corpus_inr,
                "tax_saving_inr":        plan.tax_saving_inr,
            },
            latency_ms=result.total_latency_ms,
            plan_id=plan.plan_id,
        )

    finally:
        clear_context()


async def _stream_chat(
    executor: Executor,
    session_id: str,
    user_id: str,
    message: str,
):
    """
    AsyncGenerator for Server-Sent Events streaming.
    Runs tools synchronously then streams the final LLM answer.
    """
    yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

    try:
        # Run tool calls first (non-streaming)
        result = await executor.run_turn(
            session_id=session_id,
            user_id=user_id,
            query=message,
        )

        plan = result.plan

        # Stream the final answer word by word for UX
        words = plan.final_answer.split()
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]) + " "
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            await asyncio.sleep(0.02)   # ~50 words/sec

        # Final metadata event
        yield f"data: {json.dumps({'type': 'done', 'plan_id': plan.plan_id, 'intent': plan.intent, 'tools_used': plan.tools_used})}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


# ── Onboarding ────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/onboard", response_model=OnboardingResponse, tags=["Onboarding"])
async def onboard(session_id: str):
    """
    Run the automated 3-step onboarding sequence:
      Step 1: Money Health Score (6-dimension wellness assessment)
      Step 2: FIRE feasibility check (retirement readiness)
      Step 3: Tax optimisation opportunities

    Returns all 3 results in one response. Takes 10–20 seconds.
    Recommended for new users immediately after creating a session.
    """
    executor = get_executor()
    session  = executor.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    metrics.increment("api.onboarding_requests")
    log.info("Onboarding started via API", session_id=session_id)

    results = await executor.run_onboarding(
        session_id=session_id,
        state=session.current_state,
    )

    steps = []
    for i, result in enumerate(results, 1):
        steps.append({
            "step":        i,
            "success":     result.success,
            "intent":      result.plan.intent,
            "answer":      result.plan.final_answer,
            "tools_used":  result.plan.tools_used,
            "latency_ms":  result.total_latency_ms,
            "plan_id":     result.plan.plan_id,
        })

    # Aggregate key numbers from all steps
    health_scores  = [r.plan.health_score for r in results if r.plan.health_score]
    sips           = [r.plan.recommended_sip_inr for r in results if r.plan.recommended_sip_inr]
    fire_corpora   = [r.plan.fire_corpus_inr for r in results if r.plan.fire_corpus_inr]
    tax_savings    = [r.plan.tax_saving_inr for r in results if r.plan.tax_saving_inr]

    summary = {
        "health_score":       health_scores[-1] if health_scores else None,
        "recommended_sip_inr": sips[-1] if sips else None,
        "fire_corpus_inr":    fire_corpora[-1] if fire_corpora else None,
        "tax_saving_inr":     tax_savings[-1] if tax_savings else None,
        "steps_completed":    sum(1 for r in results if r.success),
        "steps_total":        len(results),
    }

    return OnboardingResponse(
        session_id=session_id,
        steps=steps,
        summary=summary,
    )


# ── Life events ───────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/life-event", tags=["Life Events"])
async def life_event(session_id: str, body: LifeEventRequest):
    """
    Apply a life event to the user's financial state and get personalised advice.

    Life events trigger a state transition before planning:
      - salary_hike   : Updates income, recalculates savings rate
      - bonus         : Adds lump sum, asks how to allocate optimally
      - marriage      : Updates marital status, accounts for shared expenses
      - new_baby      : Adds dependent, increases expense estimate
      - home_purchase : Adds EMI to debt profile
      - job_loss      : Sets income to 0, focuses on emergency fund
      - inheritance   : Adds lump sum, suggests allocation strategy
      - medical_expense: Deducts from emergency fund / savings

    The message field is optional — a default contextual query is generated.
    """
    executor = get_executor()
    session  = executor.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    metrics.increment("api.life_event_requests", tags={"event": body.event})

    event_input = LifeEventInput(
        event=LifeEvent(body.event),
        amount=body.amount,
        description=body.description,
    )

    result = await executor.run_turn(
        session_id=session_id,
        user_id=session.user_id,
        query=body.message or f"I just experienced a {body.event} of ₹{body.amount:,.0f}. What should I do?",
        life_event=event_input,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Life event processing failed.",
        )

    plan = result.plan
    return {
        "session_id":        session_id,
        "life_event_applied": result.life_event_applied.value if result.life_event_applied else None,
        "state_changed":     result.state_transition is not None,
        "answer":            plan.final_answer,
        "intent":            plan.intent,
        "tools_used":        plan.tools_used,
        "key_numbers": {
            "recommended_sip_inr":  plan.recommended_sip_inr,
            "tax_saving_inr":       plan.tax_saving_inr,
            "health_score":         plan.health_score,
        },
        "latency_ms":  result.total_latency_ms,
        "plan_id":     plan.plan_id,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.post("/evaluate/{user_id}", tags=["Evaluation"])
async def evaluate_user(user_id: str, force: bool = False):
    """
    Trigger batch evaluation for a user's prediction history.

    Returns accuracy metrics (MAE, within-tolerance %) and a drift alert
    if the RL model is underperforming.

    Set force=true to evaluate all episodes regardless of age (for testing).
    Normally only episodes ≥30 days old are evaluated.
    """
    if _evaluator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluator not initialised.",
        )

    metrics.increment("api.evaluation_requests")
    report: EvaluationReport = await _evaluator.run_batch_evaluation(
        user_id=user_id,
        force=force,
    )

    return report.to_dict()


@app.post("/evaluate/system", tags=["Evaluation"])
async def evaluate_system():
    """
    Run system-wide evaluation across all users.
    Designed for scheduled execution (daily/weekly).
    Returns aggregate accuracy stats and retraining signal.
    """
    if _evaluator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluator not initialised.",
        )

    metrics.increment("api.system_evaluation_requests")
    report: EvaluationReport = await _evaluator.run_system_evaluation()
    return report.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# Exception handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(
        "Unhandled exception",
        path=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__,
    )
    metrics.increment("api.unhandled_errors")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error":   "An unexpected error occurred.",
            "details": str(exc) if settings.is_development else "Contact support.",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Interactive CLI demo
# ══════════════════════════════════════════════════════════════════════════════

async def run_demo() -> None:
    """
    Terminal demo — runs the full agent loop without starting an HTTP server.
    Useful for development and quick validation.
    """
    print("\n" + "=" * 65)
    print("  FinMentor AI — Interactive Demo  (type 'quit' to exit)")
    print("=" * 65)

    # Build a demo state
    state = UserFinancialState(
        age=32,
        monthly_income=120_000,
        monthly_expense=55_000,
        risk_profile=RiskProfile.MODERATE,
        risk_score=6.0,
        investment_horizon_years=25,
        emergency_fund_amount=200_000,
        existing_monthly_sip=12_000,
        section_80c_used=80_000,
        nps_contribution=3_000,
        employment_type=EmploymentType.SALARIED,
        city_tier=CityTier.METRO,
        is_married=True,
        dependents=1,
        insurance=InsuranceCoverage(
            has_term_insurance=True,  term_cover_amount=10_000_000,
            has_health_insurance=True, health_cover_amount=500_000,
        ),
        debts=DebtProfile(home_loan_emi=22_000, car_loan_emi=7_000),
        portfolio=InvestmentPortfolio(
            equity_mf=800_000, debt_mf=200_000,
            ppf=100_000, epf=300_000, nps=80_000,
        ),
        goals=[
            FinancialGoal(
                name="Retirement",
                target_amount=60_000_000,
                target_year=2057,
                priority=1,
                existing_corpus=1_480_000,
            ),
            FinancialGoal(
                name="Child Education",
                target_amount=3_000_000,
                target_year=2040,
                priority=2,
                existing_corpus=100_000,
            ),
        ],
    )

    print(f"\n📋 Demo Profile:")
    print(f"   Age          : {state.age}")
    print(f"   Income       : ₹{state.monthly_income:,.0f}/month")
    print(f"   Savings rate : {state.savings_rate*100:.1f}%")
    print(f"   FOIR         : {state.foir*100:.1f}%")
    print(f"   Portfolio    : ₹{state.portfolio.total_value:,.0f}")
    print(f"   Emergency    : {state.emergency_fund_months:.1f} months")
    print()

    executor = build_executor()
    session_id = await executor.start_session(state)
    print(f"✅ Session created: {session_id}\n")

    demo_queries = [
        "What is my Money Health Score and what should I prioritise?",
        "How much SIP do I need to retire comfortably at 60?",
        "How can I save more tax this financial year?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"─" * 65)
        print(f"[Turn {i}] You: {query}")
        print(f"─" * 65)

        result = await executor.run_turn(
            session_id=session_id,
            user_id=state.user_id,
            query=query,
        )

        plan = result.plan
        print(f"\n🤖 FinMentor AI:")
        print(plan.final_answer)
        print(f"\n📊 Tools used    : {plan.tools_used}")
        print(f"⚡ Latency       : {result.total_latency_ms:.0f}ms")
        print(f"🔑 Key numbers   : {json.dumps({'health_score': plan.health_score, 'sip': plan.recommended_sip_inr, 'fire_corpus': plan.fire_corpus_inr}, default=str)}")
        print()

    # Interactive mode
    print(f"─" * 65)
    print("💬 Now ask your own question (type 'quit' to exit):")
    print(f"─" * 65)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = await executor.run_turn(
            session_id=session_id,
            user_id=state.user_id,
            query=user_input,
        )
        print(f"\n🤖 FinMentor AI:\n{result.plan.final_answer}")
        print(f"\n[Tools: {result.plan.tools_used} | {result.total_latency_ms:.0f}ms]")

    await executor.end_session(session_id)
    print("\n👋 Session ended. Goodbye!")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinMentor AI")
    parser.add_argument("--demo",     action="store_true", help="Run interactive demo")
    parser.add_argument("--evaluate", action="store_true", help="Run system evaluation")
    parser.add_argument("--host",     default=settings.server.host)
    parser.add_argument("--port",     type=int, default=settings.server.port)
    parser.add_argument("--reload",   action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(run_demo())

    elif args.evaluate:
        async def _eval():
            store    = get_memory_store()
            ev       = build_evaluator(store)
            report   = await ev.run_system_evaluation()
            print(report.summary())
            print(json.dumps(report.to_dict(), indent=2, default=str))
        asyncio.run(_eval())

    else:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=settings.server.workers if not args.reload else 1,
            reload=args.reload or settings.server.reload,
            log_level=settings.observability.log_level.value.lower(),
            access_log=True,
        )
