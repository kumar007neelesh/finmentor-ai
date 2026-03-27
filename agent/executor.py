"""
agent/executor.py — FinMentor AI Agent Executor
================================================
The outer orchestration layer above the Planner.

Responsibilities:
  - Manage the full session lifecycle (start → turns → end)
  - Accept user queries AND life events (salary hike, bonus, marriage, etc.)
  - Apply state transitions when life events occur
  - Call Planner.plan() for each turn, Planner.follow_up() for continuations
  - Persist completed plans to Memory after every turn
  - Expose a clean run_turn() interface used by main.py and the API layer
  - Gracefully handle partial failures (LLM down, tool error) without crashing

Architecture position:
    API / CLI / main.py
           │
           ▼
       Executor.run_turn(query, life_event?)
           │
    ┌──────┴────────────────────────────────────────┐
    │  1. Apply life event → state transition       │
    │  2. Planner.plan() or Planner.follow_up()     │
    │  3. Memory.store(plan, state_snapshot)        │
    │  4. Return ExecutorResult                     │
    └───────────────────────────────────────────────┘

Usage:
    from agent.executor import Executor, build_executor, LifeEventInput

    executor = build_executor()

    # First turn
    result = await executor.run_turn(
        session_id="sess_001",
        user_id="u_001",
        query="I earn ₹1.2L/month. Help me plan my finances.",
        state=user_state,
    )
    print(result.plan.final_answer)

    # Follow-up
    result2 = await executor.run_turn(
        session_id="sess_001",
        user_id="u_001",
        query="What if I increase my SIP by ₹5,000?",
        state=user_state,
        prior_plan=result.plan,
    )

    # Life event
    result3 = await executor.run_turn(
        session_id="sess_001",
        user_id="u_001",
        query="I just got a ₹3 lakh bonus. How should I use it?",
        state=user_state,
        life_event=LifeEventInput(
            event=LifeEvent.BONUS,
            amount=300_000,
        ),
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import settings
from environment.state import (
    UserFinancialState,
    StateTransition,
    LifeEvent,
)
from agent.planner import Planner, FinancialPlan, build_planner
from llm.wrapper import LLMWrapper, build_llm_wrapper
from logger import get_logger, metrics, timed, set_context, clear_context
from tools.registry import ToolRegistry, build_registry

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Life event input
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LifeEventInput:
    """
    Structured life event provided alongside a user query.
    Triggers a state transition before planning begins.
    """
    event: LifeEvent
    amount: float = 0.0           # ₹ amount (bonus, inheritance, salary hike delta)
    description: str = ""         # Optional free-text context


# ══════════════════════════════════════════════════════════════════════════════
# State transition handlers
# ══════════════════════════════════════════════════════════════════════════════

class StateTransitionEngine:
    """
    Applies life-event-driven mutations to UserFinancialState.

    Each handler:
      - Takes the current state and event details
      - Returns a NEW state (immutable — original is preserved for transition record)
      - Logs what changed

    Mutations are intentionally conservative: we update only the fields
    that are directly and unambiguously affected by the event.
    """

    @staticmethod
    def apply(
        state: UserFinancialState,
        event: LifeEventInput,
    ) -> UserFinancialState:
        """
        Route to the correct handler and return the mutated state copy.
        """
        handler_map = {
            LifeEvent.SALARY_HIKE:     StateTransitionEngine._salary_hike,
            LifeEvent.BONUS:           StateTransitionEngine._bonus,
            LifeEvent.INHERITANCE:     StateTransitionEngine._inheritance,
            LifeEvent.MARRIAGE:        StateTransitionEngine._marriage,
            LifeEvent.NEW_BABY:        StateTransitionEngine._new_baby,
            LifeEvent.HOME_PURCHASE:   StateTransitionEngine._home_purchase,
            LifeEvent.JOB_LOSS:        StateTransitionEngine._job_loss,
            LifeEvent.MEDICAL_EXPENSE: StateTransitionEngine._medical_expense,
            LifeEvent.RETIREMENT:      StateTransitionEngine._retirement,
            LifeEvent.NONE:            StateTransitionEngine._no_op,
        }

        handler = handler_map.get(event.event, StateTransitionEngine._no_op)
        updated = handler(state, event)

        # Tag the state with the life event
        updated = updated.model_copy(update={
            "last_life_event":        event.event,
            "last_life_event_amount": event.amount,
            "last_updated":           datetime.now(tz=timezone.utc),
        })

        log.info(
            "State transition applied",
            event=event.event.value,
            amount=event.amount,
            user_id=state.user_id,
            fingerprint_before=state.fingerprint(),
            fingerprint_after=updated.fingerprint(),
        )

        return updated

    # ── Handlers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _salary_hike(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Salary hike: increase monthly_income by event amount (treated as delta ₹/month).
        Recalculate annual_income. Savings rate improves automatically.
        """
        new_income = s.monthly_income + e.amount
        return s.model_copy(update={
            "monthly_income": new_income,
            "annual_income":  new_income * 12,
        })

    @staticmethod
    def _bonus(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Bonus: add lump sum to savings_account (most conservative assumption).
        The Planner will decide optimal allocation (emergency fund / SIP / debt).
        """
        from environment.state import InvestmentPortfolio
        updated_portfolio = s.portfolio.model_copy(update={
            "savings_account": s.portfolio.savings_account + e.amount,
        })
        return s.model_copy(update={"portfolio": updated_portfolio})

    @staticmethod
    def _inheritance(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Inheritance: large lump sum → savings_account pending allocation advice.
        """
        from environment.state import InvestmentPortfolio
        updated_portfolio = s.portfolio.model_copy(update={
            "savings_account": s.portfolio.savings_account + e.amount,
        })
        return s.model_copy(update={"portfolio": updated_portfolio})

    @staticmethod
    def _marriage(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Marriage: flag as married, expect higher expenses.
        amount = spouse's monthly income contribution (if known, else 0).
        """
        extra_income = e.amount   # Spouse income if provided
        updates: Dict[str, Any] = {"is_married": True}
        if extra_income > 0:
            updates["monthly_income"] = s.monthly_income + extra_income
            updates["annual_income"]  = (s.monthly_income + extra_income) * 12
        # Conservative: increase expense estimate by 20%
        updates["monthly_expense"] = s.monthly_expense * 1.20
        return s.model_copy(update=updates)

    @staticmethod
    def _new_baby(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        New baby: +1 dependent, higher expenses (~₹15,000/month typical India).
        """
        baby_expense = e.amount if e.amount > 0 else 15_000
        return s.model_copy(update={
            "dependents":     s.dependents + 1,
            "monthly_expense": s.monthly_expense + baby_expense,
        })

    @staticmethod
    def _home_purchase(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Home purchase: add home loan EMI (amount = monthly EMI), add real estate to portfolio.
        """
        from environment.state import DebtProfile
        updated_debt = s.debts.model_copy(update={
            "home_loan_emi": s.debts.home_loan_emi + e.amount,
        })
        return s.model_copy(update={"debts": updated_debt})

    @staticmethod
    def _job_loss(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Job loss: income drops to 0, emergency fund becomes primary income source.
        Severity of the event is captured for the Planner to reason about.
        """
        return s.model_copy(update={
            "monthly_income":   0.0,
            "annual_income":    0.0,
            "employment_type":  "self_employed",  # Unemployed → self_employed proxy
        })

    @staticmethod
    def _medical_expense(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Medical expense: deduct from emergency fund / savings_account.
        """
        from environment.state import InvestmentPortfolio
        deduction = e.amount
        new_ef = max(0.0, s.emergency_fund_amount - deduction)
        remaining = deduction - s.emergency_fund_amount
        new_savings = max(0.0, s.portfolio.savings_account - remaining)
        updated_portfolio = s.portfolio.model_copy(update={"savings_account": new_savings})
        return s.model_copy(update={
            "emergency_fund_amount": new_ef,
            "portfolio": updated_portfolio,
        })

    @staticmethod
    def _retirement(s: UserFinancialState, e: LifeEventInput) -> UserFinancialState:
        """
        Retirement reached: income drops to 0 (will come from corpus).
        age = retirement_age.
        """
        return s.model_copy(update={
            "age":            s.retirement_age,
            "monthly_income": 0.0,
            "annual_income":  0.0,
            "employment_type": "retired",
        })

    @staticmethod
    def _no_op(s: UserFinancialState, _: LifeEventInput) -> UserFinancialState:
        return s


# ══════════════════════════════════════════════════════════════════════════════
# Executor result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutorResult:
    """
    Complete output of one executor turn.
    Returned to the API layer / CLI.
    """
    session_id: str
    turn_number: int
    plan: FinancialPlan
    state_before: Optional[Dict[str, Any]] = None   # Serialised snapshot
    state_after: Optional[Dict[str, Any]] = None
    state_transition: Optional[StateTransition] = None
    life_event_applied: Optional[LifeEvent] = None
    total_latency_ms: float = 0.0
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":         self.session_id,
            "turn_number":        self.turn_number,
            "plan":               self.plan.to_dict() if self.plan else None,
            "life_event_applied": self.life_event_applied.value if self.life_event_applied else None,
            "state_changed":      self.state_transition is not None,
            "total_latency_ms":   round(self.total_latency_ms, 2),
            "error":              self.error,
            "success":            self.success,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Session state tracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    """
    In-memory tracker for a single user session.
    Holds the evolving financial state, conversation history,
    and prior plans for multi-turn continuity.
    """
    session_id: str
    user_id: str
    current_state: UserFinancialState
    turn_count: int = 0
    prior_plan: Optional[FinancialPlan] = None
    transitions: List[StateTransition] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def update_state(self, new_state: UserFinancialState) -> None:
        self.current_state = new_state

    def record_transition(self, transition: StateTransition) -> None:
        self.transitions.append(transition)

    @property
    def has_prior_plan(self) -> bool:
        return self.prior_plan is not None


# ══════════════════════════════════════════════════════════════════════════════
# Executor
# ══════════════════════════════════════════════════════════════════════════════

class Executor:
    """
    Outer orchestration loop.

    Manages:
      - Session registry (in-memory; swap with Redis for horizontal scaling)
      - Life event handling → StateTransitionEngine → new state
      - Planner routing (plan vs follow_up)
      - Memory persistence after every turn
      - Graceful error containment (never raises; always returns ExecutorResult)
    """

    def __init__(
        self,
        planner: Planner,
        registry: ToolRegistry,
    ) -> None:
        self._planner  = planner
        self._registry = registry
        # In-memory session store: session_id → SessionState
        # Production: replace with Redis-backed session store
        self._sessions: Dict[str, SessionState] = {}
        self._session_lock = asyncio.Lock()

    # ── Session management ────────────────────────────────────────────────────

    async def start_session(
        self,
        state: UserFinancialState,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new session for a user.
        Returns the session_id.

        On first contact, automatically runs health_score to establish baseline.
        """
        sid = session_id or f"sess_{uuid4().hex[:8]}"
        async with self._session_lock:
            self._sessions[sid] = SessionState(
                session_id=sid,
                user_id=state.user_id,
                current_state=state,
            )

        metrics.increment("executor.sessions_started")
        log.info(
            "Session started",
            session_id=sid,
            user_id=state.user_id,
            age=state.age,
            monthly_income=state.monthly_income,
        )
        return sid

    async def end_session(self, session_id: str) -> None:
        """Clean up session from in-memory store."""
        async with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
        metrics.increment("executor.sessions_ended")
        log.info("Session ended", session_id=session_id)

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    # ── Main turn entry point ─────────────────────────────────────────────────

    @timed("executor.run_turn", tags={"layer": "executor"})
    async def run_turn(
        self,
        session_id: str,
        user_id: str,
        query: str,
        state: Optional[UserFinancialState] = None,
        life_event: Optional[LifeEventInput] = None,
        prior_plan: Optional[FinancialPlan] = None,
    ) -> ExecutorResult:
        """
        Execute one conversation turn.

        Workflow:
          1. Resolve session state (from registry or provided state)
          2. Apply life event → state transition (if provided)
          3. Route to Planner.plan() or Planner.follow_up()
          4. Persist plan to memory
          5. Return ExecutorResult

        Args:
            session_id:  Session identifier. Must exist (call start_session first)
                         OR pass state= to auto-create.
            user_id:     User identifier for audit logs.
            query:       User's natural language query.
            state:       Provide on first turn or when state changes externally.
                         If None, uses the session's current state.
            life_event:  Optional life event to apply before planning.
            prior_plan:  Provide to trigger follow_up() instead of plan().

        Returns:
            ExecutorResult — always succeeds (errors captured in .error field).
        """
        t_start = time.perf_counter()

        # Bind request context to all log lines in this scope
        set_context(session_id=session_id, user_id=user_id)

        try:
            return await self._run_turn_internal(
                session_id=session_id,
                user_id=user_id,
                query=query,
                state=state,
                life_event=life_event,
                prior_plan=prior_plan,
                t_start=t_start,
            )
        except Exception as exc:
            # Top-level safety net — executor must never raise
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            log.error(
                "Executor turn failed unexpectedly",
                error=str(exc),
                query=query[:100],
                session_id=session_id,
            )
            metrics.increment("executor.unhandled_errors")

            # Return a safe fallback result
            fallback_plan = FinancialPlan(
                session_id=session_id,
                user_id=user_id,
                query=query,
                final_answer=(
                    "I'm having trouble processing your request right now. "
                    "Please try again in a moment. Your financial data is safe."
                ),
            )
            return ExecutorResult(
                session_id=session_id,
                turn_number=0,
                plan=fallback_plan,
                total_latency_ms=elapsed_ms,
                error=str(exc),
                success=False,
            )
        finally:
            clear_context()

    async def _run_turn_internal(
        self,
        session_id: str,
        user_id: str,
        query: str,
        state: Optional[UserFinancialState],
        life_event: Optional[LifeEventInput],
        prior_plan: Optional[FinancialPlan],
        t_start: float,
    ) -> ExecutorResult:

        # ── 1. Resolve session ─────────────────────────────────────────────────
        session = self._sessions.get(session_id)

        if session is None:
            if state is None:
                raise ValueError(
                    f"Session '{session_id}' not found and no state provided. "
                    "Call start_session() first or pass state=."
                )
            # Auto-create session
            await self.start_session(state, session_id=session_id)
            session = self._sessions[session_id]
        elif state is not None:
            # Caller provided an updated state — use it
            session.update_state(state)

        current_state = session.current_state
        state_before_snapshot = current_state.financial_summary()
        session.turn_count += 1
        turn_number = session.turn_count

        log.info(
            "Turn started",
            turn=turn_number,
            query=query[:100],
            has_life_event=life_event is not None,
            is_follow_up=prior_plan is not None or session.has_prior_plan,
            session_id=session_id,
        )

        # ── 2. Apply life event ────────────────────────────────────────────────
        state_transition: Optional[StateTransition] = None
        life_event_applied: Optional[LifeEvent] = None

        if life_event and life_event.event != LifeEvent.NONE:
            log.info(
                "Applying life event",
                event=life_event.event.value,
                amount=life_event.amount,
                session_id=session_id,
            )
            metrics.increment(
                "executor.life_events",
                tags={"event": life_event.event.value},
            )

            updated_state = StateTransitionEngine.apply(current_state, life_event)

            # Record the transition
            state_transition = StateTransition.create(
                state_before=current_state,
                state_after=updated_state,
                life_event=life_event.event,
                event_amount=life_event.amount,
            )
            session.record_transition(state_transition)
            session.update_state(updated_state)
            current_state = updated_state
            life_event_applied = life_event.event

        # ── 3. Route to planner ────────────────────────────────────────────────
        # Determine if this is a follow-up or a fresh plan
        effective_prior = prior_plan or session.prior_plan
        is_follow_up = (
            effective_prior is not None
            and not life_event_applied  # Life events always trigger fresh plan
            and turn_number > 1
        )

        try:
            if is_follow_up and effective_prior is not None:
                log.info(
                    "Routing to follow_up planner",
                    prior_plan_id=effective_prior.plan_id,
                    session_id=session_id,
                )
                plan = await self._planner.follow_up(
                    query=query,
                    state=current_state,
                    prior_plan=effective_prior,
                    session_id=session_id,
                )
            else:
                log.info(
                    "Routing to fresh planner",
                    session_id=session_id,
                )
                plan = await self._planner.plan(
                    query=query,
                    state=current_state,
                    session_id=session_id,
                )

        except Exception as exc:
            log.error(
                "Planner failed",
                error=str(exc),
                session_id=session_id,
            )
            # Build a minimal plan with error info
            plan = FinancialPlan(
                session_id=session_id,
                user_id=user_id,
                query=query,
                final_answer=self._emergency_answer(current_state, life_event),
            )
            metrics.increment("executor.planner_errors")

        # Update predicted reward in transition (for Evaluator)
        if state_transition and plan.predicted_return_pct is not None:
            state_transition.predicted_reward = plan.predicted_return_pct / 100.0

        # ── 4. Persist to memory ───────────────────────────────────────────────
        if settings.agent.enable_memory:
            await self._persist_to_memory(
                plan=plan,
                state=current_state,
                session_id=session_id,
                state_transition=state_transition,
            )

        # ── 5. Update session ──────────────────────────────────────────────────
        session.prior_plan = plan

        # ── 6. Finalise result ─────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        metrics.increment("executor.turns_completed")
        metrics.observe("executor.turn_latency_ms", elapsed_ms)

        log.info(
            "Turn complete",
            turn=turn_number,
            plan_id=plan.plan_id,
            tools_used=plan.tools_used,
            latency_ms=round(elapsed_ms, 2),
            session_id=session_id,
        )

        return ExecutorResult(
            session_id=session_id,
            turn_number=turn_number,
            plan=plan,
            state_before=state_before_snapshot,
            state_after=current_state.financial_summary(),
            state_transition=state_transition,
            life_event_applied=life_event_applied,
            total_latency_ms=round(elapsed_ms, 2),
            success=True,
        )

    # ── Memory persistence ────────────────────────────────────────────────────

    async def _persist_to_memory(
        self,
        plan: FinancialPlan,
        state: UserFinancialState,
        session_id: str,
        state_transition: Optional[StateTransition],
    ) -> None:
        """
        Persist plan to memory store (JSON file by default).
        Non-blocking — errors are logged but do not fail the turn.
        """
        try:
            # Lazy import to avoid circular dependency
            from agent.memory import get_memory_store
            memory = get_memory_store()
            await memory.store(
                plan=plan,
                state=state,
                transition=state_transition,
                session_id=session_id,
            )
        except Exception as exc:
            log.warning(
                "Memory persistence failed (non-fatal)",
                error=str(exc),
                session_id=session_id,
            )

    # ── Batch processing (for onboarding flows) ───────────────────────────────

    async def run_onboarding(
        self,
        session_id: str,
        state: UserFinancialState,
    ) -> List[ExecutorResult]:
        """
        Automatically run the standard onboarding sequence for a new user:
          1. Money Health Score
          2. FIRE feasibility check
          3. Tax optimisation opportunities

        Returns a list of ExecutorResults (one per onboarding step).
        """
        log.info(
            "Running onboarding sequence",
            session_id=session_id,
            user_id=state.user_id,
        )
        metrics.increment("executor.onboarding_runs")

        onboarding_queries = [
            "Give me my complete Money Health Score and top 3 priority actions.",
            "Is my current savings rate enough to retire at my target age? What FIRE corpus do I need?",
            "Identify every tax saving I'm missing and tell me which tax regime suits me better.",
        ]

        results = []
        for i, query in enumerate(onboarding_queries):
            result = await self.run_turn(
                session_id=session_id,
                user_id=state.user_id,
                query=query,
                state=state if i == 0 else None,  # State only needed on first turn
            )
            results.append(result)

            if not result.success:
                log.warning(
                    "Onboarding step failed, continuing",
                    step=i + 1,
                    error=result.error,
                    session_id=session_id,
                )

        log.info(
            "Onboarding complete",
            steps=len(results),
            successful=sum(1 for r in results if r.success),
            session_id=session_id,
        )
        return results

    # ── Life event convenience methods ────────────────────────────────────────

    async def handle_bonus(
        self,
        session_id: str,
        user_id: str,
        state: UserFinancialState,
        bonus_amount: float,
        query: Optional[str] = None,
    ) -> ExecutorResult:
        """
        Convenience wrapper for bonus events.
        Applies the bonus to state and asks the Planner how to allocate it.
        """
        return await self.run_turn(
            session_id=session_id,
            user_id=user_id,
            query=query or f"I received a ₹{bonus_amount:,.0f} bonus. How should I allocate it optimally?",
            state=state,
            life_event=LifeEventInput(
                event=LifeEvent.BONUS,
                amount=bonus_amount,
                description="Annual performance bonus",
            ),
        )

    async def handle_salary_hike(
        self,
        session_id: str,
        user_id: str,
        state: UserFinancialState,
        hike_amount_monthly: float,
        query: Optional[str] = None,
    ) -> ExecutorResult:
        """Convenience wrapper for salary hike events."""
        return await self.run_turn(
            session_id=session_id,
            user_id=user_id,
            query=query or (
                f"My salary just increased by ₹{hike_amount_monthly:,.0f}/month. "
                "How should I update my SIP and financial plan?"
            ),
            state=state,
            life_event=LifeEventInput(
                event=LifeEvent.SALARY_HIKE,
                amount=hike_amount_monthly,
                description="Annual increment",
            ),
        )

    async def handle_marriage(
        self,
        session_id: str,
        user_id: str,
        state: UserFinancialState,
        spouse_income: float = 0.0,
        query: Optional[str] = None,
    ) -> ExecutorResult:
        """Convenience wrapper for marriage events."""
        return await self.run_turn(
            session_id=session_id,
            user_id=user_id,
            query=query or (
                "I'm getting married. How should I restructure my finances, "
                "insurance, and investments for our joint financial goals?"
            ),
            state=state,
            life_event=LifeEventInput(
                event=LifeEvent.MARRIAGE,
                amount=spouse_income,
                description="Marriage — joint financial planning",
            ),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return a summary of the current session state."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id":    session_id,
            "user_id":       session.user_id,
            "turn_count":    session.turn_count,
            "transitions":   len(session.transitions),
            "last_plan_id":  session.prior_plan.plan_id if session.prior_plan else None,
            "state_fingerprint": session.current_state.fingerprint(),
            "started_at":    session.created_at.isoformat(),
        }

    def active_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _emergency_answer(
        state: UserFinancialState,
        life_event: Optional[LifeEventInput],
    ) -> str:
        """Minimal rule-based answer when the Planner completely fails."""
        if life_event and life_event.event == LifeEvent.BONUS:
            amount = life_event.amount
            em_gap = max(
                0,
                state.monthly_expense * 6 - state.emergency_fund_amount,
            )
            sip_suggestion = amount * 0.6
            return (
                f"For your ₹{amount:,.0f} bonus, I suggest:\n"
                f"1. Emergency fund top-up: ₹{min(em_gap, amount * 0.2):,.0f}\n"
                f"2. Lump-sum MF investment (ELSS/Index Fund): ₹{sip_suggestion:,.0f}\n"
                f"3. Debt prepayment (if FOIR > 40%): remaining amount\n"
                f"Please consult a SEBI-registered advisor for personalised advice."
            )
        savings = state.monthly_savings
        return (
            f"Based on your ₹{savings:,.0f}/month surplus, "
            f"invest at least ₹{savings * 0.5:,.0f} in a diversified SIP. "
            f"Please try again for a detailed analysis."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_executor() -> Executor:
    """
    Build and wire the full agent stack:
      ToolRegistry → LLMWrapper → Planner → Executor

    Called once at application startup.
    All components are singletons shared across requests.
    """
    registry   = build_registry()
    llm        = build_llm_wrapper(registry.list_tools())
    planner    = build_planner(llm, registry)
    executor   = Executor(planner=planner, registry=registry)

    log.info(
        "Executor ready",
        tools=registry.get_tool_names(),
        model=settings.llm.model,
        max_steps=settings.agent.max_planning_steps,
    )
    return executor


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json
    from unittest.mock import AsyncMock, MagicMock, patch

    from environment.state import (
        UserFinancialState, InsuranceCoverage, DebtProfile,
        InvestmentPortfolio, FinancialGoal, EmploymentType, CityTier,
    )
    from config import RiskProfile

    def _make_state(
        age: int = 32,
        income: float = 120_000,
        expense: float = 55_000,
    ) -> UserFinancialState:
        return UserFinancialState(
            age=age,
            monthly_income=income,
            monthly_expense=expense,
            risk_profile=RiskProfile.MODERATE,
            risk_score=6.0,
            investment_horizon_years=25,
            emergency_fund_amount=200_000,
            existing_monthly_sip=12_000,
            section_80c_used=80_000,
            nps_contribution=3_000,
            employment_type=EmploymentType.SALARIED,
            city_tier=CityTier.METRO,
            is_married=False,
            dependents=0,
            insurance=InsuranceCoverage(
                has_term_insurance=True, term_cover_amount=10_000_000,
                has_health_insurance=True, health_cover_amount=500_000,
            ),
            debts=DebtProfile(home_loan_emi=20_000),
            portfolio=InvestmentPortfolio(
                equity_mf=700_000, debt_mf=200_000,
                ppf=100_000, epf=250_000, nps=80_000,
            ),
        )

    async def run_tests() -> None:
        print("=== Executor Self-Test ===\n")

        # ── Test 1: StateTransitionEngine ────────────────────────────────────
        print("── Test 1: State Transitions ──")
        state = _make_state()

        transitions = [
            (LifeEventInput(LifeEvent.SALARY_HIKE, amount=20_000), "monthly_income", 140_000),
            (LifeEventInput(LifeEvent.BONUS, amount=300_000),       "portfolio.savings_account", 300_000),
            (LifeEventInput(LifeEvent.MARRIAGE, amount=80_000),     "is_married", True),
            (LifeEventInput(LifeEvent.NEW_BABY, amount=0),          "dependents", 1),
            (LifeEventInput(LifeEvent.JOB_LOSS, amount=0),          "monthly_income", 0.0),
        ]

        for event_input, check_field, expected in transitions:
            updated = StateTransitionEngine.apply(state, event_input)

            if "." in check_field:
                obj, attr = check_field.split(".")
                actual = getattr(getattr(updated, obj), attr)
            else:
                actual = getattr(updated, check_field)

            status = "✓" if actual == expected else "✗"
            print(f"  {status} {event_input.event.value:<20} → {check_field}={actual} (expected {expected})")
            assert actual == expected, f"Transition failed: {check_field}={actual} != {expected}"

        print()

        # ── Test 2: Session management ────────────────────────────────────────
        print("── Test 2: Session Lifecycle ──")
        registry = build_registry()

        # Build a minimal mock planner
        mock_plan = FinancialPlan(
            session_id="sess_ex_001",
            user_id="u_001",
            query="test",
            final_answer="Your health score is 72/100. Increase your SIP to ₹25,000/month.",
            tools_used=["health_score", "sip_calculator"],
            health_score=72.0,
            recommended_sip_inr=25_000,
        )

        mock_planner = MagicMock()
        mock_planner.plan       = AsyncMock(return_value=mock_plan)
        mock_planner.follow_up  = AsyncMock(return_value=mock_plan)

        executor = Executor(planner=mock_planner, registry=registry)
        state = _make_state()

        # Start session
        sid = await executor.start_session(state, session_id="sess_ex_001")
        print(f"  Session started : {sid}")
        assert sid == "sess_ex_001"
        assert executor.get_session(sid) is not None

        # First turn
        result1 = await executor.run_turn(
            session_id=sid,
            user_id="u_001",
            query="How healthy are my finances?",
        )
        print(f"  Turn 1 success  : {result1.success}")
        print(f"  Turn 1 plan ID  : {result1.plan.plan_id}")
        print(f"  Turn 1 answer   : {result1.plan.final_answer[:60]}...")
        assert result1.success
        assert result1.turn_number == 1
        assert result1.plan.final_answer != ""

        # Second turn (follow-up)
        result2 = await executor.run_turn(
            session_id=sid,
            user_id="u_001",
            query="What if I step up my SIP by 15% each year?",
        )
        print(f"  Turn 2 success  : {result2.success}")
        print(f"  Turn 2 is followup: {mock_planner.follow_up.called}")
        assert result2.success
        assert result2.turn_number == 2
        # Second turn should use follow_up
        mock_planner.follow_up.assert_called_once()
        print()

        # ── Test 3: Life event turn ────────────────────────────────────────────
        print("── Test 3: Life Event Turn (Bonus) ──")
        result3 = await executor.run_turn(
            session_id=sid,
            user_id="u_001",
            query="Got a ₹2L bonus. Best use?",
            life_event=LifeEventInput(
                event=LifeEvent.BONUS,
                amount=200_000,
                description="Diwali bonus",
            ),
        )
        print(f"  Success             : {result3.success}")
        print(f"  Life event applied  : {result3.life_event_applied}")
        print(f"  State transition    : {result3.state_transition is not None}")
        print(f"  State before income : ₹{result3.state_before['income_and_savings']['monthly_income_inr']:,.0f}")
        assert result3.life_event_applied == LifeEvent.BONUS
        assert result3.state_transition is not None
        # Life event → fresh plan (not follow_up) — mock_planner.plan called again
        assert mock_planner.plan.call_count == 2
        print()

        # ── Test 4: Convenience wrappers ──────────────────────────────────────
        print("── Test 4: Convenience Life Event Wrappers ──")
        state_fresh = _make_state()
        sid2 = await executor.start_session(state_fresh, session_id="sess_ex_002")

        bonus_result = await executor.handle_bonus(
            session_id=sid2,
            user_id="u_002",
            state=state_fresh,
            bonus_amount=500_000,
        )
        print(f"  handle_bonus success: {bonus_result.success}")
        assert bonus_result.life_event_applied == LifeEvent.BONUS

        hike_result = await executor.handle_salary_hike(
            session_id=sid2,
            user_id="u_002",
            state=state_fresh,
            hike_amount_monthly=25_000,
        )
        print(f"  handle_salary_hike  : {hike_result.success}")
        assert hike_result.life_event_applied == LifeEvent.SALARY_HIKE
        print()

        # ── Test 5: Session summary ───────────────────────────────────────────
        print("── Test 5: Session Summary ──")
        summary = executor.session_summary(sid)
        print(f"  {_json.dumps(summary, indent=2, default=str)}")
        assert summary["turn_count"] == 3
        assert summary["transitions"] == 1     # One BONUS transition
        print()

        # ── Test 6: ExecutorResult serialisation ──────────────────────────────
        print("── Test 6: ExecutorResult Serialisation ──")
        result_dict = result1.to_dict()
        assert "plan" in result_dict
        assert "total_latency_ms" in result_dict
        print(f"  Keys: {list(result_dict.keys())}")
        print(f"  Latency: {result_dict['total_latency_ms']}ms")
        print("  ✓ Fully serialisable\n")

        # ── Test 7: Emergency answer fallback ─────────────────────────────────
        print("── Test 7: Emergency Answer ──")
        answer = Executor._emergency_answer(
            state_fresh,
            LifeEventInput(LifeEvent.BONUS, amount=300_000)
        )
        print(f"  {answer[:120]}...")
        assert "₹" in answer
        print("  ✓ Emergency answer contains ₹ amounts\n")

        # ── Metrics ───────────────────────────────────────────────────────────
        print("── Metrics Snapshot ──")
        snap = metrics.snapshot()
        for c in snap["counters"]:
            print(f"  {c['name']:<50} {int(c['value'])}")

        # Cleanup
        await executor.end_session(sid)
        await executor.end_session(sid2)
        assert executor.get_session(sid) is None
        print("\n  ✓ Sessions cleaned up")

    asyncio.run(run_tests())
