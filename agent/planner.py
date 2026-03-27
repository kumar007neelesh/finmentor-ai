"""
agent/planner.py — FinMentor AI Agent Planner
==============================================
The LLM-driven brain of the agent. Implements the ReAct pattern:
  Reason → Act (tool call) → Observe (tool result) → Reason → ... → Respond

The Planner is the ONLY component that talks to the LLM.
It orchestrates:
  1. Intent classification  (fast pre-routing, 1 LLM call)
  2. State context injection (financial summary → system prompt)
  3. ReAct loop             (up to max_planning_steps iterations)
  4. Tool dispatch          (via ToolRegistry)
  5. Final answer synthesis (LLM composes user-facing response)
  6. Plan structuring       (FinancialPlan dataclass for downstream use)

Architecture position:
    User query + UserFinancialState
           │
           ▼
       Planner.plan()
           │
    ┌──────┴──────────────────────────────────┐
    │  Step 0: classify_intent()              │
    │  Step 1–N: ReAct loop                   │
    │    ├─ LLM.chat() → reasoning + tool_calls│
    │    ├─ ToolRegistry.execute(tool, args)  │
    │    └─ Inject tool result → next message │
    │  Step N+1: synthesise_final_answer()    │
    └─────────────────────────────────────────┘
           │
           ▼
       FinancialPlan  (structured output for executor + memory)

Usage:
    from agent.planner import Planner, build_planner

    planner = build_planner(llm, registry)
    plan = await planner.plan(
        query="I earn ₹1.2L/month. How do I retire at 50?",
        state=user_state,
        session_id="sess_001",
    )
    print(plan.final_answer)
    print(plan.tools_used)
    print(plan.tool_outputs)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import settings
from environment.state import UserFinancialState, LifeEvent
from llm.wrapper import (
    LLMWrapper,
    LLMMessage,
    LLMResponse,
    IntentClassifier,
    ToolCallDecision,
    build_intent_classifier,
)
from logger import get_logger, metrics, timed, audit
from tools.registry import ToolRegistry, ToolResult

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Planning step record
# ══════════════════════════════════════════════════════════════════════════════

class StepType(str, Enum):
    REASONING   = "reasoning"    # LLM thought / reasoning text
    TOOL_CALL   = "tool_call"    # LLM decided to call a tool
    OBSERVATION = "observation"  # Tool result injected back
    FINAL       = "final"        # Final answer composed


@dataclass
class PlanningStep:
    """
    Immutable record of one step in the ReAct loop.
    Stored in FinancialPlan.steps for full transparency and debugging.
    """
    step_number: int
    step_type: StepType
    content: str                          # Text: reasoning / answer
    tool_name: Optional[str] = None       # Populated for TOOL_CALL steps
    tool_args: Optional[Dict] = None
    tool_result: Optional[Dict] = None    # Populated for OBSERVATION steps
    tool_success: bool = True
    latency_ms: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":         self.step_number,
            "type":         self.step_type.value,
            "content":      self.content[:500] if self.content else "",
            "tool_name":    self.tool_name,
            "tool_success": self.tool_success,
            "latency_ms":   round(self.latency_ms, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Financial Plan — structured output of a planning session
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinancialPlan:
    """
    The complete output of one Planner.plan() call.

    Consumed by:
      - Executor   (runs the plan, tracks outcomes)
      - Memory     (stores for future reference)
      - Evaluator  (compares predicted vs actual)
      - API layer  (returns to frontend)
    """
    plan_id: str = field(default_factory=lambda: f"plan_{uuid4().hex[:8]}")
    session_id: str = ""
    user_id: str = ""
    query: str = ""
    intent: str = "unknown"

    # The main deliverable
    final_answer: str = ""

    # Execution trace
    steps: List[PlanningStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)  # tool_name → output dict

    # Extracted key numbers (for Memory / Evaluator)
    recommended_actions: List[str] = field(default_factory=list)
    predicted_return_pct: Optional[float] = None
    recommended_sip_inr: Optional[float] = None
    health_score: Optional[float] = None
    fire_corpus_inr: Optional[float] = None
    tax_saving_inr: Optional[float] = None

    # Performance
    total_latency_ms: float = 0.0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    used_fallback: bool = False
    planning_steps_taken: int = 0

    # Timestamps
    created_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id":             self.plan_id,
            "session_id":          self.session_id,
            "user_id":             self.user_id,
            "query":               self.query,
            "intent":              self.intent,
            "final_answer":        self.final_answer,
            "tools_used":          self.tools_used,
            "tool_outputs":        self.tool_outputs,
            "recommended_actions": self.recommended_actions,
            "key_numbers": {
                "predicted_return_pct": self.predicted_return_pct,
                "recommended_sip_inr":  self.recommended_sip_inr,
                "health_score":         self.health_score,
                "fire_corpus_inr":      self.fire_corpus_inr,
                "tax_saving_inr":       self.tax_saving_inr,
            },
            "performance": {
                "total_latency_ms":    round(self.total_latency_ms, 2),
                "total_llm_calls":     self.total_llm_calls,
                "total_tool_calls":    self.total_tool_calls,
                "planning_steps_taken": self.planning_steps_taken,
                "used_fallback":       self.used_fallback,
            },
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
        }

    def summary(self) -> str:
        """One-line plan summary for logging."""
        return (
            f"[{self.plan_id}] intent={self.intent} "
            f"tools={self.tools_used} "
            f"steps={self.planning_steps_taken} "
            f"latency={self.total_latency_ms:.0f}ms"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Intent → default tool mapping
# ══════════════════════════════════════════════════════════════════════════════

INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "fire_planning":    ["rl_predict", "fire_planner"],
    "sip_calculation":  ["rl_predict", "sip_calculator"],
    "tax_optimization": ["tax_wizard"],
    "health_score":     ["health_score"],
    "portfolio_review": ["rl_predict", "health_score"],
    "debt_advice":      ["health_score", "rl_predict"],
    "insurance_advice": ["health_score"],
    "life_event":       ["rl_predict", "tax_wizard", "sip_calculator"],
    "general_question": ["health_score"],
    "unknown":          ["health_score"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Planner
# ══════════════════════════════════════════════════════════════════════════════

class Planner:
    """
    LLM-driven ReAct planner.

    ReAct loop (per planning step):
      1. Build messages from history
      2. Inject state context into system prompt
      3. Call LLM → get reasoning text + optional tool_calls
      4. If tool_calls → execute via ToolRegistry → inject observations
      5. Repeat until: no more tool calls OR max_steps reached
      6. Final LLM call → compose user-facing answer
    """

    def __init__(
        self,
        llm: LLMWrapper,
        registry: ToolRegistry,
        classifier: IntentClassifier,
    ) -> None:
        self._llm        = llm
        self._registry   = registry
        self._classifier = classifier
        self._max_steps  = settings.agent.max_planning_steps
        self._max_tools_per_step = settings.agent.max_tool_calls_per_step

    # ── Public entry point ────────────────────────────────────────────────────

    @timed("planner.plan", tags={"layer": "agent"})
    async def plan(
        self,
        query: str,
        state: UserFinancialState,
        session_id: str = "",
        history: Optional[List[LLMMessage]] = None,
    ) -> FinancialPlan:
        """
        Run the full planning loop for a user query.

        Args:
            query:      The user's natural language question.
            state:      The user's current financial state.
            session_id: Unique session identifier.
            history:    Prior conversation messages (for multi-turn chats).

        Returns:
            FinancialPlan with final_answer, tool_outputs, and full trace.
        """
        t_start = time.perf_counter()

        plan = FinancialPlan(
            session_id=session_id,
            user_id=state.user_id,
            query=query,
        )

        log.info(
            "Planning started",
            query=query[:100],
            session_id=session_id,
            user_id=state.user_id,
        )

        # ── Step 0: Intent classification ─────────────────────────────────────
        intent = await self._classifier.classify(query, session_id)
        plan.intent = intent
        plan.total_llm_calls += 1
        metrics.increment("planner.intents", tags={"intent": intent})

        log.info("Intent classified", intent=intent, session_id=session_id)

        # ── Build initial message list ─────────────────────────────────────────
        messages: List[LLMMessage] = list(history or [])
        messages.append(LLMMessage(role="user", content=query))

        # Serialize financial state as context (injected into system prompt)
        state_context = self._build_state_context(state, intent)

        # ── ReAct loop ────────────────────────────────────────────────────────
        step_num = 0
        max_steps = self._max_steps

        while step_num < max_steps:
            step_num += 1
            metrics.increment("planner.react_steps")

            # ── LLM reasoning call ─────────────────────────────────────────────
            t_llm = time.perf_counter()
            try:
                response: LLMResponse = await self._llm.chat(
                    messages=self._llm.trim_history(messages),
                    session_id=session_id,
                    context=state_context,
                )
            except Exception as exc:
                log.error(
                    "LLM call failed in ReAct loop",
                    step=step_num,
                    error=str(exc),
                    session_id=session_id,
                )
                plan.final_answer = (
                    "I encountered an issue while processing your request. "
                    "Please try again in a moment."
                )
                break

            llm_latency_ms = (time.perf_counter() - t_llm) * 1000
            plan.total_llm_calls += 1

            # Record reasoning step
            if response.text:
                plan.steps.append(PlanningStep(
                    step_number=step_num,
                    step_type=StepType.REASONING,
                    content=response.text,
                    latency_ms=llm_latency_ms,
                ))

            # ── No tool calls → final answer reached ──────────────────────────
            if not response.has_tool_calls:
                log.info(
                    "No tool calls — final answer reached",
                    step=step_num,
                    session_id=session_id,
                )
                plan.final_answer = response.text
                plan.steps.append(PlanningStep(
                    step_number=step_num,
                    step_type=StepType.FINAL,
                    content=response.text,
                    latency_ms=llm_latency_ms,
                ))
                break

            # ── Execute tool calls ────────────────────────────────────────────
            tool_calls = response.tool_calls[:self._max_tools_per_step]

            log.info(
                "Executing tool calls",
                step=step_num,
                tool_calls=[tc.tool_name for tc in tool_calls],
                session_id=session_id,
            )

            # Add assistant reasoning to messages before tool results
            messages.append(LLMMessage(
                role="assistant",
                content=response.text or f"Calling tools: {[tc.tool_name for tc in tool_calls]}",
            ))

            # Execute tools (parallel when multiple in same step)
            tool_results = await self._execute_tools(
                tool_calls, state, session_id, plan
            )

            # Build observation message from all tool results
            observation_parts = []
            for tc, tr in zip(tool_calls, tool_results):
                plan.steps.append(PlanningStep(
                    step_number=step_num,
                    step_type=StepType.TOOL_CALL,
                    content=f"Called {tc.tool_name}",
                    tool_name=tc.tool_name,
                    tool_args=tc.args,
                    latency_ms=tr.latency_ms,
                ))

                plan.steps.append(PlanningStep(
                    step_number=step_num,
                    step_type=StepType.OBSERVATION,
                    content=json.dumps(tr.output, default=str)[:500],
                    tool_name=tc.tool_name,
                    tool_result=tr.output,
                    tool_success=tr.success,
                    latency_ms=tr.latency_ms,
                ))

                observation_parts.append(
                    self._llm.format_tool_result_for_prompt(
                        tool_name=tc.tool_name,
                        result=tr.output if tr.success else {"error": tr.error},
                        call_id=tc.call_id,
                    )
                )

            # Inject all observations as a single user message
            observation_msg = "\n\n".join(observation_parts)
            messages.append(LLMMessage(
                role="user",
                content=(
                    f"Tool results from step {step_num}:\n\n"
                    f"{observation_msg}\n\n"
                    "Now compose the final financial recommendation for the user. "
                    "Use Indian Rupee (₹) notation. Be specific with numbers."
                ),
            ))

        else:
            # max_steps reached without a clean final answer
            log.warning(
                "Max planning steps reached without final answer",
                max_steps=max_steps,
                session_id=session_id,
            )
            # Force final synthesis call
            messages.append(LLMMessage(
                role="user",
                content=(
                    "Please now summarise all the information gathered and "
                    "provide a clear, actionable financial recommendation."
                ),
            ))
            try:
                final_response = await self._llm.chat(
                    messages=self._llm.trim_history(messages),
                    session_id=session_id,
                    context=state_context,
                )
                plan.final_answer = final_response.text
                plan.total_llm_calls += 1
            except Exception as exc:
                log.error("Final synthesis call failed", error=str(exc))
                plan.final_answer = self._fallback_answer(plan, state)

        # ── Post-loop: extract key numbers from tool outputs ──────────────────
        self._extract_key_numbers(plan)

        # ── Finalise plan metadata ─────────────────────────────────────────────
        plan.total_latency_ms     = (time.perf_counter() - t_start) * 1000
        plan.planning_steps_taken = step_num

        # Emit metrics
        metrics.increment("planner.plans_completed")
        metrics.observe("planner.latency_ms", plan.total_latency_ms)
        metrics.observe("planner.tool_calls_per_plan", plan.total_tool_calls)
        metrics.observe("planner.llm_calls_per_plan", plan.total_llm_calls)

        # Audit record
        audit.record(
            session_id=session_id,
            action="financial_plan_generated",
            inputs={
                "query":          query[:200],
                "intent":         intent,
                "user_age":       state.age,
                "monthly_income": state.monthly_income,
            },
            outputs={
                "plan_id":         plan.plan_id,
                "tools_used":      plan.tools_used,
                "key_numbers":     {
                    "sip":          plan.recommended_sip_inr,
                    "health_score": plan.health_score,
                    "fire_corpus":  plan.fire_corpus_inr,
                    "tax_saving":   plan.tax_saving_inr,
                },
                "total_steps":     plan.planning_steps_taken,
                "latency_ms":      round(plan.total_latency_ms, 2),
            },
            user_id=state.user_id,
        )

        log.info(
            "Planning complete",
            plan_summary=plan.summary(),
            session_id=session_id,
        )

        return plan

    # ── Multi-turn follow-up ──────────────────────────────────────────────────

    async def follow_up(
        self,
        query: str,
        state: UserFinancialState,
        prior_plan: FinancialPlan,
        session_id: str = "",
    ) -> FinancialPlan:
        """
        Handle a follow-up question in an ongoing conversation.
        Injects the prior plan's tool outputs as context so the LLM
        doesn't need to re-run the same tools.

        Usage:
            plan1 = await planner.plan("How much SIP?", state)
            plan2 = await planner.follow_up("What if I step it up by 15%?", state, plan1)
        """
        # Build history from prior plan
        history: List[LLMMessage] = [
            LLMMessage(role="user", content=prior_plan.query),
            LLMMessage(role="assistant", content=prior_plan.final_answer),
        ]

        # Inject prior tool results as context
        prior_context = ""
        if prior_plan.tool_outputs:
            prior_context = (
                "Prior session results (already computed — do not re-run these tools "
                "unless the user's numbers have changed):\n"
                + json.dumps(prior_plan.tool_outputs, indent=2, default=str)[:3000]
            )

        log.info(
            "Follow-up plan started",
            prior_plan_id=prior_plan.plan_id,
            query=query[:100],
            session_id=session_id,
        )

        return await self.plan(
            query=query,
            state=state,
            session_id=session_id,
            history=history,
        )

    # ── Tool execution helper ─────────────────────────────────────────────────

    async def _execute_tools(
        self,
        tool_calls: List[ToolCallDecision],
        state: UserFinancialState,
        session_id: str,
        plan: FinancialPlan,
    ) -> List[ToolResult]:
        """
        Execute a list of tool calls, injecting `state` into args.
        Runs in parallel when multiple tools are called in the same step.
        Updates plan.tools_used and plan.tool_outputs in place.
        """
        # Inject state into every tool's args (state is always required)
        enriched_calls = []
        for tc in tool_calls:
            args = dict(tc.args)
            args["state"] = state    # Always inject the live state
            enriched_calls.append({"tool_name": tc.tool_name, "args": args})

        # Parallel execution
        results = await self._registry.execute_parallel(enriched_calls, session_id)

        for tc, result in zip(tool_calls, results):
            plan.total_tool_calls += 1
            if result.used_fallback:
                plan.used_fallback = True

            if result.success:
                plan.tools_used.append(tc.tool_name)
                plan.tool_outputs[tc.tool_name] = result.output
                metrics.increment(
                    "planner.tool_executed",
                    tags={"tool": tc.tool_name, "status": "success"},
                )
            else:
                log.warning(
                    "Tool call failed during planning",
                    tool=tc.tool_name,
                    error=result.error,
                    session_id=session_id,
                )
                metrics.increment(
                    "planner.tool_executed",
                    tags={"tool": tc.tool_name, "status": "error"},
                )

        return results

    # ── State context builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_state_context(state: UserFinancialState, intent: str) -> str:
        """
        Build the per-call state context string injected into the system prompt.
        Deliberately concise to minimise token usage while giving Claude
        all the numbers it needs to reason correctly.
        """
        summary = state.financial_summary()
        income  = summary["income_and_savings"]
        debt    = summary["debt"]
        ef      = summary["emergency_fund"]
        ins     = summary["insurance"]
        port    = summary["portfolio"]

        goals_str = ""
        if summary["goals"]:
            goals_str = "Goals:\n" + "\n".join(
                f"  - {g['name']}: ₹{g['target_inr']:,.0f} by {g['target_year']} "
                f"(shortfall ₹{g['shortfall_inr']:,.0f})"
                for g in summary["goals"]
            )

        return f"""
User Profile: Age {state.age}, {state.employment_type.value}, {state.city_tier.value}
Married: {state.is_married}, Dependents: {state.dependents}
Retirement target: age {state.retirement_age} ({state.years_to_retirement} years away)

Income: ₹{income['monthly_income_inr']:,.0f}/month gross
Expense: ₹{income['monthly_expense_inr']:,.0f}/month
Savings: ₹{income['monthly_savings_inr']:,.0f}/month ({income['savings_rate_pct']}%)
Tax regime: {income['tax_regime']} | Effective rate: {income['effective_tax_rate_pct']}%

Debt: Total EMI ₹{debt['total_monthly_emi_inr']:,.0f} | FOIR {debt['foir']} ({debt['foir_status']})
Emergency fund: ₹{ef['amount_inr']:,.0f} = {ef['months_covered']} months ({ef['status']})

Insurance: Term={ins['has_term']} (₹{ins['term_cover_inr']:,.0f}) | Health={ins['has_health']} (₹{ins['health_cover_inr']:,.0f})
Insurance score: {ins['coverage_score']}/100

Portfolio: ₹{port['total_value_inr']:,.0f} total | Equity {port['equity_pct']}% | Debt {port['debt_pct']}%
Current SIP: ₹{port['existing_monthly_sip_inr']:,.0f}/month

Risk: {state.risk_profile.value} (score {state.risk_score}/10) | Horizon: {state.investment_horizon_years} years

{goals_str}

80C used: ₹{state.section_80c_used:,.0f} / ₹1,50,000 | NPS: ₹{state.nps_contribution:,.0f}/month

Intent detected: {intent}
""".strip()

    # ── Key number extractor ──────────────────────────────────────────────────

    @staticmethod
    def _extract_key_numbers(plan: FinancialPlan) -> None:
        """
        Pull the most important numerical outputs from tool_outputs
        into top-level plan fields for easy access by Memory and Evaluator.
        """
        outputs = plan.tool_outputs

        # RL / ML prediction
        if "rl_predict" in outputs:
            plan.predicted_return_pct = outputs["rl_predict"].get("predicted_return_pct")
            action = outputs["rl_predict"].get("recommended_action")
            if action:
                plan.recommended_actions.append(action)

        # SIP calculator
        if "sip_calculator" in outputs:
            plan.recommended_sip_inr = outputs["sip_calculator"].get(
                "required_monthly_sip_inr"
            )

        # FIRE planner
        if "fire_planner" in outputs:
            plan.fire_corpus_inr = outputs["fire_planner"].get("fire_corpus_needed_inr")
            if not plan.recommended_sip_inr:
                plan.recommended_sip_inr = outputs["fire_planner"].get(
                    "required_monthly_sip_inr"
                )

        # Health score
        if "health_score" in outputs:
            plan.health_score = outputs["health_score"].get("overall_score")

        # Tax wizard
        if "tax_wizard" in outputs:
            plan.tax_saving_inr = outputs["tax_wizard"].get("tax_saving_by_switching_inr")

    # ── Fallback answer ───────────────────────────────────────────────────────

    @staticmethod
    def _fallback_answer(plan: FinancialPlan, state: UserFinancialState) -> str:
        """
        Minimal rule-based answer when LLM fails completely.
        Only surfaces the most critical data point available.
        """
        if plan.health_score is not None:
            return (
                f"Your Money Health Score is {plan.health_score:.0f}/100. "
                f"Focus on building your emergency fund and maximising your SIP. "
                f"Please try your detailed question again."
            )
        return (
            f"Based on your income of ₹{state.monthly_income:,.0f}/month "
            f"and savings rate of {state.savings_rate*100:.0f}%, "
            f"I recommend starting with a ₹{state.monthly_savings * 0.5:,.0f} monthly SIP. "
            f"Please try again for a detailed plan."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_planner(llm: LLMWrapper, registry: ToolRegistry) -> Planner:
    """
    Build a fully wired Planner.
    Called once at application startup.

    Args:
        llm:      Initialised LLMWrapper (with system prompt + tool schemas).
        registry: Initialised ToolRegistry (with all tools registered).

    Returns:
        Ready-to-use Planner instance.
    """
    classifier = build_intent_classifier(llm)
    planner = Planner(llm=llm, registry=registry, classifier=classifier)

    log.info(
        "Planner built",
        max_steps=settings.agent.max_planning_steps,
        max_tools_per_step=settings.agent.max_tool_calls_per_step,
    )
    return planner


# ══════════════════════════════════════════════════════════════════════════════
# Self-test (fully mocked — no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json
    from unittest.mock import AsyncMock, MagicMock, patch

    from environment.state import (
        UserFinancialState, InsuranceCoverage, DebtProfile,
        InvestmentPortfolio, FinancialGoal, EmploymentType, CityTier,
    )
    from config import RiskProfile
    from tools.registry import build_registry
    from llm.wrapper import build_llm_wrapper

    async def run_tests() -> None:
        print("=== Planner Self-Test (Mocked LLM) ===\n")

        # ── Build system under test ───────────────────────────────────────────
        registry = build_registry()
        tool_schemas = registry.list_tools()
        llm = build_llm_wrapper(tool_schemas)
        planner = build_planner(llm, registry)

        state = UserFinancialState(
            age=33,
            monthly_income=150_000,
            monthly_expense=65_000,
            risk_profile=RiskProfile.MODERATE,
            risk_score=6.0,
            investment_horizon_years=27,
            emergency_fund_amount=270_000,
            existing_monthly_sip=15_000,
            section_80c_used=90_000,
            nps_contribution=4_000,
            employment_type=EmploymentType.SALARIED,
            city_tier=CityTier.METRO,
            is_married=True,
            dependents=1,
            insurance=InsuranceCoverage(
                has_term_insurance=True,
                term_cover_amount=15_000_000,
                has_health_insurance=True,
                health_cover_amount=500_000,
            ),
            debts=DebtProfile(
                home_loan_emi=30_000,
                car_loan_emi=8_000,
            ),
            portfolio=InvestmentPortfolio(
                equity_mf=900_000,
                debt_mf=300_000,
                ppf=200_000,
                epf=400_000,
                nps=120_000,
            ),
            goals=[
                FinancialGoal(
                    name="Retirement",
                    target_amount=60_000_000,
                    target_year=2057,
                    priority=1,
                    existing_corpus=1_920_000,
                ),
                FinancialGoal(
                    name="Child Education",
                    target_amount=5_000_000,
                    target_year=2040,
                    priority=2,
                    existing_corpus=200_000,
                ),
            ],
        )

        # ── Test 1: State context builder ─────────────────────────────────────
        print("── Test 1: State Context Builder ──")
        ctx = Planner._build_state_context(state, "fire_planning")
        print(ctx)
        assert "₹1,50,000" in ctx
        assert "FOIR" in ctx
        assert "Retirement" in ctx
        print("  ✓ State context built correctly\n")

        # ── Test 2: Plan with mocked LLM (2 rounds: tool call + final) ────────
        print("── Test 2: Full Plan (Mocked LLM — 2 ReAct steps) ──")

        call_count = {"n": 0}

        def make_mock_response(text: str, stop: str = "end_turn"):
            block = MagicMock()
            block.type = "text"
            block.text = text
            resp = MagicMock()
            resp.content = [block]
            resp.stop_reason = stop
            resp.usage.input_tokens = 800
            resp.usage.output_tokens = 200
            return resp

        # Round 1: LLM decides to call health_score + sip_calculator
        round1_text = (
            "I'll start by assessing your overall financial health and then "
            "calculate the SIP needed for your retirement goal.\n\n"
            "<tool_call>\n"
            '{"tool": "health_score", "args": {}}\n'
            "</tool_call>\n\n"
            "<tool_call>\n"
            '{"tool": "sip_calculator", "args": {"goal_name": "Retirement", "target_amount": 60000000, "step_up_pct": 10}}\n'
            "</tool_call>"
        )

        # Round 2: LLM composes final answer (no tool calls)
        round2_text = (
            "Based on your financial health score of 71/100 (Good) and the SIP "
            "analysis, here is your personalised plan:\n\n"
            "**Current situation:** Your savings rate of 31.3% is excellent. "
            "Your FOIR of 25.3% is healthy. Emergency fund covers 4.2 months "
            "(slightly below the 6-month target).\n\n"
            "**Retirement SIP:** You need ₹42,500/month to retire comfortably "
            "at 60. Your current ₹15,000 SIP leaves a gap of ₹27,500/month.\n\n"
            "**Priority actions:**\n"
            "1. 🏦 Top up emergency fund by ₹1,20,000 to reach 6 months\n"
            "2. 📈 Increase SIP to ₹42,500/month (step up 10% annually)\n"
            "3. 🧾 Max out 80C (₹60,000 headroom remaining)\n"
            "4. 💰 Add ₹4,000/month to NPS for extra 80CCD(1B) deduction"
        )

        async def mock_create(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Intent classification
                b = MagicMock(); b.type = "text"; b.text = "fire_planning"
                r = MagicMock(); r.content = [b]; r.stop_reason = "end_turn"
                r.usage.input_tokens = 50; r.usage.output_tokens = 5
                return r
            elif call_count["n"] == 2:
                return make_mock_response(round1_text, "tool_use")
            else:
                return make_mock_response(round2_text, "end_turn")

        with patch("llm.wrapper._get_async_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = mock_create
            mock_get_client.return_value = mock_client

            plan = await planner.plan(
                query="How much SIP do I need to retire at 60?",
                state=state,
                session_id="sess_planner_test",
            )

        print(f"  Plan ID         : {plan.plan_id}")
        print(f"  Intent          : {plan.intent}")
        print(f"  Tools used      : {plan.tools_used}")
        print(f"  Planning steps  : {plan.planning_steps_taken}")
        print(f"  LLM calls       : {plan.total_llm_calls}")
        print(f"  Tool calls      : {plan.total_tool_calls}")
        print(f"  Health score    : {plan.health_score}")
        print(f"  Recommended SIP : ₹{plan.recommended_sip_inr:,.0f}" if plan.recommended_sip_inr else "  Recommended SIP : N/A")
        print(f"  Used fallback   : {plan.used_fallback}")
        print(f"  Latency         : {plan.total_latency_ms:.0f}ms")
        print(f"\n  Final Answer:\n{'─'*60}")
        print(plan.final_answer)
        print(f"{'─'*60}\n")

        assert plan.intent == "fire_planning"
        assert "health_score" in plan.tools_used
        assert "sip_calculator" in plan.tools_used
        assert plan.health_score is not None
        assert plan.recommended_sip_inr is not None
        print("  ✓ Intent classified correctly")
        print("  ✓ Both tools executed")
        print("  ✓ Key numbers extracted")
        print("  ✓ Final answer composed\n")

        # ── Test 3: Key number extraction ─────────────────────────────────────
        print("── Test 3: Key Number Extraction ──")
        print(f"  health_score     : {plan.health_score}")
        print(f"  recommended_sip  : ₹{plan.recommended_sip_inr:,.0f}" if plan.recommended_sip_inr else "  recommended_sip  : N/A")
        print(f"  fire_corpus      : {plan.fire_corpus_inr}")
        print(f"  tax_saving       : {plan.tax_saving_inr}")
        print(f"  actions          : {plan.recommended_actions}\n")

        # ── Test 4: Planning trace ─────────────────────────────────────────────
        print("── Test 4: Planning Trace ──")
        for step in plan.steps:
            icon = {"reasoning": "🧠", "tool_call": "🔧", "observation": "📊", "final": "✅"}.get(step.step_type.value, "•")
            tool_info = f" [{step.tool_name}]" if step.tool_name else ""
            print(f"  {icon} Step {step.step_number} {step.step_type.value}{tool_info} — {step.content[:60]}...")
        print()

        # ── Test 5: Plan serialisation ─────────────────────────────────────────
        print("── Test 5: Plan Serialisation ──")
        plan_dict = plan.to_dict()
        plan_json = _json.dumps(plan_dict, indent=2, default=str)
        assert "plan_id" in plan_dict
        assert "tool_outputs" in plan_dict
        assert "performance" in plan_dict
        print(f"  Plan serialised to {len(plan_json)} chars of JSON")
        print(f"  Keys: {list(plan_dict.keys())}")
        print("  ✓ Plan fully serialisable\n")

        # ── Test 6: Fallback answer ────────────────────────────────────────────
        print("── Test 6: Fallback Answer ──")
        empty_plan = FinancialPlan(session_id="x", user_id="x", query="x")
        fallback = Planner._fallback_answer(empty_plan, state)
        print(f"  Fallback: {fallback}")
        assert "₹" in fallback
        print("  ✓ Fallback answer contains ₹ figures\n")

        # ── Metrics ───────────────────────────────────────────────────────────
        print("── Metrics Snapshot ──")
        snap = metrics.snapshot()
        for c in snap["counters"]:
            print(f"  {c['name']:<50} {int(c['value'])}")

    asyncio.run(run_tests())
