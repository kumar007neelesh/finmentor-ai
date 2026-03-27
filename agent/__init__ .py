"""agent/ — FinMentor AI Agent Package"""
from agent.planner import (
    Planner,
    FinancialPlan,
    PlanningStep,
    StepType,
    INTENT_TOOL_MAP,
    build_planner,
)
from agent.executor import (
    Executor,
    ExecutorResult,
    SessionState,
    LifeEventInput,
    StateTransitionEngine,
    build_executor,
)
from agent.memory import (
    Episode,
    BaseMemoryStore,
    JSONMemoryStore,
    InMemoryStore,
    RedisMemoryStore,
    get_memory_store,
    reset_memory_store,
)

__all__ = [
    "Planner", "FinancialPlan", "PlanningStep", "StepType",
    "INTENT_TOOL_MAP", "build_planner",
    "Executor", "ExecutorResult", "SessionState", "LifeEventInput",
    "StateTransitionEngine", "build_executor",
    "Episode", "BaseMemoryStore", "JSONMemoryStore", "InMemoryStore",
    "RedisMemoryStore", "get_memory_store", "reset_memory_store",
]

from agent.evaluator import (
    Evaluator,
    EvaluationReport,
    EpisodeEvaluation,
    DriftAlert,
    DriftSeverity,
    build_evaluator,
    synthetic_return_provider,
    nifty_benchmark_return_provider,
)
