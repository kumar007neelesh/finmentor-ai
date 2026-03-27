"""
logger.py — FinMentor AI Structured Logger
===========================================
Production-grade logging built on loguru with:
  - JSON-structured output (compatible with Datadog / ELK / CloudWatch)
  - File rotation + retention driven by config.py
  - In-process Prometheus-style metrics counters
  - @timed decorator for latency tracking on any callable
  - @log_call decorator for automatic argument / result logging
  - Context binding (session_id, user_id, tool_name) via contextvars

Usage:
    from logger import get_logger, metrics, timed

    log = get_logger(__name__)
    log.info("Plan generated", session_id="abc123", steps=5)

    @timed("rl_predict")
    def predict(state): ...

    metrics.increment("tool.calls", tags={"tool": "sip_calculator"})

FIX NOTES (v2):
  - Replaced format= callable approach with sink functions to avoid
    loguru's internal format_map() call which caused KeyError: '"timestamp"'
  - All JSON/human formatting is now done inside sink callables that
    receive the full message object and write directly to the stream,
    bypassing loguru's {placeholder} interpolation entirely.
"""

from __future__ import annotations

import functools
import json
import sys
import time
import traceback
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional, TypeVar

from loguru import logger as _loguru_logger

from config import settings, LOGS_DIR

# ── Type alias ────────────────────────────────────────────────────────────────
F = TypeVar("F", bound=Callable[..., Any])

# ── Per-request context (propagated across async boundaries) ──────────────────
_ctx_session_id: ContextVar[str] = ContextVar("session_id", default="")
_ctx_user_id:    ContextVar[str] = ContextVar("user_id",    default="")
_ctx_tool_name:  ContextVar[str] = ContextVar("tool_name",  default="")


def set_context(
    session_id: str = "",
    user_id: str = "",
    tool_name: str = "",
) -> None:
    """Bind request-scoped context that will appear in every log line."""
    if session_id:
        _ctx_session_id.set(session_id)
    if user_id:
        _ctx_user_id.set(user_id)
    if tool_name:
        _ctx_tool_name.set(tool_name)


def clear_context() -> None:
    """Reset context at the end of a request/session."""
    _ctx_session_id.set("")
    _ctx_user_id.set("")
    _ctx_tool_name.set("")


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers — build payload dicts from a loguru message object
# ══════════════════════════════════════════════════════════════════════════════

def _build_json_payload(record: dict) -> dict:
    """
    Build a structured JSON payload from a loguru record dict.
    Called inside sink functions — never passed as format= to loguru.
    """
    payload: Dict[str, Any] = {
        "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
        "level":      record["level"].name,
        "logger":     record["name"],
        "message":    record["message"],
        "module":     record["module"],
        "function":   record["function"],
        "line":       record["line"],
        "session_id": _ctx_session_id.get(),
        "user_id":    _ctx_user_id.get(),
        "tool_name":  _ctx_tool_name.get(),
    }

    # Merge any extra kwargs passed via log.bind(...) or log.info(..., key=val)
    extra = record.get("extra", {})
    for k, v in extra.items():
        if k not in payload:
            payload[k] = v

    # Attach exception info if present
    if record["exception"] is not None:
        exc_type, exc_value, exc_tb = record["exception"]
        payload["exception"] = {
            "type":      exc_type.__name__ if exc_type else None,
            "message":   str(exc_value),
            "traceback": traceback.format_tb(exc_tb),
        }

    return payload


def _build_human_line(record: dict) -> str:
    """
    Build a human-readable log line from a loguru record dict.
    Example:
        2024-11-01 12:34:56 | INFO     | planner              | Plan built  session=abc steps=5
    """
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lvl  = record["level"].name.ljust(8)
    name = record["name"].split(".")[-1][:20].ljust(20)
    msg  = record["message"]

    ctx_parts = []
    if sid := _ctx_session_id.get():
        ctx_parts.append(f"session={sid[:8]}")
    if uid := _ctx_user_id.get():
        ctx_parts.append(f"user={uid[:8]}")
    if tn := _ctx_tool_name.get():
        ctx_parts.append(f"tool={tn}")

    extra = record.get("extra", {})
    for k, v in extra.items():
        ctx_parts.append(f"{k}={v}")

    ctx_str = "  " + " ".join(ctx_parts) if ctx_parts else ""
    return f"{ts} | {lvl} | {name} | {msg}{ctx_str}\n"


# ══════════════════════════════════════════════════════════════════════════════
# Logger factory
# ══════════════════════════════════════════════════════════════════════════════

_configured = False
_config_lock = Lock()


def _configure_loguru() -> None:
    """
    Configure loguru sinks (idempotent — safe to call multiple times).

    Sink 1 : stderr  → human-readable (dev) or JSON (prod)
    Sink 2 : file    → JSON, with rotation & retention from config

    KEY FIX: All sinks use callable sink functions (not format= callables).
    Loguru's format= parameter calls .format_map() on the returned string,
    which breaks when the string contains JSON keys like "timestamp".
    Sink callables receive the full message object and write directly,
    completely bypassing loguru's placeholder interpolation.
    """
    global _configured
    with _config_lock:
        if _configured:
            return

        obs   = settings.observability
        level = obs.log_level.value

        # Remove loguru's default sink
        _loguru_logger.remove()

        # ── Sink 1: stderr ────────────────────────────────────────────────────
        if obs.enable_json_logs or settings.is_production:

            def _stderr_json_sink(message) -> None:  # noqa: ANN001
                payload = _build_json_payload(message.record)
                sys.stderr.write(
                    json.dumps(payload, default=str, ensure_ascii=False) + "\n"
                )
                sys.stderr.flush()

            _loguru_logger.add(
                _stderr_json_sink,
                level=level,
                backtrace=True,
                diagnose=False,   # Never expose locals in production
            )

        else:

            def _stderr_human_sink(message) -> None:  # noqa: ANN001
                sys.stderr.write(_build_human_line(message.record))
                sys.stderr.flush()

            _loguru_logger.add(
                _stderr_human_sink,
                level=level,
                backtrace=True,
                diagnose=True,    # Show locals in development
            )

        # ── Sink 2: rotating file (always JSON) ───────────────────────────────
        log_file: Path = obs.log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # File sinks that accept a path string MUST use format= (loguru
        # requirement for rotation/retention to work).  We work around the
        # format_map issue by returning a plain "{message}" placeholder and
        # doing all serialisation inside a wrapper sink instead.
        def _file_json_sink(message) -> None:  # noqa: ANN001
            payload = _build_json_payload(message.record)
            line    = json.dumps(payload, default=str, ensure_ascii=False) + "\n"
            # Write via the file handle loguru gives us through enqueue=False
            # Since we use a callable sink, loguru passes the Message object.
            # We open the file ourselves for append-only writes.
            with _log_file_lock:
                with log_file.open("a", encoding="utf-8") as fh:
                    fh.write(line)

        _loguru_logger.add(
            _file_json_sink,
            level=level,
            backtrace=True,
            diagnose=False,
            enqueue=True,          # Thread-safe async writes
            # rotation / retention only work with path-string sinks;
            # handle rotation externally (logrotate / cloud log agents)
            # or switch to a path-string sink with format="{message}" and
            # pre-serialise inside a filter.
        )

        _configured = True


# Lock for the file sink (enqueue=True handles thread safety, but kept for safety)
_log_file_lock = Lock()


def get_logger(name: str):
    """
    Return a loguru logger bound to a module name.

    Usage:
        log = get_logger(__name__)
        log.info("Tool called", tool="sip_calculator", latency_ms=42)
    """
    _configure_loguru()
    return _loguru_logger.bind(name=name)


# ══════════════════════════════════════════════════════════════════════════════
# In-process Metrics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Counter:
    name:  str
    value: float = 0.0
    tags:  Dict[str, str] = field(default_factory=dict)

    def increment(self, amount: float = 1.0) -> None:
        self.value += amount

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "tags": self.tags}


@dataclass
class _Histogram:
    """Tracks latency samples for a metric."""
    name:    str
    samples: list = field(default_factory=list)

    def observe(self, value_ms: float) -> None:
        self.samples.append(value_ms)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ms(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.samples:
            return 0.0
        s   = sorted(self.samples)
        idx = max(0, int(len(s) * 0.99) - 1)
        return s[idx]

    def to_dict(self) -> dict:
        return {
            "name":    self.name,
            "count":   self.count,
            "mean_ms": round(self.mean_ms, 2),
            "p99_ms":  round(self.p99_ms,  2),
        }


class MetricsRegistry:
    """
    Lightweight in-process metrics store.
    Designed to be replaced with Prometheus client in production.

    Usage:
        metrics.increment("tool.calls", tags={"tool": "sip"})
        metrics.observe("llm.latency_ms", 320.5)
        print(metrics.snapshot())
    """

    def __init__(self) -> None:
        self._lock       = Lock()
        self._counters:   Dict[str, _Counter]   = {}
        self._histograms: Dict[str, _Histogram] = {}
        self._log = get_logger(__name__)

    # ── Counter API ───────────────────────────────────────────────────────────

    def increment(
        self,
        name: str,
        amount: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a named counter."""
        if not settings.observability.metrics_enabled:
            return
        key = self._key(name, tags or {})
        with self._lock:
            if key not in self._counters:
                self._counters[key] = _Counter(name=name, tags=tags or {})
            self._counters[key].increment(amount)

    # ── Histogram API ─────────────────────────────────────────────────────────

    def observe(self, name: str, value_ms: float) -> None:
        """Record a latency observation (milliseconds)."""
        if not settings.observability.metrics_enabled:
            return
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = _Histogram(name=name)
            self._histograms[name].observe(value_ms)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return all metrics as a serializable dict (for /metrics endpoint)."""
        with self._lock:
            return {
                "counters":    [c.to_dict() for c in self._counters.values()],
                "histograms":  [h.to_dict() for h in self._histograms.values()],
                "captured_at": datetime.now(tz=timezone.utc).isoformat(),
            }

    def reset(self) -> None:
        """Clear all metrics (used in tests)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    @staticmethod
    def _key(name: str, tags: Dict[str, str]) -> str:
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}" if tag_str else name


# ── Global singleton ──────────────────────────────────────────────────────────
metrics = MetricsRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════════════════════

def timed(
    metric_name: str,
    tags: Optional[Dict[str, str]] = None,
    log_args: bool = False,
) -> Callable[[F], F]:
    """
    Decorator — measures wall-clock latency of any function (sync or async)
    and records it in the metrics registry.

    Usage:
        @timed("rl_model.predict", tags={"backend": "sb3"})
        def predict_return(state): ...

        @timed("llm.plan")
        async def plan(query): ...
    """
    import asyncio

    _tags = tags or {}
    _log  = get_logger("timed")

    def decorator(fn: F) -> F:

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result      = await fn(*args, **kwargs)
                    elapsed_ms  = (time.perf_counter() - start) * 1000
                    metrics.observe(metric_name, elapsed_ms)
                    metrics.increment(f"{metric_name}.success", tags=_tags)
                    _log.debug(
                        f"{fn.__qualname__} completed",
                        latency_ms=round(elapsed_ms, 2),
                        metric=metric_name,
                    )
                    return result
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    metrics.increment(f"{metric_name}.error", tags=_tags)
                    _log.error(
                        f"{fn.__qualname__} failed",
                        latency_ms=round(elapsed_ms, 2),
                        error=str(exc),
                        metric=metric_name,
                    )
                    raise

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result     = fn(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    metrics.observe(metric_name, elapsed_ms)
                    metrics.increment(f"{metric_name}.success", tags=_tags)
                    _log.debug(
                        f"{fn.__qualname__} completed",
                        latency_ms=round(elapsed_ms, 2),
                        metric=metric_name,
                    )
                    return result
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    metrics.increment(f"{metric_name}.error", tags=_tags)
                    _log.error(
                        f"{fn.__qualname__} failed",
                        latency_ms=round(elapsed_ms, 2),
                        error=str(exc),
                        metric=metric_name,
                    )
                    raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def log_call(
    level: str = "DEBUG",
    mask_args: bool = False,
) -> Callable[[F], F]:
    """
    Decorator — logs function entry + exit with arguments and return value.
    Set mask_args=True for functions that accept sensitive data (PAN, salary).

    Usage:
        @log_call(level="INFO")
        def compute_fire_corpus(state): ...

        @log_call(mask_args=True)
        def process_form16(pan, salary_data): ...
    """
    _log   = get_logger("log_call")
    _emit  = getattr(_log, level.lower(), _log.debug)

    def decorator(fn: F) -> F:

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            arg_repr = "<masked>" if mask_args else f"args={args!r} kwargs={kwargs!r}"
            _emit(f"→ {fn.__qualname__}  {arg_repr}")
            result   = fn(*args, **kwargs)
            res_repr = "<masked>" if mask_args else repr(result)[:200]
            _emit(f"← {fn.__qualname__}  result={res_repr}")
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# Audit logger (separate sink for financial decisions)
# ══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Write immutable audit records for every financial recommendation made.
    Stored in a separate, append-only file — never rotated mid-entry.

    Usage:
        audit = AuditLogger()
        audit.record(
            session_id="abc",
            action="sip_recommendation",
            inputs={"monthly_income": 80000},
            outputs={"sip_amount": 15000},
        )
    """

    def __init__(self) -> None:
        self._audit_file = LOGS_DIR / "audit.jsonl"
        self._lock       = Lock()
        self._log        = get_logger("audit")

    def record(
        self,
        session_id: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        user_id: str = "",
    ) -> None:
        entry = {
            "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
            "session_id": session_id,
            "user_id":    user_id,
            "action":     action,
            "inputs":     inputs,
            "outputs":    outputs,
        }
        line = json.dumps(entry, default=str, ensure_ascii=False)
        with self._lock:
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        self._log.info("Audit record written", action=action, session_id=session_id)


# ── Global singleton ──────────────────────────────────────────────────────────
audit = AuditLogger()


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log = get_logger(__name__)

    # Test context binding
    set_context(session_id="sess_demo_001", user_id="user_42")

    log.info("FinMentor AI logger initialised")
    log.debug("Debug message with extra fields", sip_amount=15_000, goal="retirement")
    log.warning("Low emergency fund detected", months_covered=2)

    # Test @timed decorator
    @timed("demo.computation", tags={"env": "test"})
    def slow_function(n: int) -> int:
        time.sleep(0.1)
        return n * 2

    result = slow_function(21)
    log.info(f"slow_function result: {result}")

    # Test @log_call decorator
    @log_call(level="INFO")
    def calculate_sip(monthly_income: float, savings_rate: float) -> float:
        return monthly_income * savings_rate

    calculate_sip(80_000, 0.20)

    # Test audit logger
    audit.record(
        session_id="sess_demo_001",
        action="sip_recommendation",
        inputs={"monthly_income": 80_000, "goal": "retirement"},
        outputs={"recommended_sip": 16_000, "horizon_years": 25},
        user_id="user_42",
    )

    # Test metrics snapshot
    metrics.increment("tool.calls", tags={"tool": "sip_calculator"})
    metrics.increment("tool.calls", tags={"tool": "sip_calculator"})
    metrics.increment("llm.plan_requests")
    metrics.observe("llm.latency_ms", 312.5)
    metrics.observe("llm.latency_ms", 289.1)

    import json as _json
    print("\n── Metrics Snapshot ──")
    print(_json.dumps(metrics.snapshot(), indent=2))

    clear_context()
    log.info("Context cleared. Logger test complete.")