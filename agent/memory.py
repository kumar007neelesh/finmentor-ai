"""
agent/memory.py — FinMentor AI Episodic Memory
===============================================
Persistent memory store for the agent.

Every completed FinancialPlan and StateTransition is stored as an Episode.
The memory serves two purposes:

  1. CONTINUITY — The Planner can retrieve past episodes so it doesn't
     repeat advice the user has already received, and can track progress
     over time ("Your health score improved from 58 → 72 since last month").

  2. EVALUATION DATA — The Evaluator (File 10) reads stored episodes to
     compare predicted_return vs actual_return over time.

Backends:
  - JSONMemoryStore   (default, file-per-session, zero dependencies)
  - RedisMemoryStore  (production, horizontal scaling, TTL support)
  - InMemoryStore     (unit tests, no disk I/O)

The public interface is identical across all backends — swap by changing
config.agent.memory_backend.

Architecture position:
    Executor ──► memory.store(plan, state, transition)
    Planner  ──► memory.retrieve_recent(user_id, n=5)  → context injection
    Evaluator ──► memory.retrieve_for_evaluation(user_id) → scored episodes

Usage:
    from agent.memory import get_memory_store, Episode

    memory = get_memory_store()

    # Store after every turn
    await memory.store(plan=plan, state=state, transition=transition, session_id=sid)

    # Retrieve for planner context
    episodes = await memory.retrieve_recent(user_id="u_001", n=5)
    for ep in episodes:
        print(ep.summary())

    # Retrieve for evaluation
    pending = await memory.retrieve_for_evaluation(user_id="u_001")
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import settings, MEMORY_DIR
from environment.state import UserFinancialState, StateTransition, LifeEvent
from agent.planner import FinancialPlan
from logger import get_logger, metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Episode — atomic unit stored in memory
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """
    A single memory record: one conversation turn with its plan,
    optional state transition, and evaluation fields.

    Immutable once created — append-only memory pattern.
    """

    # Identity
    episode_id: str = field(default_factory=lambda: f"ep_{uuid4().hex[:10]}")
    session_id: str = ""
    user_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # Query context
    query: str = ""
    intent: str = "unknown"

    # Plan snapshot (key fields only — not full trace to save space)
    plan_id: str = ""
    final_answer: str = ""
    tools_used: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    # Key numbers for quick lookup (avoids parsing full plan)
    health_score: Optional[float] = None
    recommended_sip_inr: Optional[float] = None
    predicted_return_pct: Optional[float] = None
    fire_corpus_inr: Optional[float] = None
    tax_saving_inr: Optional[float] = None

    # State snapshot (serialised financial summary for comparison)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    state_fingerprint: str = ""

    # Life event
    life_event: str = LifeEvent.NONE.value
    life_event_amount: float = 0.0

    # Evaluation fields (filled in later by Evaluator)
    actual_return_pct: Optional[float] = None
    evaluation_score: Optional[float] = None
    evaluated_at: Optional[str] = None
    evaluation_notes: str = ""

    # Performance
    total_latency_ms: float = 0.0
    used_fallback: bool = False

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_plan(
        cls,
        plan: FinancialPlan,
        state: UserFinancialState,
        transition: Optional[StateTransition] = None,
    ) -> "Episode":
        """Build an Episode from a completed FinancialPlan."""
        life_event_value = LifeEvent.NONE.value
        life_event_amount = 0.0
        if transition:
            life_event_value  = transition.life_event.value if hasattr(transition.life_event, "value") else str(transition.life_event)
            life_event_amount = transition.event_amount

        return cls(
            session_id=plan.session_id,
            user_id=plan.user_id,
            query=plan.query,
            intent=plan.intent,
            plan_id=plan.plan_id,
            final_answer=plan.final_answer[:1000],  # Truncate to save space
            tools_used=plan.tools_used,
            recommended_actions=plan.recommended_actions,
            health_score=plan.health_score,
            recommended_sip_inr=plan.recommended_sip_inr,
            predicted_return_pct=plan.predicted_return_pct,
            fire_corpus_inr=plan.fire_corpus_inr,
            tax_saving_inr=plan.tax_saving_inr,
            state_snapshot=state.financial_summary(),
            state_fingerprint=state.fingerprint(),
            life_event=life_event_value,
            life_event_amount=life_event_amount,
            total_latency_ms=plan.total_latency_ms,
            used_fallback=plan.used_fallback,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def age_days(self) -> float:
        """How many days ago this episode was created."""
        created = datetime.fromisoformat(self.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - created).total_seconds() / 86400

    @property
    def is_evaluated(self) -> bool:
        return self.evaluated_at is not None

    @property
    def needs_evaluation(self) -> bool:
        """
        True if the episode has a predicted return but no actual return yet,
        AND is old enough (>30 days) to have observable outcomes.
        """
        return (
            self.predicted_return_pct is not None
            and self.actual_return_pct is None
            and self.age_days >= 30
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        # Filter to only known fields to be forward-compatible
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def summary(self) -> str:
        """One-line summary for logging and LLM context injection."""
        score_str  = f"  score={self.health_score:.0f}" if self.health_score else ""
        sip_str    = f"  sip=₹{self.recommended_sip_inr:,.0f}" if self.recommended_sip_inr else ""
        ret_str    = f"  ret={self.predicted_return_pct:.1f}%" if self.predicted_return_pct else ""
        event_str  = f"  event={self.life_event}" if self.life_event != "none" else ""
        return (
            f"[{self.episode_id}] {self.created_at[:10]}  "
            f"intent={self.intent}{score_str}{sip_str}{ret_str}{event_str}  "
            f"latency={self.total_latency_ms:.0f}ms"
        )

    def context_for_llm(self) -> str:
        """
        Compact text representation of this episode for LLM context injection.
        Injected into the Planner's system prompt when retrieving recent history.
        """
        lines = [
            f"Date: {self.created_at[:10]}",
            f"Query: {self.query[:100]}",
            f"Intent: {self.intent}",
        ]
        if self.health_score is not None:
            lines.append(f"Health Score: {self.health_score:.0f}/100")
        if self.recommended_sip_inr is not None:
            lines.append(f"Recommended SIP: ₹{self.recommended_sip_inr:,.0f}/month")
        if self.fire_corpus_inr is not None:
            lines.append(f"FIRE Corpus Needed: ₹{self.fire_corpus_inr:,.0f}")
        if self.tax_saving_inr is not None:
            lines.append(f"Tax Saving: ₹{self.tax_saving_inr:,.0f}")
        if self.life_event != "none":
            lines.append(f"Life Event: {self.life_event} (₹{self.life_event_amount:,.0f})")
        if self.actual_return_pct is not None:
            lines.append(f"Actual Return (measured): {self.actual_return_pct:.1f}%")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class BaseMemoryStore(ABC):
    """
    Interface all memory backends implement.
    The Executor and Evaluator call these methods — never backend-specific ones.
    """

    @abstractmethod
    async def store(
        self,
        plan: FinancialPlan,
        state: UserFinancialState,
        transition: Optional[StateTransition],
        session_id: str,
    ) -> Episode:
        """Persist a completed plan turn as an Episode. Returns the Episode."""
        ...

    @abstractmethod
    async def retrieve_recent(
        self,
        user_id: str,
        n: int = 5,
        intent_filter: Optional[str] = None,
    ) -> List[Episode]:
        """Return the n most recent episodes for a user (newest first)."""
        ...

    @abstractmethod
    async def retrieve_for_evaluation(
        self,
        user_id: str,
    ) -> List[Episode]:
        """
        Return episodes that have a predicted_return but no actual_return
        and are ≥30 days old (ready for Evaluator to score).
        """
        ...

    @abstractmethod
    async def update_evaluation(
        self,
        episode_id: str,
        actual_return_pct: float,
        evaluation_score: float,
        notes: str = "",
    ) -> bool:
        """Write Evaluator results back into the episode record."""
        ...

    @abstractmethod
    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Fetch a single episode by ID."""
        ...

    @abstractmethod
    async def count(self, user_id: str) -> int:
        """Total episodes stored for a user."""
        ...

    @abstractmethod
    async def clear_user(self, user_id: str) -> int:
        """Delete all episodes for a user. Returns count deleted."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def get_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Compare the user's earliest vs most recent episode to show progress.
        Returns a dict suitable for LLM context injection.
        """
        episodes = await self.retrieve_recent(user_id, n=settings.agent.memory_max_episodes)
        if not episodes:
            return {}

        latest  = episodes[0]
        earliest = episodes[-1]

        return {
            "total_episodes": len(episodes),
            "first_date":     earliest.created_at[:10],
            "latest_date":    latest.created_at[:10],
            "health_score_change": (
                round((latest.health_score or 0) - (earliest.health_score or 0), 1)
                if latest.health_score and earliest.health_score else None
            ),
            "sip_change_inr": (
                round((latest.recommended_sip_inr or 0) - (earliest.recommended_sip_inr or 0), 0)
                if latest.recommended_sip_inr and earliest.recommended_sip_inr else None
            ),
            "life_events_seen": list({
                ep.life_event for ep in episodes if ep.life_event != "none"
            }),
            "intents_seen": list({ep.intent for ep in episodes}),
        }

    async def context_for_planner(self, user_id: str, n: int = 3) -> str:
        """
        Build a compact multi-episode context string for LLM injection.
        Called by the Planner before each planning session.
        """
        episodes = await self.retrieve_recent(user_id, n=n)
        if not episodes:
            return "No prior financial planning sessions found for this user."

        lines = [f"Prior financial planning history ({len(episodes)} recent sessions):"]
        for ep in episodes:
            lines.append(f"\n--- Session {ep.created_at[:10]} (intent: {ep.intent}) ---")
            lines.append(ep.context_for_llm())

        progress = await self.get_progress_summary(user_id)
        if progress.get("health_score_change") is not None:
            delta = progress["health_score_change"]
            direction = "improved" if delta > 0 else "declined"
            lines.append(
                f"\nTrend: Health score {direction} by {abs(delta):.0f} points "
                f"since {progress['first_date']}."
            )

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Backend 1: JSON file store (default, zero dependencies)
# ══════════════════════════════════════════════════════════════════════════════

class JSONMemoryStore(BaseMemoryStore):
    """
    File-per-user JSON memory store.

    Layout:
        memory_store/
          u_001.json    ← list of Episode dicts, newest first
          u_002.json
          ...

    Thread-safe via per-user file locks.
    Suitable for single-instance deployments and development.
    """

    def __init__(self, base_dir: Path = MEMORY_DIR) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, Lock] = {}
        self._locks_lock = Lock()

    # ── Lock management ───────────────────────────────────────────────────────

    def _get_lock(self, user_id: str) -> Lock:
        with self._locks_lock:
            if user_id not in self._locks:
                self._locks[user_id] = Lock()
            return self._locks[user_id]

    # ── File path ─────────────────────────────────────────────────────────────

    def _user_file(self, user_id: str) -> Path:
        # Sanitise user_id to prevent path traversal
        safe_uid = "".join(c for c in user_id if c.isalnum() or c in "_-")[:64]
        return self._base_dir / f"{safe_uid}.json"

    # ── Low-level I/O ─────────────────────────────────────────────────────────

    def _read_episodes(self, user_id: str) -> List[Dict[str, Any]]:
        """Read raw episode dicts from disk. Returns [] on missing file."""
        path = self._user_file(user_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "Memory file corrupted, starting fresh",
                user_id=user_id,
                error=str(exc),
            )
            return []

    def _write_episodes(self, user_id: str, episodes: List[Dict[str, Any]]) -> None:
        """Write raw episode dicts to disk. Atomic write via temp file."""
        path = self._user_file(user_id)
        tmp  = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(episodes, f, ensure_ascii=False, indent=2, default=str)
            tmp.replace(path)   # Atomic on POSIX; best-effort on Windows
        except OSError as exc:
            log.error("Memory write failed", user_id=user_id, error=str(exc))
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    # ── Prune to max_episodes ─────────────────────────────────────────────────

    def _prune(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the most recent max_episodes (newest first)."""
        limit = settings.agent.memory_max_episodes
        return episodes[:limit]

    # ── BaseMemoryStore implementation ────────────────────────────────────────

    async def store(
        self,
        plan: FinancialPlan,
        state: UserFinancialState,
        transition: Optional[StateTransition],
        session_id: str,
    ) -> Episode:
        """Persist plan as Episode. Runs file I/O in thread pool."""
        episode = Episode.from_plan(plan, state, transition)

        def _write() -> None:
            lock = self._get_lock(plan.user_id)
            with lock:
                episodes = self._read_episodes(plan.user_id)
                episodes.insert(0, episode.to_dict())   # Newest first
                episodes = self._prune(episodes)
                self._write_episodes(plan.user_id, episodes)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)

        metrics.increment("memory.episodes_stored")
        log.info(
            "Episode stored",
            episode_id=episode.episode_id,
            user_id=plan.user_id,
            intent=plan.intent,
            session_id=session_id,
        )
        return episode

    async def retrieve_recent(
        self,
        user_id: str,
        n: int = 5,
        intent_filter: Optional[str] = None,
    ) -> List[Episode]:
        """Return the n most recent episodes (newest first)."""
        def _read() -> List[Episode]:
            raw = self._read_episodes(user_id)
            episodes = [Episode.from_dict(d) for d in raw]
            if intent_filter:
                episodes = [e for e in episodes if e.intent == intent_filter]
            return episodes[:n]

        loop = asyncio.get_event_loop()
        episodes = await loop.run_in_executor(None, _read)
        metrics.increment("memory.retrieve_recent_calls")
        return episodes

    async def retrieve_for_evaluation(self, user_id: str) -> List[Episode]:
        """Return episodes that need Evaluator scoring."""
        def _read() -> List[Episode]:
            raw = self._read_episodes(user_id)
            episodes = [Episode.from_dict(d) for d in raw]
            return [e for e in episodes if e.needs_evaluation]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read)

    async def update_evaluation(
        self,
        episode_id: str,
        actual_return_pct: float,
        evaluation_score: float,
        notes: str = "",
    ) -> bool:
        """
        Update evaluation fields in the stored episode.
        Must scan all user files to find the episode — in production,
        use Redis with episode_id as key.
        """
        def _update() -> bool:
            for json_file in self._base_dir.glob("*.json"):
                user_id = json_file.stem
                lock = self._get_lock(user_id)
                with lock:
                    raw = self._read_episodes(user_id)
                    updated = False
                    for ep_dict in raw:
                        if ep_dict.get("episode_id") == episode_id:
                            ep_dict["actual_return_pct"]  = actual_return_pct
                            ep_dict["evaluation_score"]   = evaluation_score
                            ep_dict["evaluation_notes"]   = notes
                            ep_dict["evaluated_at"]       = datetime.now(tz=timezone.utc).isoformat()
                            updated = True
                            break
                    if updated:
                        self._write_episodes(user_id, raw)
                        return True
            return False

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _update)
        if result:
            metrics.increment("memory.evaluations_written")
            log.info("Evaluation written", episode_id=episode_id)
        else:
            log.warning("Episode not found for evaluation update", episode_id=episode_id)
        return result

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Fetch a single episode by ID across all user files."""
        def _find() -> Optional[Episode]:
            for json_file in self._base_dir.glob("*.json"):
                raw = self._read_episodes(json_file.stem)
                for ep_dict in raw:
                    if ep_dict.get("episode_id") == episode_id:
                        return Episode.from_dict(ep_dict)
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _find)

    async def count(self, user_id: str) -> int:
        def _count() -> int:
            return len(self._read_episodes(user_id))
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _count)

    async def clear_user(self, user_id: str) -> int:
        def _clear() -> int:
            lock = self._get_lock(user_id)
            with lock:
                raw = self._read_episodes(user_id)
                count = len(raw)
                path = self._user_file(user_id)
                if path.exists():
                    path.unlink()
            return count
        loop = asyncio.get_event_loop()
        deleted = await loop.run_in_executor(None, _clear)
        log.info("User memory cleared", user_id=user_id, deleted=deleted)
        return deleted


# ══════════════════════════════════════════════════════════════════════════════
# Backend 2: In-memory (unit tests, zero disk I/O)
# ══════════════════════════════════════════════════════════════════════════════

class InMemoryStore(BaseMemoryStore):
    """
    Pure in-process dict-backed memory.
    Used in unit tests and CI pipelines where disk I/O is undesirable.
    State is lost on process exit.
    """

    def __init__(self) -> None:
        # user_id → list of Episode (newest first)
        self._store: Dict[str, List[Episode]] = {}
        self._lock = Lock()

    async def store(
        self,
        plan: FinancialPlan,
        state: UserFinancialState,
        transition: Optional[StateTransition],
        session_id: str,
    ) -> Episode:
        episode = Episode.from_plan(plan, state, transition)
        with self._lock:
            if plan.user_id not in self._store:
                self._store[plan.user_id] = []
            self._store[plan.user_id].insert(0, episode)
            # Prune
            limit = settings.agent.memory_max_episodes
            self._store[plan.user_id] = self._store[plan.user_id][:limit]
        metrics.increment("memory.episodes_stored")
        return episode

    async def retrieve_recent(
        self,
        user_id: str,
        n: int = 5,
        intent_filter: Optional[str] = None,
    ) -> List[Episode]:
        with self._lock:
            episodes = list(self._store.get(user_id, []))
        if intent_filter:
            episodes = [e for e in episodes if e.intent == intent_filter]
        return episodes[:n]

    async def retrieve_for_evaluation(self, user_id: str) -> List[Episode]:
        with self._lock:
            episodes = list(self._store.get(user_id, []))
        return [e for e in episodes if e.needs_evaluation]

    async def update_evaluation(
        self,
        episode_id: str,
        actual_return_pct: float,
        evaluation_score: float,
        notes: str = "",
    ) -> bool:
        with self._lock:
            for episodes in self._store.values():
                for ep in episodes:
                    if ep.episode_id == episode_id:
                        ep.actual_return_pct = actual_return_pct
                        ep.evaluation_score  = evaluation_score
                        ep.evaluation_notes  = notes
                        ep.evaluated_at      = datetime.now(tz=timezone.utc).isoformat()
                        metrics.increment("memory.evaluations_written")
                        return True
        return False

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        with self._lock:
            for episodes in self._store.values():
                for ep in episodes:
                    if ep.episode_id == episode_id:
                        return ep
        return None

    async def count(self, user_id: str) -> int:
        with self._lock:
            return len(self._store.get(user_id, []))

    async def clear_user(self, user_id: str) -> int:
        with self._lock:
            count = len(self._store.get(user_id, []))
            self._store.pop(user_id, None)
        return count

    def all_episodes(self) -> List[Episode]:
        """Test helper — return every stored episode across all users."""
        with self._lock:
            return [ep for eps in self._store.values() for ep in eps]


# ══════════════════════════════════════════════════════════════════════════════
# Backend 3: Redis (production stub — implement when Redis is available)
# ══════════════════════════════════════════════════════════════════════════════

class RedisMemoryStore(BaseMemoryStore):
    """
    Redis-backed episodic memory for production horizontal scaling.

    Key design:
      - Sorted set per user:  `fm:episodes:{user_id}`
                                score = Unix timestamp (for range queries)
                                member = episode_id
      - Hash per episode:     `fm:ep:{episode_id}` → field/value pairs
      - TTL: 365 days per episode key (auto-expiry)

    Install: pip install redis[asyncio]

    Set: FINMENTOR__AGENT__MEMORY_BACKEND=redis
         REDIS_URL=redis://localhost:6379/0
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
        self._ttl_seconds = 365 * 24 * 3600   # 1 year

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self._url, decode_responses=True)
                await self._client.ping()
                log.info("Redis memory store connected", url=self._url)
            except Exception as exc:
                log.error("Redis connection failed", error=str(exc))
                raise RuntimeError(f"Redis unavailable: {exc}") from exc
        return self._client

    async def store(
        self,
        plan: FinancialPlan,
        state: UserFinancialState,
        transition: Optional[StateTransition],
        session_id: str,
    ) -> Episode:
        episode = Episode.from_plan(plan, state, transition)
        r = await self._get_client()
        ts = time.time()
        user_key = f"fm:episodes:{plan.user_id}"
        ep_key   = f"fm:ep:{episode.episode_id}"

        pipe = r.pipeline()
        pipe.zadd(user_key, {episode.episode_id: ts})
        pipe.hset(ep_key, mapping={
            k: json.dumps(v, default=str) for k, v in episode.to_dict().items()
        })
        pipe.expire(ep_key, self._ttl_seconds)
        # Trim to max_episodes
        limit = settings.agent.memory_max_episodes
        pipe.zremrangebyrank(user_key, 0, -(limit + 1))
        await pipe.execute()

        metrics.increment("memory.episodes_stored")
        log.info("Episode stored (Redis)", episode_id=episode.episode_id)
        return episode

    async def retrieve_recent(
        self,
        user_id: str,
        n: int = 5,
        intent_filter: Optional[str] = None,
    ) -> List[Episode]:
        r = await self._get_client()
        user_key = f"fm:episodes:{user_id}"
        # Get n*3 IDs so we have room to filter by intent
        fetch_n  = n * 3 if intent_filter else n
        ep_ids = await r.zrevrange(user_key, 0, fetch_n - 1)

        episodes = []
        for ep_id in ep_ids:
            ep_data = await r.hgetall(f"fm:ep:{ep_id}")
            if not ep_data:
                continue
            decoded = {k: self._decode_field(v) for k, v in ep_data.items()}
            ep = Episode.from_dict(decoded)
            if intent_filter and ep.intent != intent_filter:
                continue
            episodes.append(ep)
            if len(episodes) >= n:
                break

        return episodes

    async def retrieve_for_evaluation(self, user_id: str) -> List[Episode]:
        episodes = await self.retrieve_recent(user_id, n=settings.agent.memory_max_episodes)
        return [e for e in episodes if e.needs_evaluation]

    async def update_evaluation(
        self,
        episode_id: str,
        actual_return_pct: float,
        evaluation_score: float,
        notes: str = "",
    ) -> bool:
        r = await self._get_client()
        ep_key = f"fm:ep:{episode_id}"
        if not await r.exists(ep_key):
            return False
        await r.hset(ep_key, mapping={
            "actual_return_pct": actual_return_pct,
            "evaluation_score":  evaluation_score,
            "evaluation_notes":  notes,
            "evaluated_at":      datetime.now(tz=timezone.utc).isoformat(),
        })
        metrics.increment("memory.evaluations_written")
        return True

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        r = await self._get_client()
        ep_data = await r.hgetall(f"fm:ep:{episode_id}")
        if not ep_data:
            return None
        decoded = {k: self._decode_field(v) for k, v in ep_data.items()}
        return Episode.from_dict(decoded)

    async def count(self, user_id: str) -> int:
        r = await self._get_client()
        return await r.zcard(f"fm:episodes:{user_id}")

    async def clear_user(self, user_id: str) -> int:
        r = await self._get_client()
        user_key = f"fm:episodes:{user_id}"
        ep_ids = await r.zrange(user_key, 0, -1)
        pipe = r.pipeline()
        for ep_id in ep_ids:
            pipe.delete(f"fm:ep:{ep_id}")
        pipe.delete(user_key)
        await pipe.execute()
        return len(ep_ids)

    @staticmethod
    def _decode_field(value: str) -> Any:
        """Decode a JSON-serialised Redis hash field back to Python type."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value


# ══════════════════════════════════════════════════════════════════════════════
# Singleton factory
# ══════════════════════════════════════════════════════════════════════════════

_memory_instance: Optional[BaseMemoryStore] = None
_memory_lock = asyncio.Lock()


def get_memory_store() -> BaseMemoryStore:
    """
    Return the configured memory store singleton.

    Backend selected by config.agent.memory_backend:
      "json"    → JSONMemoryStore  (default)
      "memory"  → InMemoryStore   (tests)
      "redis"   → RedisMemoryStore (production)

    Call once at startup; reuse the singleton everywhere.
    """
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance

    backend = settings.agent.memory_backend.lower()

    if backend == "json":
        _memory_instance = JSONMemoryStore()
        log.info("Memory store: JSON", path=str(MEMORY_DIR))

    elif backend in ("memory", "inmemory", "in_memory"):
        _memory_instance = InMemoryStore()
        log.info("Memory store: InMemory (volatile)")

    elif backend == "redis":
        _memory_instance = RedisMemoryStore()
        log.info("Memory store: Redis")

    else:
        log.warning(
            f"Unknown memory backend '{backend}', falling back to JSON",
        )
        _memory_instance = JSONMemoryStore()

    return _memory_instance


def reset_memory_store(store: Optional[BaseMemoryStore] = None) -> None:
    """
    Replace the global singleton.
    Used in tests to inject a fresh InMemoryStore.

    Usage:
        from agent.memory import reset_memory_store, InMemoryStore
        reset_memory_store(InMemoryStore())
    """
    global _memory_instance
    _memory_instance = store


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

    def _make_state(user_id: str = "u_test_001") -> UserFinancialState:
        return UserFinancialState(
            user_id=user_id,
            age=33,
            monthly_income=130_000,
            monthly_expense=60_000,
            risk_profile=RiskProfile.MODERATE,
            risk_score=6.0,
            investment_horizon_years=24,
            emergency_fund_amount=250_000,
            existing_monthly_sip=14_000,
            section_80c_used=90_000,
            insurance=InsuranceCoverage(
                has_term_insurance=True, term_cover_amount=12_000_000,
                has_health_insurance=True, health_cover_amount=500_000,
            ),
            debts=DebtProfile(home_loan_emi=22_000),
            portfolio=InvestmentPortfolio(
                equity_mf=800_000, debt_mf=250_000,
                ppf=120_000, epf=300_000,
            ),
            goals=[
                FinancialGoal(
                    name="Retirement", target_amount=60_000_000,
                    target_year=2057, priority=1, existing_corpus=1_470_000,
                ),
            ],
        )

    def _make_plan(
        user_id: str,
        session_id: str,
        intent: str = "health_score",
        health_score: float = 68.5,
        sip: float = 28_000,
    ) -> FinancialPlan:
        return FinancialPlan(
            session_id=session_id,
            user_id=user_id,
            query=f"Test query for intent={intent}",
            intent=intent,
            final_answer="Here is your financial plan...",
            tools_used=["health_score", "sip_calculator"],
            health_score=health_score,
            recommended_sip_inr=sip,
            predicted_return_pct=12.5,
            total_latency_ms=450.0,
        )

    async def run_tests() -> None:
        print("=== Memory Store Self-Test ===\n")

        # ── Use InMemoryStore for speed ───────────────────────────────────────
        reset_memory_store(InMemoryStore())
        memory = get_memory_store()

        state = _make_state("u_mem_001")

        # ── Test 1: Store episodes ────────────────────────────────────────────
        print("── Test 1: Store Episodes ──")
        plans = [
            _make_plan("u_mem_001", "sess_1", "health_score",    65.0, 25_000),
            _make_plan("u_mem_001", "sess_2", "sip_calculation", 68.0, 27_000),
            _make_plan("u_mem_001", "sess_3", "fire_planning",   72.0, 30_000),
        ]

        stored_episodes = []
        for plan in plans:
            ep = await memory.store(plan=plan, state=state, transition=None, session_id=plan.session_id)
            stored_episodes.append(ep)
            print(f"  Stored: {ep.summary()}")

        count = await memory.count("u_mem_001")
        print(f"  Total stored: {count}")
        assert count == 3
        print("  ✓ All episodes stored\n")

        # ── Test 2: Retrieve recent ───────────────────────────────────────────
        print("── Test 2: Retrieve Recent ──")
        recent = await memory.retrieve_recent("u_mem_001", n=2)
        print(f"  Retrieved {len(recent)} episodes (requested 2)")
        assert len(recent) == 2
        # Most recent first
        assert recent[0].health_score == 72.0
        assert recent[1].health_score == 68.0
        print(f"  Latest health score  : {recent[0].health_score}")
        print(f"  Previous health score: {recent[1].health_score}")
        print("  ✓ Newest-first ordering correct\n")

        # ── Test 3: Intent filter ─────────────────────────────────────────────
        print("── Test 3: Intent Filter ──")
        fire_eps = await memory.retrieve_recent("u_mem_001", n=5, intent_filter="fire_planning")
        print(f"  Episodes with intent=fire_planning: {len(fire_eps)}")
        assert len(fire_eps) == 1
        assert fire_eps[0].intent == "fire_planning"
        print("  ✓ Intent filter correct\n")

        # ── Test 4: Get by ID ─────────────────────────────────────────────────
        print("── Test 4: Get Episode by ID ──")
        ep_id = stored_episodes[0].episode_id
        fetched = await memory.get_episode(ep_id)
        assert fetched is not None
        assert fetched.episode_id == ep_id
        print(f"  Fetched episode: {fetched.episode_id}")
        print("  ✓ Get by ID correct\n")

        # ── Test 5: Update evaluation ─────────────────────────────────────────
        print("── Test 5: Update Evaluation ──")
        ok = await memory.update_evaluation(
            episode_id=ep_id,
            actual_return_pct=11.8,
            evaluation_score=0.94,
            notes="Portfolio performed close to prediction",
        )
        assert ok
        updated_ep = await memory.get_episode(ep_id)
        assert updated_ep.actual_return_pct == 11.8
        assert updated_ep.evaluation_score  == 0.94
        assert updated_ep.is_evaluated
        print(f"  Actual return: {updated_ep.actual_return_pct}%")
        print(f"  Eval score   : {updated_ep.evaluation_score}")
        print("  ✓ Evaluation written\n")

        # ── Test 6: Needs evaluation (age check) ──────────────────────────────
        print("── Test 6: Needs Evaluation ──")
        # Fresh episodes are <30 days old — should NOT need evaluation
        pending = await memory.retrieve_for_evaluation("u_mem_001")
        print(f"  Episodes needing evaluation (fresh): {len(pending)}")
        # All have predicted_return, but age_days < 30 → needs_evaluation = False
        assert len(pending) == 0, f"Expected 0, got {len(pending)}"
        print("  ✓ Fresh episodes correctly excluded from evaluation\n")

        # ── Test 7: Progress summary ──────────────────────────────────────────
        print("── Test 7: Progress Summary ──")
        progress = await memory.get_progress_summary("u_mem_001")
        print(f"  {_json.dumps(progress, indent=2, default=str)}")
        assert progress["total_episodes"] == 3
        assert progress["health_score_change"] == pytest_approx(7.0)
        print("  ✓ Progress summary: health score improved by 7 points\n")

        # ── Test 8: Context for planner ───────────────────────────────────────
        print("── Test 8: Context for Planner ──")
        ctx = await memory.context_for_planner("u_mem_001", n=2)
        print(ctx)
        assert "Prior financial planning history" in ctx
        assert "72" in ctx   # Latest health score
        print("  ✓ LLM-ready context generated\n")

        # ── Test 9: Episode serialisation ─────────────────────────────────────
        print("── Test 9: Episode Serialisation ──")
        ep = stored_episodes[1]
        ep_dict = ep.to_dict()
        ep_back = Episode.from_dict(ep_dict)
        assert ep_back.episode_id == ep.episode_id
        assert ep_back.health_score == ep.health_score
        ep_json = _json.dumps(ep_dict, default=str)
        print(f"  Episode JSON: {len(ep_json)} chars")
        print("  ✓ Round-trip serialisation correct\n")

        # ── Test 10: Clear user ───────────────────────────────────────────────
        print("── Test 10: Clear User ──")
        deleted = await memory.clear_user("u_mem_001")
        remaining = await memory.count("u_mem_001")
        print(f"  Deleted: {deleted}  Remaining: {remaining}")
        assert deleted == 3
        assert remaining == 0
        print("  ✓ User memory cleared\n")

        # ── Test 11: JSON backend ─────────────────────────────────────────────
        print("── Test 11: JSONMemoryStore ──")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            json_store = JSONMemoryStore(base_dir=Path(tmpdir))
            ep1 = await json_store.store(
                plan=_make_plan("u_json_001", "sess_json_1"),
                state=state, transition=None, session_id="sess_json_1",
            )
            print(f"  Stored: {ep1.episode_id}")
            retrieved = await json_store.retrieve_recent("u_json_001")
            assert len(retrieved) == 1
            assert retrieved[0].episode_id == ep1.episode_id
            print(f"  Retrieved: {retrieved[0].episode_id}")
            print("  ✓ JSON backend round-trip correct\n")

        # ── Metrics snapshot ──────────────────────────────────────────────────
        print("── Metrics Snapshot ──")
        snap = metrics.snapshot()
        for c in snap["counters"]:
            if "memory" in c["name"]:
                print(f"  {c['name']:<45} {int(c['value'])}")

    # Helper for approx comparison in tests
    def pytest_approx(val, rel=0.01):
        class Approx:
            def __eq__(self, other):
                return abs(other - val) <= abs(val) * rel
        return Approx()

    asyncio.run(run_tests())
