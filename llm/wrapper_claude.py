"""
llm/wrapper.py — FinMentor AI LLM Wrapper
==========================================
Thin, production-grade wrapper around the Anthropic Claude API.

Responsibilities:
  - Send messages to Claude with the FinMentor system prompt
  - Inject registered tool schemas so Claude knows what tools exist
  - Parse Claude's response to extract: text reply + tool call decisions
  - Retry on transient API errors (rate limits, 5xx) with exponential backoff
  - Support streaming responses for low-latency UX
  - Enforce the LLM's role boundary: reasoning + tool selection ONLY,
    never numerical prediction

Architecture position:
    Planner ──► LLMWrapper.chat(messages, tool_schemas)
                     │
                     ▼
               Anthropic API  (claude-sonnet-4-20250514)
                     │
                     ▼
               LLMResponse
                 ├── text          (reasoning / final answer)
                 ├── tool_calls    (list of {tool_name, args})
                 └── finish_reason ("end_turn" | "tool_use" | "max_tokens")

Usage:
    from llm.wrapper import LLMWrapper, build_llm_wrapper, LLMMessage

    llm = build_llm_wrapper(tool_schemas=[...])

    response = await llm.chat(
        messages=[LLMMessage(role="user", content="I earn ₹1.2L/month. Help me plan.")],
    )
    print(response.text)
    print(response.tool_calls)   # [{"tool": "health_score", "args": {...}}]
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
import httpx

from config import settings
from logger import get_logger, metrics, timed

log = get_logger(__name__)

# ── Anthropic client (shared, thread-safe) ────────────────────────────────────
_sync_client: Optional[anthropic.Anthropic] = None
_async_client: Optional[anthropic.AsyncAnthropic] = None


def _get_sync_client() -> anthropic.Anthropic:
    global _sync_client
    if _sync_client is None:
        _sync_client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=httpx.Timeout(settings.llm.timeout_seconds),
            max_retries=0,   # We handle retries ourselves for full control
        )
    return _sync_client


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=httpx.Timeout(settings.llm.timeout_seconds),
            max_retries=0,
        )
    return _async_client


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMMessage:
    """A single turn in the conversation history."""
    role: str           # "user" | "assistant"
    content: str        # Text content

    def to_api_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCallDecision:
    """
    Represents Claude's decision to call a tool.
    Extracted from the API response content blocks.
    """
    tool_name: str
    args: Dict[str, Any]
    call_id: str = ""           # Anthropic tool_use block ID (for multi-turn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args":      self.args,
            "call_id":   self.call_id,
        }


@dataclass
class LLMResponse:
    """
    Structured output from a single LLM call.
    The Planner only ever receives this — never raw API response.
    """
    text: str                                    # Reasoning / final answer text
    tool_calls: List[ToolCallDecision] = field(default_factory=list)
    finish_reason: str = "end_turn"              # "end_turn" | "tool_use" | "max_tokens"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":          self.text,
            "tool_calls":    [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "tokens":        {"input": self.input_tokens, "output": self.output_tokens},
            "latency_ms":    round(self.latency_ms, 2),
            "model":         self.model,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Custom exceptions
# ══════════════════════════════════════════════════════════════════════════════

class LLMError(RuntimeError):
    """Base class for LLM layer errors."""


class LLMRateLimitError(LLMError):
    """Raised when the API returns 429. Triggers retry with backoff."""


class LLMTimeoutError(LLMError):
    """Raised when the request exceeds timeout_seconds."""


class LLMAuthError(LLMError):
    """Raised on 401 — invalid API key. Non-retryable."""


class LLMContextLengthError(LLMError):
    """Raised when the conversation history exceeds the context window."""


# ══════════════════════════════════════════════════════════════════════════════
# System prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
    """
    Construct the full system prompt injected into every API call.

    Sections:
      1. Identity & role constraints
      2. Tool usage rules (WHEN and HOW to call each tool)
      3. Registered tool schemas (name, description, input fields)
      4. Output format instructions
      5. Indian finance domain rules
    """
    tool_list = "\n".join(
        f"  - {t['name']}: {t['description']}"
        for t in tool_schemas
    )
    tool_schema_json = json.dumps(tool_schemas, indent=2)

    return f"""{settings.llm.system_prompt}

═══════════════════════════════════════════════════════════
ROLE BOUNDARY — CRITICAL
═══════════════════════════════════════════════════════════
You are the REASONING and PLANNING layer. You:
  ✅ Analyse the user's financial situation
  ✅ Decide WHICH tools to call and WHEN
  ✅ Interpret tool outputs and synthesise a final recommendation
  ✅ Ask clarifying questions when user data is incomplete
  ❌ NEVER produce numerical financial predictions yourself
  ❌ NEVER guess SIP amounts, returns, tax savings, or corpus values
  ❌ NEVER skip calling a tool when numerical data is needed

═══════════════════════════════════════════════════════════
TOOL USAGE RULES
═══════════════════════════════════════════════════════════
Available tools:
{tool_list}

Tool selection logic:
  1. User asks "how much should I invest?"        → sip_calculator
  2. User asks "when can I retire?" or FIRE       → fire_planner
  3. User asks "save tax" or "Form 16"            → tax_wizard
  4. User asks "how healthy are my finances?"     → health_score
  5. User asks about portfolio / allocation       → rl_predict (then sip_calculator)
  6. Always start a NEW user session with        → health_score

You may call multiple tools per response when needed.
Always call tools BEFORE composing the final answer.

═══════════════════════════════════════════════════════════
REGISTERED TOOL SCHEMAS
═══════════════════════════════════════════════════════════
{tool_schema_json}

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════
When you need to call a tool, respond with a JSON block:

<tool_call>
{{
  "tool": "<tool_name>",
  "args": {{
    "<field>": <value>
  }}
}}
</tool_call>

You may include multiple <tool_call> blocks in one response.
After ALL tool calls are resolved, compose a final plain-text
answer for the user in simple, warm, jargon-free Indian English.
Use ₹ for amounts. Format numbers with Indian comma notation (1,00,000).

═══════════════════════════════════════════════════════════
INDIAN FINANCE DOMAIN RULES
═══════════════════════════════════════════════════════════
- Always consider both old and new tax regimes before advising
- SIP = Systematic Investment Plan (mutual fund monthly investment)
- ELSS = Equity Linked Savings Scheme (80C tax saving + equity)
- PPF = Public Provident Fund (guaranteed, 80C, 15-year lock-in)
- EPF = Employee Provident Fund (employer-matched, retirement)
- NPS = National Pension System (80CCD(1B) extra ₹50,000 deduction)
- FOIR < 40% is healthy; above 50% is a red flag
- Emergency fund = 6× monthly expenses in liquid instruments
- Term insurance = minimum 10× annual income
- Recommend SEBI-registered platforms only (not crypto, not chit funds)
- Never guarantee returns; use "expected" or "historical average"
"""


# ══════════════════════════════════════════════════════════════════════════════
# Response parser
# ══════════════════════════════════════════════════════════════════════════════

def _parse_tool_calls_from_text(text: str) -> tuple[str, List[ToolCallDecision]]:
    """
    Extract <tool_call>...</tool_call> blocks from the model's text output.

    This handles the case where we use text-based tool calling (no native
    tool_use blocks). Falls back gracefully if JSON is malformed.

    Returns:
        (cleaned_text, list_of_tool_call_decisions)
    """
    import re
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = pattern.findall(text)

    tool_calls = []
    for i, raw in enumerate(matches):
        try:
            parsed = json.loads(raw.strip())
            tool_calls.append(ToolCallDecision(
                tool_name=parsed.get("tool", ""),
                args=parsed.get("args", {}),
                call_id=f"tc_{i}",
            ))
        except json.JSONDecodeError as exc:
            log.warning(
                "Failed to parse tool_call JSON block",
                block_index=i,
                error=str(exc),
                raw=raw[:200],
            )

    # Remove tool_call blocks from the text to get clean prose
    clean_text = pattern.sub("", text).strip()
    return clean_text, tool_calls


def _parse_native_tool_calls(content_blocks: list) -> tuple[str, List[ToolCallDecision]]:
    """
    Parse native Anthropic tool_use content blocks.
    Used when tool_use blocks appear directly in the API response
    (i.e., native function-calling mode).
    """
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCallDecision(
                tool_name=block.name,
                args=block.input if isinstance(block.input, dict) else {},
                call_id=block.id,
            ))

    return "\n".join(text_parts).strip(), tool_calls


# ══════════════════════════════════════════════════════════════════════════════
# LLM Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class LLMWrapper:
    """
    Production wrapper around the Claude API.

    Key behaviours:
      - Async-first (uses AsyncAnthropic internally)
      - Retry with exponential backoff on 429 / 5xx
      - Injects tool schemas into system prompt once, reused across calls
      - Parses both text-based and native tool_use responses
      - Token budget tracking (warns when approaching context limit)
      - Streaming support for real-time UX
    """

    # Anthropic context limit (claude-sonnet-4) — conservative budget
    _CONTEXT_TOKEN_LIMIT = 180_000
    _WARN_TOKEN_THRESHOLD = 150_000

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._model = settings.llm.model
        self._max_tokens = settings.llm.max_tokens
        self._temperature = settings.llm.temperature
        self._max_retries = settings.llm.max_retries
        self._retry_backoff = settings.llm.retry_backoff_seconds
        self._total_tokens_used = 0

    # ── Async chat ────────────────────────────────────────────────────────────

    @timed("llm.chat", tags={"layer": "llm"})
    async def chat(
        self,
        messages: List[LLMMessage],
        session_id: str = "",
        context: Optional[str] = None,   # Extra context appended to system prompt
    ) -> LLMResponse:
        """
        Send a conversation to Claude and return a structured LLMResponse.

        Args:
            messages:   Full conversation history (user + assistant turns).
            session_id: Used for logging / metrics.
            context:    Optional extra text appended to system prompt this call
                        (e.g., serialised UserFinancialState summary).

        Returns:
            LLMResponse with text, tool_calls, token counts, latency.

        Raises:
            LLMAuthError:          Invalid API key.
            LLMContextLengthError: History too long.
            LLMError:              All retries exhausted.
        """
        system = self._system_prompt
        if context:
            system = system + f"\n\n═══ USER FINANCIAL CONTEXT ═══\n{context}\n"

        api_messages = [m.to_api_dict() for m in messages]

        t0 = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                client = _get_async_client()
                response = await client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=system,
                    messages=api_messages,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # ── Parse response ────────────────────────────────────────────
                finish_reason = response.stop_reason or "end_turn"
                input_tokens  = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self._total_tokens_used += input_tokens + output_tokens

                # Check for native tool_use blocks first
                has_native_tool_use = any(
                    getattr(b, "type", "") == "tool_use"
                    for b in response.content
                )

                if has_native_tool_use:
                    text, tool_calls = _parse_native_tool_calls(response.content)
                else:
                    # Extract text and parse <tool_call> XML tags
                    raw_text = "".join(
                        b.text for b in response.content
                        if getattr(b, "type", "") == "text"
                    )
                    text, tool_calls = _parse_tool_calls_from_text(raw_text)

                # ── Token budget warning ──────────────────────────────────────
                if input_tokens > self._WARN_TOKEN_THRESHOLD:
                    log.warning(
                        "Approaching context token limit",
                        input_tokens=input_tokens,
                        limit=self._CONTEXT_TOKEN_LIMIT,
                        session_id=session_id,
                    )

                # ── Metrics ───────────────────────────────────────────────────
                metrics.increment("llm.requests")
                metrics.increment(
                    "llm.tool_calls_requested",
                    amount=len(tool_calls),
                )
                metrics.observe("llm.latency_ms", elapsed_ms)
                metrics.observe("llm.input_tokens", input_tokens)
                metrics.observe("llm.output_tokens", output_tokens)

                log.info(
                    "LLM response received",
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_calls=len(tool_calls),
                    finish_reason=finish_reason,
                    latency_ms=round(elapsed_ms, 2),
                    session_id=session_id,
                )

                return LLMResponse(
                    text=text,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    model=self._model,
                )

            except anthropic.RateLimitError as exc:
                last_exc = exc
                wait = self._retry_backoff * (2 ** (attempt - 1))
                log.warning(
                    f"Rate limited (attempt {attempt}/{self._max_retries}). "
                    f"Retrying in {wait:.1f}s...",
                    session_id=session_id,
                )
                metrics.increment("llm.rate_limit_hits")
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
                else:
                    raise LLMRateLimitError(
                        f"Rate limit hit on all {self._max_retries} attempts."
                    ) from exc

            except anthropic.AuthenticationError as exc:
                log.error("LLM authentication failed — check ANTHROPIC_API_KEY")
                metrics.increment("llm.auth_errors")
                raise LLMAuthError("Invalid Anthropic API key.") from exc

            except anthropic.BadRequestError as exc:
                if "context_length" in str(exc).lower():
                    raise LLMContextLengthError(
                        "Conversation history exceeds context limit. "
                        "Trim earlier messages before retrying."
                    ) from exc
                raise LLMError(f"Bad request to LLM API: {exc}") from exc

            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
                last_exc = exc
                wait = self._retry_backoff * (2 ** (attempt - 1))
                log.warning(
                    f"LLM connection error (attempt {attempt}/{self._max_retries}). "
                    f"Retrying in {wait:.1f}s...",
                    error=str(exc),
                    session_id=session_id,
                )
                metrics.increment("llm.connection_errors")
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
                else:
                    raise LLMTimeoutError(
                        f"LLM API unreachable after {self._max_retries} attempts."
                    ) from exc

            except anthropic.InternalServerError as exc:
                last_exc = exc
                wait = self._retry_backoff * (2 ** (attempt - 1))
                log.warning(
                    f"LLM server error 5xx (attempt {attempt}/{self._max_retries}). "
                    f"Retrying in {wait:.1f}s...",
                    session_id=session_id,
                )
                metrics.increment("llm.server_errors")
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
                else:
                    raise LLMError(
                        f"LLM API returned 5xx on all {self._max_retries} attempts."
                    ) from exc

        # Should not reach here, but satisfy type checker
        raise LLMError(f"LLM chat failed after all retries: {last_exc}")

    # ── Streaming chat ────────────────────────────────────────────────────────

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        session_id: str = "",
        context: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming version of chat(). Yields text chunks as they arrive.
        Tool calls are NOT extracted during streaming — use chat() for agentic loops.
        Ideal for the final user-facing response where latency matters.

        Usage:
            async for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)
        """
        system = self._system_prompt
        if context:
            system = system + f"\n\n═══ USER FINANCIAL CONTEXT ═══\n{context}\n"

        client = _get_async_client()
        log.info("Starting streaming LLM response", session_id=session_id)

        async with client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[m.to_api_dict() for m in messages],
        ) as stream:
            async for text_chunk in stream.text_stream:
                metrics.increment("llm.stream_chunks")
                yield text_chunk

        metrics.increment("llm.stream_responses")
        log.info("Streaming response complete", session_id=session_id)

    # ── Single-turn shortcut ──────────────────────────────────────────────────

    async def ask(
        self,
        prompt: str,
        context: Optional[str] = None,
        session_id: str = "",
    ) -> LLMResponse:
        """
        Convenience method for single-turn queries (no history needed).
        Used by the Planner for classification / routing decisions.

        Example:
            resp = await llm.ask("Classify this intent: 'save tax on salary'")
        """
        return await self.chat(
            messages=[LLMMessage(role="user", content=prompt)],
            context=context,
            session_id=session_id,
        )

    # ── Tool result injection ─────────────────────────────────────────────────

    @staticmethod
    def format_tool_result_for_prompt(
        tool_name: str,
        result: Dict[str, Any],
        call_id: str = "",
    ) -> str:
        """
        Format a tool result as a structured string to append to conversation.
        This is inserted as an assistant message after a tool call so Claude
        can reason about the result before composing its final response.

        The Planner calls this to build the next message in the chain.
        """
        result_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        return (
            f"<tool_result tool=\"{tool_name}\" call_id=\"{call_id}\">\n"
            f"{result_json}\n"
            f"</tool_result>"
        )

    # ── Token budget ──────────────────────────────────────────────────────────

    def token_budget_remaining(self, current_input_tokens: int) -> int:
        """Estimate remaining tokens before hitting context limit."""
        return max(0, self._CONTEXT_TOKEN_LIMIT - current_input_tokens)

    @property
    def total_tokens_used(self) -> int:
        """Cumulative tokens across all calls in this session."""
        return self._total_tokens_used

    def reset_token_counter(self) -> None:
        self._total_tokens_used = 0

    # ── History management ────────────────────────────────────────────────────

    @staticmethod
    def trim_history(
        messages: List[LLMMessage],
        max_turns: int = 20,
    ) -> List[LLMMessage]:
        """
        Trim conversation history to avoid context length overflow.
        Always keeps the FIRST message (initial user state) and the
        last `max_turns` messages.
        """
        if len(messages) <= max_turns:
            return messages
        # Keep first message (user financial context) + recent history
        trimmed = [messages[0]] + messages[-(max_turns - 1):]
        log.info(
            "Conversation history trimmed",
            original_len=len(messages),
            trimmed_len=len(trimmed),
        )
        return trimmed


# ══════════════════════════════════════════════════════════════════════════════
# Intent classifier (lightweight Claude call for routing)
# ══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Fast, single-turn Claude call to classify user intent into one of the
    known tool categories. Used by the Planner to pre-select which tool(s)
    to call before the full agentic loop.

    This avoids wasting the full planning budget on simple routing decisions.
    """

    INTENTS = [
        "fire_planning",
        "sip_calculation",
        "tax_optimization",
        "health_score",
        "portfolio_review",
        "debt_advice",
        "insurance_advice",
        "general_question",
        "life_event",          # bonus, marriage, inheritance, new baby
        "unknown",
    ]

    CLASSIFICATION_PROMPT = """
Classify the user's financial query into EXACTLY ONE of these intents:
{intents}

User query: "{query}"

Respond with ONLY the intent label — no explanation, no punctuation.
""".strip()

    def __init__(self, llm: LLMWrapper) -> None:
        self._llm = llm

    async def classify(self, query: str, session_id: str = "") -> str:
        """
        Returns the most likely intent label for a user query.
        Falls back to "unknown" on any error.
        """
        prompt = self.CLASSIFICATION_PROMPT.format(
            intents="\n".join(f"  - {i}" for i in self.INTENTS),
            query=query,
        )
        try:
            response = await self._llm.ask(prompt, session_id=session_id)
            intent = response.text.strip().lower().replace(" ", "_")
            if intent not in self.INTENTS:
                log.warning(
                    "Unexpected intent label from classifier",
                    raw=intent,
                    fallback="unknown",
                )
                return "unknown"
            metrics.increment("llm.intent_classified", tags={"intent": intent})
            log.debug("Intent classified", intent=intent, query=query[:80])
            return intent
        except Exception as exc:
            log.error("Intent classification failed", error=str(exc))
            return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_wrapper(tool_schemas: List[Dict[str, Any]]) -> LLMWrapper:
    """
    Build a fully configured LLMWrapper with the FinMentor system prompt
    and all registered tool schemas injected.

    Called once at application startup; result shared across all requests.

    Args:
        tool_schemas: Output of ToolRegistry.list_tools()
    """
    if not settings.anthropic_api_key:
        log.warning(
            "ANTHROPIC_API_KEY not set — LLM calls will fail. "
            "Set the env var or add it to .env"
        )

    system_prompt = _build_system_prompt(tool_schemas)

    log.info(
        "LLMWrapper built",
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        tools_registered=len(tool_schemas),
        system_prompt_chars=len(system_prompt),
    )

    return LLMWrapper(system_prompt=system_prompt)


def build_intent_classifier(llm: LLMWrapper) -> IntentClassifier:
    """Build an IntentClassifier backed by the given LLMWrapper."""
    return IntentClassifier(llm)


# ══════════════════════════════════════════════════════════════════════════════
# Self-test (mock mode — no real API key needed)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json
    from unittest.mock import AsyncMock, MagicMock, patch

    MOCK_TOOL_SCHEMAS = [
        {"name": "health_score",   "description": "Compute money health score", "tags": ["health"], "input_fields": ["state"]},
        {"name": "sip_calculator", "description": "Calculate monthly SIP",       "tags": ["sip"],    "input_fields": ["state", "goal_name", "target_amount"]},
        {"name": "fire_planner",   "description": "Build FIRE roadmap",          "tags": ["fire"],   "input_fields": ["state"]},
        {"name": "tax_wizard",     "description": "Identify tax savings",         "tags": ["tax"],    "input_fields": ["state"]},
        {"name": "rl_predict",     "description": "RL portfolio prediction",      "tags": ["rl"],     "input_fields": ["state"]},
    ]

    async def run_tests() -> None:
        print("=== LLMWrapper Self-Test (Mocked API) ===\n")

        # ── Test 1: System prompt construction ───────────────────────────────
        system_prompt = _build_system_prompt(MOCK_TOOL_SCHEMAS)
        print("── Test 1: System Prompt ──")
        print(f"  Length: {len(system_prompt)} chars")
        assert "ROLE BOUNDARY" in system_prompt
        assert "sip_calculator" in system_prompt
        assert "NEVER produce numerical" in system_prompt
        print("  ✓ Contains role boundary rules")
        print("  ✓ Contains tool schemas")
        print("  ✓ Contains Indian finance rules\n")

        # ── Test 2: Tool call text parser ─────────────────────────────────────
        print("── Test 2: Tool Call Parser ──")
        mock_response_text = """
I'll start by computing your Money Health Score.

<tool_call>
{
  "tool": "health_score",
  "args": {"state": "user_state_placeholder"}
}
</tool_call>

Then I'll calculate the SIP needed for your retirement goal.

<tool_call>
{
  "tool": "sip_calculator",
  "args": {
    "state": "user_state_placeholder",
    "goal_name": "Retirement",
    "target_amount": 60000000
  }
}
</tool_call>

Let me review these results and provide your plan.
"""
        clean, tool_calls = _parse_tool_calls_from_text(mock_response_text)
        print(f"  Tool calls parsed : {len(tool_calls)}")
        for tc in tool_calls:
            print(f"    → {tc.tool_name}  args={list(tc.args.keys())}")
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_name == "health_score"
        assert tool_calls[1].tool_name == "sip_calculator"
        assert "Let me review" in clean
        print("  ✓ Correct number of tool calls extracted")
        print("  ✓ Clean text stripped of <tool_call> blocks\n")

        # ── Test 3: LLMMessage formatting ─────────────────────────────────────
        print("── Test 3: LLMMessage Formatting ──")
        msg = LLMMessage(role="user", content="Help me plan my finances")
        d = msg.to_api_dict()
        assert d == {"role": "user", "content": "Help me plan my finances"}
        print("  ✓ LLMMessage serialises correctly\n")

        # ── Test 4: Tool result formatting ────────────────────────────────────
        print("── Test 4: Tool Result Formatting ──")
        formatted = LLMWrapper.format_tool_result_for_prompt(
            tool_name="health_score",
            result={"overall_score": 67.5, "grade": "Good"},
            call_id="tc_0",
        )
        assert "health_score" in formatted
        assert "67.5" in formatted
        print(f"  ✓ Tool result formatted:\n{formatted[:200]}\n")

        # ── Test 5: History trimming ───────────────────────────────────────────
        print("── Test 5: History Trimming ──")
        long_history = [
            LLMMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(30)
        ]
        trimmed = LLMWrapper.trim_history(long_history, max_turns=10)
        assert len(trimmed) == 10
        assert trimmed[0].content == "Message 0"   # First always kept
        print(f"  Original: {len(long_history)} messages")
        print(f"  Trimmed : {len(trimmed)} messages")
        print("  ✓ First message preserved, tail kept\n")

        # ── Test 6: Mock LLM chat ──────────────────────────────────────────────
        print("── Test 6: Mocked LLM Chat ──")

        # Build mock response
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = (
            "I'll assess your financial health first.\n"
            "<tool_call>\n{\"tool\": \"health_score\", \"args\": {}}\n</tool_call>\n"
            "Based on results, here is your plan."
        )
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1024
        mock_response.usage.output_tokens = 256

        wrapper = build_llm_wrapper(MOCK_TOOL_SCHEMAS)

        with patch("llm.wrapper._get_async_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await wrapper.chat(
                messages=[LLMMessage(role="user", content="Assess my finances")],
                session_id="sess_test_001",
            )

        print(f"  Text length    : {len(response.text)} chars")
        print(f"  Tool calls     : {len(response.tool_calls)}")
        print(f"  Finish reason  : {response.finish_reason}")
        print(f"  Input tokens   : {response.input_tokens}")
        print(f"  Output tokens  : {response.output_tokens}")
        print(f"  Has tool calls : {response.has_tool_calls}")
        assert response.has_tool_calls
        assert response.tool_calls[0].tool_name == "health_score"
        print("  ✓ Tool call extracted from mocked response\n")

        # ── Test 7: Intent classifier ──────────────────────────────────────────
        print("── Test 7: IntentClassifier ──")
        queries_and_expected = [
            ("How much SIP do I need for retirement?",        "sip_calculation"),
            ("Can I retire at 45?",                           "fire_planning"),
            ("Save tax on my ₹20L salary",                   "tax_optimization"),
            ("Rate my financial health",                      "health_score"),
            ("I got a bonus of ₹5 lakhs, what should I do?", "life_event"),
        ]

        classifier = build_intent_classifier(wrapper)
        for query, expected in queries_and_expected:
            with patch("llm.wrapper._get_async_client") as mock_get_client:
                mock_block2 = MagicMock()
                mock_block2.type = "text"
                mock_block2.text = expected   # Mock returns expected label
                mock_response2 = MagicMock()
                mock_response2.content = [mock_block2]
                mock_response2.stop_reason = "end_turn"
                mock_response2.usage.input_tokens = 50
                mock_response2.usage.output_tokens = 5
                mock_client2 = MagicMock()
                mock_client2.messages.create = AsyncMock(return_value=mock_response2)
                mock_get_client.return_value = mock_client2

                intent = await classifier.classify(query, session_id="sess_test_001")

            print(f"  Query: '{query[:50]}...' → {intent}")
            assert intent == expected, f"Expected {expected}, got {intent}"

        print("  ✓ All intents classified correctly\n")

        print("── Metrics Snapshot ──")
        snap = metrics.snapshot()
        for c in snap["counters"]:
            print(f"  {c['name']:<45} {int(c['value'])}")

    asyncio.run(run_tests())
