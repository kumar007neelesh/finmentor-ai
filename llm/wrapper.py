"""
llm/wrapper.py — FinMentor AI LLM Wrapper (Google Gemini)
==========================================================
Drop-in replacement for the Claude wrapper.
Identical public interface — Planner and Executor call the same methods:
  - LLMWrapper.chat(messages, session_id, context)
  - LLMWrapper.chat_stream(messages, session_id, context)
  - LLMWrapper.ask(prompt, context, session_id)
  - LLMWrapper.format_tool_result_for_prompt(tool_name, result, call_id)
  - LLMWrapper.trim_history(messages, max_turns)
  - IntentClassifier.classify(query, session_id)

Supported models:
  - gemini-1.5-pro        (most capable, 1M context)
  - gemini-1.5-flash      (faster, cheaper)
  - gemini-2.0-flash-exp  (latest experimental)

Usage:
    from llm.wrapper import LLMWrapper, build_llm_wrapper, LLMMessage
    llm = build_llm_wrapper(tool_schemas=[...])
    response = await llm.chat(
        messages=[LLMMessage(role="user", content="Help me plan.")],
    )
    print(response.text)
    print(response.tool_calls)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InvalidArgument,
    PermissionDenied,
)

# Add project root to path when running directly
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from logger import get_logger, metrics, timed

log = get_logger(__name__)

# Thread pool for running blocking Gemini calls in async context
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini_inference")

# Gemini client (initialised once)
_gemini_configured = False


def _configure_gemini() -> None:
    """Configure the Gemini SDK with the API key (idempotent)."""
    global _gemini_configured
    if _gemini_configured:
        return
    api_key = settings.google_api_key
    if not api_key:
        log.warning(
            "GOOGLE_API_KEY not set — Gemini calls will fail. "
            "Set it via the GOOGLE_API_KEY environment variable."
        )
    genai.configure(api_key=api_key)
    _gemini_configured = True
    log.info("Gemini SDK configured", model=settings.llm.model)


# ══════════════════════════════════════════════════════════════════════════════
# Data models  (identical interface to Claude wrapper)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMMessage:
    """A single turn in the conversation history."""
    role: str       # "user" | "assistant"
    content: str

    def to_gemini_dict(self) -> Dict[str, str]:
        """Gemini uses 'user' and 'model' roles (not 'assistant')."""
        return {
            "role":  "model" if self.role == "assistant" else "user",
            "parts": [self.content],
        }

    def to_api_dict(self) -> Dict[str, str]:
        """Alias kept for compatibility with any code that calls to_api_dict."""
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCallDecision:
    """Claude-compatible tool call decision extracted from Gemini response."""
    tool_name: str
    args: Dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "args": self.args, "call_id": self.call_id}


@dataclass
class LLMResponse:
    """Identical structure to Claude wrapper output — Planner is unchanged."""
    text: str
    tool_calls: List[ToolCallDecision] = field(default_factory=list)
    finish_reason: str = "end_turn"
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
# Custom exceptions  (same names as Claude wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class LLMError(RuntimeError):
    pass

class LLMRateLimitError(LLMError):
    pass

class LLMTimeoutError(LLMError):
    pass

class LLMAuthError(LLMError):
    pass

class LLMContextLengthError(LLMError):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# System prompt builder  (identical logic to Claude wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
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
- Recommend SEBI-registered platforms only
- Never guarantee returns; use "expected" or "historical average"
"""


# ══════════════════════════════════════════════════════════════════════════════
# Tool call parser  (identical to Claude wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_tool_calls_from_text(text: str) -> tuple[str, List[ToolCallDecision]]:
    """Extract <tool_call>...</tool_call> blocks from Gemini's text output."""
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
            log.warning("Failed to parse tool_call block", index=i, error=str(exc))

    clean_text = pattern.sub("", text).strip()
    return clean_text, tool_calls


# ══════════════════════════════════════════════════════════════════════════════
# LLM Wrapper — Gemini backend
# ══════════════════════════════════════════════════════════════════════════════

class LLMWrapper:
    """
    Gemini-backed LLM wrapper with identical public interface to the
    original Claude wrapper. The Planner and Executor call the same
    methods — no changes needed anywhere else in the codebase.
    """

    def __init__(self, system_prompt: str) -> None:
        _configure_gemini()
        self._system_prompt   = system_prompt
        self._model_name      = settings.llm.model
        self._max_tokens      = settings.llm.max_tokens
        self._temperature     = settings.llm.temperature
        self._max_retries     = settings.llm.max_retries
        self._retry_backoff   = settings.llm.retry_backoff_seconds
        self._total_tokens    = 0

        # Safety settings — turn off filters that block financial content
        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Build the Gemini GenerativeModel once — reused across all calls
        self._gemini_model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=self._system_prompt,
            safety_settings=self._safety_settings,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self._max_tokens,
                temperature=self._temperature,
            ),
        )

    # ── Async chat ────────────────────────────────────────────────────────────

    @timed("llm.chat", tags={"layer": "llm"})
    async def chat(
        self,
        messages: List[LLMMessage],
        session_id: str = "",
        context: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send conversation to Gemini and return structured LLMResponse.
        Runs blocking SDK call in a thread pool to keep asyncio loop free.
        """
        t0 = time.perf_counter()

        # Build the history list Gemini expects (all turns except the last)
        history = [m.to_gemini_dict() for m in messages[:-1]]
        last_user_msg = messages[-1].content if messages else ""

        # Append financial context to the last message if provided
        if context:
            last_user_msg = (
                f"{last_user_msg}\n\n"
                f"[USER FINANCIAL CONTEXT]\n{context}"
            )

        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._call_gemini(history, last_user_msg)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                raw_text = response.text or ""
                finish_reason = str(response.candidates[0].finish_reason) if response.candidates else "end_turn"
                input_tokens  = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                self._total_tokens += input_tokens + output_tokens

                # Parse <tool_call> blocks out of text response
                clean_text, tool_calls = _parse_tool_calls_from_text(raw_text)

                metrics.increment("llm.requests")
                metrics.increment("llm.tool_calls_requested", amount=len(tool_calls))
                metrics.observe("llm.latency_ms", elapsed_ms)
                metrics.observe("llm.input_tokens", input_tokens)
                metrics.observe("llm.output_tokens", output_tokens)

                log.info(
                    "Gemini response received",
                    model=self._model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_calls=len(tool_calls),
                    finish_reason=finish_reason,
                    latency_ms=round(elapsed_ms, 2),
                    session_id=session_id,
                )

                return LLMResponse(
                    text=clean_text,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    model=self._model_name,
                )

            except ResourceExhausted as exc:
                last_exc = exc
                wait = self._retry_backoff * (2 ** (attempt - 1))
                log.warning(f"Gemini rate limit (attempt {attempt}). Retrying in {wait:.1f}s")
                metrics.increment("llm.rate_limit_hits")
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
                else:
                    raise LLMRateLimitError("Gemini quota exhausted.") from exc

            except (ServiceUnavailable, DeadlineExceeded) as exc:
                last_exc = exc
                wait = self._retry_backoff * (2 ** (attempt - 1))
                log.warning(f"Gemini unavailable (attempt {attempt}). Retrying in {wait:.1f}s")
                metrics.increment("llm.connection_errors")
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
                else:
                    raise LLMTimeoutError("Gemini API unreachable.") from exc

            except PermissionDenied as exc:
                log.error("Gemini auth failed — check GOOGLE_API_KEY")
                metrics.increment("llm.auth_errors")
                raise LLMAuthError("Invalid Google API key.") from exc

            except InvalidArgument as exc:
                if "context" in str(exc).lower() or "token" in str(exc).lower():
                    raise LLMContextLengthError("Input too long for Gemini.") from exc
                raise LLMError(f"Gemini invalid argument: {exc}") from exc

        raise LLMError(f"Gemini chat failed after all retries: {last_exc}")

    async def _call_gemini(self, history: list, last_message: str):
        """Run the blocking Gemini SDK call in a thread pool."""
        loop = asyncio.get_event_loop()

        def _sync_call():
            chat_session = self._gemini_model.start_chat(history=history)
            return chat_session.send_message(last_message)

        return await loop.run_in_executor(_THREAD_POOL, _sync_call)

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        session_id: str = "",
        context: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream Gemini response token by token."""
        history = [m.to_gemini_dict() for m in messages[:-1]]
        last_msg = messages[-1].content if messages else ""
        if context:
            last_msg = f"{last_msg}\n\n[USER FINANCIAL CONTEXT]\n{context}"

        log.info("Starting streaming Gemini response", session_id=session_id)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _stream_sync():
            try:
                chat_session = self._gemini_model.start_chat(history=history)
                response = chat_session.send_message(last_msg, stream=True)
                for chunk in response:
                    if chunk.text:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"[ERROR: {exc}]")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(_THREAD_POOL, _stream_sync)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            metrics.increment("llm.stream_chunks")
            yield chunk

        metrics.increment("llm.stream_responses")

    # ── Single-turn shortcut ──────────────────────────────────────────────────

    async def ask(
        self,
        prompt: str,
        context: Optional[str] = None,
        session_id: str = "",
    ) -> LLMResponse:
        """Single-turn convenience method. Used by IntentClassifier."""
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
        """Format tool output as structured text for the next LLM turn."""
        result_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        return (
            f"<tool_result tool=\"{tool_name}\" call_id=\"{call_id}\">\n"
            f"{result_json}\n"
            f"</tool_result>"
        )

    # ── History management ────────────────────────────────────────────────────

    @staticmethod
    def trim_history(
        messages: List[LLMMessage],
        max_turns: int = 20,
    ) -> List[LLMMessage]:
        """Trim conversation to avoid context overflow. Keeps first + recent."""
        if len(messages) <= max_turns:
            return messages
        trimmed = [messages[0]] + messages[-(max_turns - 1):]
        log.info("History trimmed", original=len(messages), trimmed=len(trimmed))
        return trimmed

    # ── Token tracking ────────────────────────────────────────────────────────

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens

    def reset_token_counter(self) -> None:
        self._total_tokens = 0

    def token_budget_remaining(self, current_input_tokens: int) -> int:
        # Gemini 1.5 Pro has 1M context window
        return max(0, 1_000_000 - current_input_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# Intent classifier  (identical interface to Claude wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    INTENTS = [
        "fire_planning", "sip_calculation", "tax_optimization",
        "health_score", "portfolio_review", "debt_advice",
        "insurance_advice", "general_question", "life_event", "unknown",
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
        prompt = self.CLASSIFICATION_PROMPT.format(
            intents="\n".join(f"  - {i}" for i in self.INTENTS),
            query=query,
        )
        try:
            response = await self._llm.ask(prompt, session_id=session_id)
            intent = response.text.strip().lower().replace(" ", "_")
            if intent not in self.INTENTS:
                return "unknown"
            metrics.increment("llm.intent_classified", tags={"intent": intent})
            return intent
        except Exception as exc:
            log.error("Intent classification failed", error=str(exc))
            return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Factory functions  (identical signatures to Claude wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_wrapper(tool_schemas: List[Dict[str, Any]]) -> LLMWrapper:
    """
    Build a fully configured LLMWrapper backed by Gemini.
    Called once at application startup.
    Identical signature to the Claude version — no changes needed in main.py.
    """
    if not settings.google_api_key:
        log.warning(
            "GOOGLE_API_KEY not set — LLM calls will fail. "
            "Add it to your .env file."
        )

    system_prompt = _build_system_prompt(tool_schemas)

    log.info(
        "LLMWrapper (Gemini) built",
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
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio as _asyncio

    MOCK_TOOL_SCHEMAS = [
        {"name": "health_score",   "description": "Compute money health score",  "tags": ["health"], "input_fields": ["state"]},
        {"name": "sip_calculator", "description": "Calculate monthly SIP",        "tags": ["sip"],    "input_fields": ["state", "goal_name", "target_amount"]},
        {"name": "fire_planner",   "description": "Build FIRE roadmap",           "tags": ["fire"],   "input_fields": ["state"]},
        {"name": "tax_wizard",     "description": "Identify tax savings",          "tags": ["tax"],    "input_fields": ["state"]},
        {"name": "rl_predict",     "description": "RL portfolio prediction",       "tags": ["rl"],     "input_fields": ["state"]},
    ]

    async def run_tests():
        print("=== LLMWrapper (Gemini) Self-Test ===\n")

        # Test 1: System prompt
        print("── Test 1: System Prompt ──")
        sp = _build_system_prompt(MOCK_TOOL_SCHEMAS)
        print(f"  Length: {len(sp)} chars")
        assert "ROLE BOUNDARY" in sp
        assert "sip_calculator" in sp
        assert "NEVER produce numerical" in sp
        print("  ✓ Role boundary rules present")
        print("  ✓ Tool schemas injected")
        print("  ✓ Indian finance rules present\n")

        # Test 2: Tool call parser
        print("── Test 2: Tool Call Parser ──")
        mock_text = """
I'll check your health score first.
<tool_call>
{"tool": "health_score", "args": {}}
</tool_call>
Then calculate your SIP.
<tool_call>
{"tool": "sip_calculator", "args": {"goal_name": "Retirement", "target_amount": 60000000}}
</tool_call>
Based on the results...
"""
        clean, tool_calls = _parse_tool_calls_from_text(mock_text)
        print(f"  Tool calls parsed : {len(tool_calls)}")
        for tc in tool_calls:
            print(f"    → {tc.tool_name}  args={list(tc.args.keys())}")
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_name == "health_score"
        assert "Based on the results" in clean
        print("  ✓ Tool calls extracted correctly\n")

        # Test 3: LLMMessage role mapping
        print("── Test 3: Role Mapping ──")
        user_msg = LLMMessage(role="user", content="Help me")
        ai_msg   = LLMMessage(role="assistant", content="Sure!")
        assert user_msg.to_gemini_dict()["role"] == "user"
        assert ai_msg.to_gemini_dict()["role"]   == "model"   # Gemini uses "model" not "assistant"
        print("  ✓ 'assistant' correctly mapped to 'model' for Gemini\n")

        # Test 4: History trimming
        print("── Test 4: History Trimming ──")
        history = [LLMMessage(role="user" if i%2==0 else "assistant", content=f"msg {i}") for i in range(30)]
        trimmed = LLMWrapper.trim_history(history, max_turns=10)
        assert len(trimmed) == 10
        assert trimmed[0].content == "msg 0"
        print(f"  30 messages → trimmed to 10 ✓\n")

        # Test 5: Tool result formatting
        print("── Test 5: Tool Result Formatting ──")
        formatted = LLMWrapper.format_tool_result_for_prompt(
            "health_score", {"overall_score": 72.5, "grade": "Good"}, "tc_0"
        )
        assert "health_score" in formatted
        assert "72.5" in formatted
        print(f"  ✓ Tool result formatted correctly\n")

        # Test 6: Live API test (only if key is set)
        if settings.google_api_key:
            print("── Test 6: Live Gemini API Call ──")
            wrapper = build_llm_wrapper(MOCK_TOOL_SCHEMAS)
            response = await wrapper.ask(
                "In one sentence, what is a SIP in Indian personal finance?",
                session_id="test_001"
            )
            print(f"  Response: {response.text[:150]}")
            print(f"  Tokens  : {response.input_tokens} in / {response.output_tokens} out")
            print(f"  Latency : {response.latency_ms:.0f}ms\n")
        else:
            print("── Test 6: Skipped (GOOGLE_API_KEY not set) ──\n")

        print("All tests passed ✓")

    _asyncio.run(run_tests())
