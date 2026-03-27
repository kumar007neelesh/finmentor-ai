"""
config.py — FinMentor AI Central Configuration
================================================
Single source of truth for all system-wide settings.
Config-driven design ensures zero hardcoded values anywhere else.

Usage:
    from config import settings
    print(settings.llm.model)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# ── Load .env early so os.getenv picks it up ──────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.resolve()
MODELS_DIR = ROOT_DIR / "saved_models"
LOGS_DIR   = ROOT_DIR / "logs"
MEMORY_DIR = ROOT_DIR / "memory_store"

for _dir in (MODELS_DIR, LOGS_DIR, MEMORY_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE     = "moderate"
    AGGRESSIVE   = "aggressive"


class ModelBackend(str, Enum):
    STABLE_BASELINES3 = "sb3"
    RLLIB             = "rllib"
    XGBOOST           = "xgboost"
    LSTM              = "lstm"
    MOCK              = "mock"


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# Sub-config models
# ══════════════════════════════════════════════════════════════════════════════

class LLMConfig(BaseModel):
    model: str = Field(
        default="gemini-2.5-flash",   # ← changed from claude-sonnet
        description="Gemini model ID.",
    )
    max_tokens: int = Field(default=4096, ge=256, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=30, ge=5)
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_seconds: float = Field(default=1.5, ge=0.5)

    system_prompt: str = Field(
        default=(
            "You are FinMentor AI, an expert Indian personal finance advisor. "
            "You reason step-by-step, select the right financial tools, and "
            "produce actionable plans grounded in Indian tax law (80C, 80D, NPS), "
            "SEBI regulations, and proven investment frameworks (SIP, FIRE, etc.). "
            "NEVER make up numerical predictions — always delegate to financial tools."
        )
    )

class RLModelConfig(BaseModel):
    """Configuration for the RL prediction model."""

    # Suppress the model_ namespace warning
    model_config = ConfigDict(protected_namespaces=())

    backend: ModelBackend = Field(default=ModelBackend.MOCK)
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to saved SB3/RLlib model. None → use mock.",
    )
    algorithm: str = Field(default="PPO")
    observation_dim: int = Field(default=20)
    action_space_size: int = Field(default=5)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    device: str = Field(default="cpu")


class MLFallbackConfig(BaseModel):
    """Configuration for the supervised ML fallback model."""

    backend: ModelBackend = Field(default=ModelBackend.XGBOOST)
    model_path: Optional[Path] = Field(default=None)
    feature_names: List[str] = Field(
        default_factory=lambda: [
            "age", "monthly_income", "monthly_expense", "existing_sip",
            "emergency_fund_months", "debt_emi_ratio", "risk_score",
            "investment_horizon_years", "has_term_insurance",
            "has_health_insurance",
        ]
    )


class AgentConfig(BaseModel):
    """Configuration for the core agent loop."""

    max_planning_steps: int = Field(default=8)
    max_tool_calls_per_step: int = Field(default=3)
    enable_memory: bool = Field(default=True)
    memory_max_episodes: int = Field(default=50)
    memory_backend: str = Field(default="json")

    evaluation_enabled: bool = Field(default=True)
    prediction_tolerance_pct: float = Field(default=10.0)


class FinanceConfig(BaseModel):
    """Indian financial constants used across all tools."""

    tax_slabs_new_regime: Dict[str, float] = Field(
        default_factory=lambda: {
            "0-300000":    0.00,
            "300001-700000": 0.05,
            "700001-1000000": 0.10,
            "1000001-1200000": 0.15,
            "1200001-1500000": 0.20,
            "1500001+":    0.30,
        }
    )

    section_80c_limit: int = 150_000
    section_80d_self_limit: int = 25_000
    section_80d_parents_limit: int = 50_000
    nps_80ccd1b_limit: int = 50_000
    hra_exemption_metro_pct: float = 0.50
    hra_exemption_nonmetro_pct: float = 0.40

    default_equity_return_pct: float = 12.0
    default_debt_return_pct: float = 7.0
    default_inflation_pct: float = 6.0
    default_emergency_fund_months: int = 6

    fire_multiplier: int = 25


class ServerConfig(BaseModel):
    """API server settings for production deployment."""

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    reload: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class ObservabilityConfig(BaseModel):
    """Logging and monitoring settings."""

    log_level: LogLevel = LogLevel.INFO
    log_file: Path = LOGS_DIR / "finmentor.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    enable_json_logs: bool = True
    metrics_enabled: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Root settings object
# ══════════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """
    FinMentor AI — Root Settings
    All sub-configs nested here.
    Override via env vars: FINMENTOR__LLM__MODEL=claude-opus-4-5
    """

    # API Keys (from env, never hardcoded)
    anthropic_api_key: str = Field(default="")
    google_api_key: str = Field(default="")  
    @field_validator("google_api_key", mode="before")
    @classmethod
    def google_key_from_env(cls, v: str) -> str:
        if not v:
            v = os.getenv("GOOGLE_API_KEY", "")
        return v    
    # Sub-configs
    llm:         LLMConfig            = Field(default_factory=LLMConfig)
    rl_model:    RLModelConfig        = Field(default_factory=RLModelConfig)
    ml_fallback: MLFallbackConfig     = Field(default_factory=MLFallbackConfig)
    agent:       AgentConfig          = Field(default_factory=AgentConfig)
    finance:     FinanceConfig        = Field(default_factory=FinanceConfig)
    server:      ServerConfig         = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Environment flag
    environment: str = Field(default="development")

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def api_key_from_env(cls, v: str) -> str:
        """Fall back to ANTHROPIC_API_KEY env var if not set via pydantic-settings."""
        if not v:
            v = os.getenv("ANTHROPIC_API_KEY", "")
        return v
    
    @field_validator("google_api_key", mode="before") 
    @classmethod
    def google_key_from_env(cls, v: str) -> str:
        if not v:
            v = os.getenv("GOOGLE_API_KEY", "")
        return v


    @field_validator("environment", mode="before")
    @classmethod
    def environment_from_env(cls, v: str) -> str:
        if not v:
            v = os.getenv("FINMENTOR_ENV", "development")
        return v

    def model_post_init(self, __context) -> None:
        """Validate API key presence in production after full model init."""
        if self.environment == "production" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in production environment. "
                "Set it via the ANTHROPIC_API_KEY environment variable."
            )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    model_config = ConfigDict(
        env_prefix="FINMENTOR__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
settings = Settings()

# Convenience re-export so other modules can do:
#   from config import OBS_DIM
OBS_DIM = settings.rl_model.observation_dim


# ══════════════════════════════════════════════════════════════════════════════
# Convenience helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_model_path(filename: str) -> Path:
    return MODELS_DIR / filename


def get_log_path(filename: str) -> Path:
    return LOGS_DIR / filename


def get_memory_path(session_id: str) -> Path:
    return MEMORY_DIR / f"{session_id}.json"


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FinMentor AI — Config Loaded ===")
    print(f"  Environment    : {settings.environment}")
    print(f"  LLM Model      : {settings.llm.model}")
    print(f"  RL Backend     : {settings.rl_model.backend.value}")
    print(f"  ML Fallback    : {settings.ml_fallback.backend.value}")
    print(f"  Max Plan Steps : {settings.agent.max_planning_steps}")
    print(f"  Log Level      : {settings.observability.log_level.value}")
    print(f"  80C Limit      : ₹{settings.finance.section_80c_limit:,}")
    print(f"  FIRE Multiplier: {settings.finance.fire_multiplier}×")
    print(f"  API Key set    : {'Yes' if settings.anthropic_api_key else 'No (set ANTHROPIC_API_KEY)'}")
    print()
    print("  Paths:")
    print(f"    MODELS_DIR : {MODELS_DIR}")
    print(f"    LOGS_DIR   : {LOGS_DIR}")
    print(f"    MEMORY_DIR : {MEMORY_DIR}")
    print()
    print("  ✅ Config OK — no errors")