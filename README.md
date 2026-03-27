<div align="center">

# 🪙 FinMentor AI

### AI-Powered Personal Finance Mentor for India

*Turning confused savers into confident investors — one conversation at a time.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Claude](https://img.shields.io/badge/LLM-Claude%20%2F%20Gemini-orange?style=flat)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

[Live Demo](#) · [API Docs](http://localhost:8080/docs) · [Report Bug](issues)

![FinMentor AI Screenshot](static/screenshot.png)

</div>

---

## The Problem

95% of Indians have no financial plan. A professional financial advisor costs ₹25,000+ per year and serves only High Net-worth Individuals. FinMentor AI changes that — bringing expert-level financial guidance to anyone with a smartphone.

---

## What It Does

FinMentor AI is a conversational financial advisor that understands your complete financial picture and gives personalised, actionable advice grounded in Indian tax law, SEBI regulations, and proven investment frameworks.

| Feature | What it gives you |
|---|---|
| **Money Health Score** | A 0–100 score across 6 dimensions: emergency fund, insurance, diversification, debt health, tax efficiency, and retirement readiness |
| **FIRE Path Planner** | Month-by-month roadmap to Financial Independence — corpus needed, required SIP, asset allocation, and post-retirement longevity |
| **SIP Calculator** | Exact monthly SIP to hit any goal (retirement, education, home) with step-up projections and inflation adjustment |
| **Tax Wizard** | Every deduction you're missing under 80C, 80D, NPS — old vs new regime comparison with your real numbers |
| **Life Event Advisor** | Instant financial restructuring for bonus, salary hike, marriage, new baby, job loss, inheritance, or medical emergency |
| **Portfolio Advisor** | RL model-driven allocation recommendations with confidence scoring and automatic fallback |

---

## How It Works

### System Architecture

```
User Query + Financial Profile
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                      │
│                                                             │
│  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  LLM Layer   │───▶│   Planner   │───▶│   Executor    │  │
│  │(Claude/Gemini│    │ (ReAct Loop)│    │(Session Mgmt) │  │
│  └──────────────┘    └─────────────┘    └───────┬───────┘  │
│                                                  │          │
│  ┌───────────────────────────────────────────────▼───────┐  │
│  │                    Tool Registry                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌─────────┐  │  │
│  │  │rl_predict│  │sip_calc  │  │ fire │  │tax/health│  │  │
│  │  │(RL Model)│  │(Math)    │  │plnnr │  │ tools    │  │  │
│  │  └──────────┘  └──────────┘  └──────┘  └─────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │    Memory    │    │  Evaluator  │    │  Environment  │  │
│  │(JSON/Redis)  │    │(Drift Detect│    │(UserState/20d)│  │
│  └──────────────┘    └─────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### The ReAct Planning Loop

Every user query goes through a structured reasoning loop — the agent doesn't just answer, it thinks:

```
User: "How much SIP do I need to retire at 55?"
         │
         ▼
1. CLASSIFY INTENT    → fire_planning  (1 fast LLM call)
         │
         ▼
2. REASON             → "I need to assess health first, then compute FIRE corpus"
         │
         ▼
3. ACT                → Call health_score tool + rl_predict tool (parallel)
         │
         ▼
4. OBSERVE            → health=68/100, predicted_return=12.3%, action=increase_equity
         │
         ▼
5. REASON AGAIN       → "Now I can compute the FIRE corpus with these numbers"
         │
         ▼
6. ACT                → Call fire_planner tool
         │
         ▼
7. SYNTHESISE         → LLM composes final answer with all tool data
         │
         ▼
User gets: ₹42,500/month SIP, ₹6.2Cr corpus needed, 23-year roadmap
```

### The RL Model

The system includes a pretrained Reinforcement Learning model (PPO/Stable-Baselines3 compatible) that predicts the optimal portfolio action from a 20-dimensional observation vector built from the user's financial state:

```
UserFinancialState → obs_vector (20 features, all normalized 0–1)
    ├── age, income (log-scaled), savings_rate, FOIR
    ├── emergency_fund_ratio, portfolio_to_income
    ├── equity_pct, debt_pct, risk_score
    ├── insurance_score, has_term, has_health
    ├── investment_horizon, goal_shortfall_ratio
    ├── effective_tax_rate, existing_sip_ratio
    └── debt_ratio, city_tier, employment_type
          │
          ▼
   RL Model (PPO)
          │
          ▼
   Action + Confidence Score
   ├── increase_equity    → if long horizon, adequate EF
   ├── reduce_equity      → if high FOIR or conservative profile
   ├── increase_sip       → if low savings rate
   ├── build_emergency    → if EF < 3 months (top priority)
   └── optimize_tax       → if 30% bracket with unused deductions
```

If confidence < 65% (configurable), it automatically falls back to the ML heuristic model — the LLM never sees this complexity, it just gets a clean `PredictionResult`.

### Memory & Evaluation

Every conversation turn is stored as an `Episode` in the memory system. After 30 days, the Evaluator compares the predicted portfolio return against actual market returns and scores 5 dimensions:

- Return accuracy (35% weight)
- Tool selection appropriateness (25%)
- Goal coverage (20%)
- Risk profile alignment (12%)
- Response clarity (8%)

If the rolling Mean Absolute Error exceeds 2× the configured tolerance, a retraining alert fires — giving you a signal to retrain the RL model on fresh data.

---

## Project Structure

```
finmentor_ai/
│
├── config.py               Central configuration — single source of truth
│                           All settings via env vars, zero hardcoded values
│
├── logger.py               Structured JSON logging (loguru) + in-process
│                           metrics + @timed decorator + audit trail
│
├── environment/
│   └── state.py            UserFinancialState — the canonical data model
│                           Validates all user data, computes derived ratios
│                           (FOIR, savings_rate, effective_tax_rate),
│                           serialises to 20-dim NumPy obs vector for RL
│
├── models/
│   └── rl_loader.py        RL model loader — wraps SB3 PPO or deterministic
│                           mock as a callable tool. Handles confidence
│                           scoring, LowConfidenceError, async inference
│
├── tools/
│   └── registry.py         Tool registry — registers, validates (Pydantic),
│                           and dispatches all financial tools. Handles
│                           RL→ML fallback chain and parallel async execution
│                           Contains: sip_calculator, fire_planner,
│                           tax_wizard, health_score, rl_predict, ml_predict
│
├── llm/
│   ├── wrapper.py          LLM wrapper (Claude) — manages API calls, retry
│   └── wrapper_gemini.py   LLM wrapper (Gemini) — drop-in replacement
│                           Both expose identical interface to the Planner
│
├── agent/
│   ├── planner.py          ReAct loop — LLM-driven reasoning and tool
│   │                       orchestration. Produces FinancialPlan struct
│   │
│   ├── executor.py         Session lifecycle — handles life events, applies
│   │                       state transitions, routes plan vs follow-up,
│   │                       wires memory persistence
│   │
│   ├── memory.py           Episodic memory — stores every plan as Episode
│   │                       Three backends: JSON (default), InMemory (tests),
│   │                       Redis (production). Provides context for planner
│   │
│   └── evaluator.py        Prediction evaluator — MAE tracking, drift
│                           detection, retraining alerts, plan quality scoring
│
├── static/
│   └── index.html          Frontend — single HTML file, no framework,
│                           connects to backend via fetch() API calls
│
└── main.py                 FastAPI application — all HTTP endpoints,
                            lifespan startup, serves frontend, CLI demo mode
```

---

## Tech Stack

### Backend

| Layer | Technology | Why |
|---|---|---|
| Web framework | **FastAPI** | Async-native, auto OpenAPI docs, Pydantic validation |
| LLM | **Anthropic Claude** (default) or **Google Gemini** | Best reasoning quality; swap via one config change |
| RL Model | **Stable-Baselines3** (PPO/DQN/A2C) | Industry-standard RL; mock backend for dev |
| Data validation | **Pydantic v2** | Type-safe financial state, catches bad inputs at ingestion |
| Configuration | **pydantic-settings** | Env var driven, zero hardcoded values, nested config |
| Logging | **loguru** | Structured JSON logs, file rotation, async-safe |
| Numerics | **NumPy** | 20-dim observation vector, financial math |
| Server | **Uvicorn** | ASGI, production-grade, supports multiple workers |

### Frontend

| Layer | Technology | Why |
|---|---|---|
| UI | **Vanilla HTML/CSS/JS** | Zero build step, zero dependencies, instant load |
| Fonts | **DM Serif Display + DM Sans** | Elegant, readable, financial-grade aesthetic |
| Charts | None (text-based KPI cards) | Keeps bundle size at 0kb |
| API | **fetch()** | Native browser API, no axios needed |

### Infrastructure

| Layer | Technology | Why |
|---|---|---|
| Containerisation | **Docker** (multi-stage) | Reproducible builds, ~200MB image |
| Memory store | **JSON** (dev) / **Redis** (prod) | Zero-dep default, scalable upgrade path |
| Deployment | **Render.com** | Free tier, auto-deploy from GitHub, Python native |
| CI/CD | **GitHub** → Render auto-deploy | Push to main = live in 2 minutes |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions` | Create session with financial profile |
| `POST` | `/sessions/{id}/chat` | Send a query, get AI financial plan |
| `POST` | `/sessions/{id}/onboard` | Auto 3-step onboarding sequence |
| `POST` | `/sessions/{id}/life-event` | Handle life event (bonus, marriage, etc.) |
| `GET` | `/sessions/{id}` | Get session summary |
| `DELETE` | `/sessions/{id}` | End session |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus-compatible metrics snapshot |
| `POST` | `/evaluate/{user_id}` | Batch evaluate prediction accuracy |
| `POST` | `/evaluate/system` | System-wide drift check |

Full interactive docs at `http://localhost:8080/docs` when running locally.

---

## Local Setup

### Prerequisites

- Python 3.11 or higher
- An Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)
- OR a Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/finmentor-ai.git
cd finmentor-ai

# 2. Create and activate virtual environment
python -m venv env

# Windows
env\Scripts\activate

# Mac/Linux
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API key
```

### Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here    # Required if using Claude
GOOGLE_API_KEY=AIzaSy-your-key-here       # Required if using Gemini

FINMENTOR_ENV=development
FINMENTOR__LLM__MODEL=claude-sonnet-4-20250514   # or gemini-1.5-pro
FINMENTOR__RL_MODEL__BACKEND=mock
FINMENTOR__AGENT__MEMORY_BACKEND=json
```

### Running

```bash
# Option 1: Full application (API + Frontend)
python main.py
# Open http://localhost:8080

# Option 2: Interactive terminal demo (no browser needed)
python main.py --demo

# Option 3: Different port
python main.py --port 9000

# Option 4: Development mode with auto-reload
python main.py --reload
```

### Testing Individual Modules

Every file has a built-in self-test. Run them in order to verify your setup:

```bash
python config.py                    # No API key needed
python logger.py                    # No API key needed
python -m environment.state         # No API key needed
python -m models.rl_loader          # No API key needed
python -m tools.registry            # No API key needed
python -m llm.wrapper               # No API key needed (mocked)
python -m agent.planner             # No API key needed (mocked)
python -m agent.executor            # No API key needed (mocked)
python -m agent.memory              # No API key needed
python -m agent.evaluator           # No API key needed
python main.py --demo               # Needs API key
```

---

## Deployment

### Deploy to Render (Free)

1. Fork this repository on GitHub
2. Go to [render.com](https://render.com) and sign in with GitHub
3. Click **New +** → **Web Service** → select your fork
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
5. Add environment variables in the Render dashboard:
   - `ANTHROPIC_API_KEY` = your key
   - `FINMENTOR_ENV` = production
   - `FINMENTOR__RL_MODEL__BACKEND` = mock
6. Click **Create Web Service**

Your app will be live at `https://your-app-name.onrender.com` in ~3 minutes.

### Deploy with Docker

```bash
# Build
docker build -t finmentor-ai .

# Run
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-your-key \
  -e FINMENTOR_ENV=production \
  finmentor-ai

# Or with Docker Compose
docker compose up --build
```

---

## Switching LLM Providers

The LLM layer is fully swappable. To switch from Claude to Gemini:

```bash
# 1. Install Gemini SDK
pip install google-generativeai

# 2. Swap the wrapper file
mv llm/wrapper.py llm/wrapper_claude.py
mv llm/wrapper_gemini.py llm/wrapper.py

# 3. Update .env
GOOGLE_API_KEY=AIzaSy-your-key
FINMENTOR__LLM__MODEL=gemini-1.5-pro
```

No other files need to change. The Planner, Executor, and Tools are completely LLM-agnostic.

---

## Configuration Reference

All settings are driven by environment variables using the `FINMENTOR__` prefix:

```bash
# LLM
FINMENTOR__LLM__MODEL=claude-sonnet-4-20250514
FINMENTOR__LLM__MAX_TOKENS=4096
FINMENTOR__LLM__TEMPERATURE=0.2

# RL Model
FINMENTOR__RL_MODEL__BACKEND=mock          # mock | sb3
FINMENTOR__RL_MODEL__CONFIDENCE_THRESHOLD=0.65

# Agent
FINMENTOR__AGENT__MAX_PLANNING_STEPS=8
FINMENTOR__AGENT__MEMORY_BACKEND=json      # json | redis | memory
FINMENTOR__AGENT__PREDICTION_TOLERANCE_PCT=10.0

# Server
FINMENTOR__SERVER__PORT=8080
FINMENTOR__SERVER__WORKERS=4

# Logging
FINMENTOR__OBSERVABILITY__LOG_LEVEL=INFO
FINMENTOR__OBSERVABILITY__ENABLE_JSON_LOGS=true
```

---

## Indian Finance Domain Knowledge

FinMentor AI is specifically calibrated for Indian personal finance:

- **Tax laws:** Sections 80C (₹1.5L limit), 80D (health insurance), 80CCD(1B) (NPS ₹50K extra), HRA exemption
- **Tax regimes:** Old vs new regime comparison with exact numbers (FY 2024-25 slabs)
- **Instruments:** ELSS, PPF, EPF, NPS, FD, Sovereign Gold Bonds, direct equity
- **Rules of thumb:** FOIR < 40% healthy, emergency fund = 6× expenses, term cover = 10× income
- **FIRE:** 25× annual expenses rule (4% withdrawal rate), 100-age equity allocation
- **Returns:** 12% long-run equity CAGR, 7% debt, 6% inflation (all configurable)

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-mutual-fund-xray`)
3. Make your changes with tests
4. Run the self-tests for affected modules
5. Submit a pull request with a clear description

### Areas that would benefit from contribution

- CAMS/KFintech statement parser for MF Portfolio X-Ray
- Real portfolio return API integration (for the Evaluator)
- Trained RL model weights (currently using deterministic mock)
- Couple's joint financial planning tool
- WhatsApp Business API integration

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Built as part of the Economic Times AI Mentor challenge. Designed to make professional-grade financial planning accessible to every Indian, not just HNIs.

---

<div align="center">

Made with ❤️ for India's 1.4 billion aspiring investors

*"The best time to start investing was yesterday. The second best time is today."*

</div>
