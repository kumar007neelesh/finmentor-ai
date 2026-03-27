# FinMentor AI

AI-powered personal finance mentor for India. Built with Claude/Gemini LLM,
custom RL model, and financial planning tools.

## Features
- Money Health Score (6-dimension wellness assessment)
- FIRE Path Planner (retirement corpus calculator)
- SIP Calculator (goal-based investment planning)
- Tax Wizard (80C, 80D, NPS optimisation)
- Life Event Advisor (bonus, marriage, baby, job loss)

## Local Setup

1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/finmentor-ai.git
   cd finmentor-ai

2. Create virtual environment
   python -m venv env
   env\Scripts\activate        # Windows
   source env/bin/activate     # Mac/Linux

3. Install dependencies
   pip install -r requirements.txt

4. Set environment variables
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY

5. Run the application
   python main.py

6. Open in browser
   http://localhost:8080

## Tech Stack
- Backend: FastAPI + Python
- LLM: Anthropic Claude (or Google Gemini)
- Frontend: Vanilla HTML/CSS/JS
- Memory: JSON file store