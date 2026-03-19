# Research Paper Assistant

Research Paper Assistant is a Streamlit app that reads a research-paper PDF and produces:

1. Detailed key insights
2. A level-adapted explanation (`beginner`, `intermediate`, `advanced`)
3. A fit assessment against the user's topic of interest
4. Better-fit paper recommendations from Tavily search
5. A formatted downloadable PDF report

The workflow is built with LangGraph and runs in sequence:

1. Extract text from uploaded PDF
2. Generate structured insights with Groq LLM
3. Rewrite content by selected learning level
4. Search external papers with Tavily using the user's topic
5. Judge if the uploaded paper matches the user's topic
6. Recommend better-fit papers
7. Build a polished report PDF

## Tech Stack

- Python
- Streamlit
- LangGraph
- Groq API (`llama-3.1-8b-instant` by default)
- Tavily Search API
- pypdf
- reportlab

## Project Structure

- `app.py`: Streamlit UI (upload, level selection, run, download)
- `rag.py`: Core assistant logic and LangGraph workflow
- `requirements.txt`: Python dependencies

## Code Documentation

- `app.py` and `rag.py` include extensive docstrings and inline comments to explain flow, design choices, and function responsibilities.

## Code Documentation

- `app.py` and `rag.py` include extensive docstrings and inline comments to explain flow, design choices, and function responsibilities.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
# Optional:
# GROQ_LLM_MODEL=llama-3.1-8b-instant
# GROQ_BASE_URL=https://api.groq.com/openai/v1
# TAVILY_BASE_URL=https://api.tavily.com
```

## Run

```bash
streamlit run app.py
```

Then:

1. Upload a PDF
2. Enter your topic of interest
3. Choose difficulty level
4. Click **Generate Simplified Research Report**
5. Download the generated PDF

## Output Paths

- Uploaded PDFs: `outputs/uploads/`
- Generated reports: `outputs/reports/`

## Notes

- The generated report is section-based with paragraph explanations (not bullet-only summaries).
- The app now includes topic-fit analysis and external paper recommendations.
- If `GROQ_API_KEY` is missing, the app stops and shows an error.

## Troubleshooting

- `RuntimeError: Groq request failed ... rate_limit_exceeded`:
  - Cause: your account exceeded tokens-per-minute quota.
  - Fix: wait for the cooldown shown in the error and run again, or reduce prompt/output token sizes in `rag.py`, or upgrade Groq limits.
- `No external papers found`:
  - Ensure `TAVILY_API_KEY` is set correctly in `.env`.

## Troubleshooting

- `RuntimeError: Groq request failed ... rate_limit_exceeded`:
  - Cause: your account exceeded tokens-per-minute quota.
  - Fix: wait for the cooldown shown in the error and run again, or reduce prompt/output token sizes in `rag.py`, or upgrade Groq limits.
- `No external papers found`:
  - Ensure `TAVILY_API_KEY` is set correctly in `.env`.
