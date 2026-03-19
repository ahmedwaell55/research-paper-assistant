"""Core LangGraph pipeline for research-paper analysis and report generation.

This module contains the full backend orchestration used by the Streamlit app:
- PDF text extraction
- Insight extraction via Groq LLM
- Level-adapted simplification
- Topic-fit analysis plus Tavily-based recommendations
- Final report rendering to PDF
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, TypedDict

import requests
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from langgraph.graph import END, START, StateGraph


Level = Literal["beginner", "intermediate", "advanced"]


class PaperState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes.

    Each key may be added by a node and consumed by downstream nodes.
    total=False keeps fields optional so state can be built incrementally.
    """

    # Inputs
    pdf_path: str
    level: Level
    output_dir: str
    user_topic: str

    # Derived metadata and extracted text
    paper_title: str
    raw_text: str

    # LLM-generated sections
    key_insights: str
    simplified_content: str

    # Recommendation subsystem data
    tavily_results: str
    fit_assessment: str # Direct judgment of how well the paper matches the user topic, with explanation.
    recommended_papers: str

    # Final artifact
    formatted_pdf_path: str


class ResearchPaperAssistant:
    """High-level orchestrator for paper analysis, adaptation, and reporting."""

    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str = "",
        groq_model: str = "llama-3.1-8b-instant",
        groq_base_url: str = "https://api.groq.com/openai/v1",
        tavily_base_url: str = "https://api.tavily.com",
    ) -> None:
        """Initialize API settings and precompile LangGraph workflow."""
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY")

        # Store provider configuration.
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        self.groq_model = groq_model
        self.groq_base_url = groq_base_url.rstrip("/")
        self.tavily_base_url = tavily_base_url.rstrip("/")

        # Compile graph once; invoked many times for user runs.
        self.graph = self._build_graph()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and trim text."""
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize a string to be safe for filenames by replacing unsafe characters with underscores."""
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
        return cleaned[:80] or "research_paper" # If result is empty → return default

    @staticmethod
    def _split_blocks(text: str) -> list[str]:
        """Split text into paragraph-like blocks using blank-line boundaries.
        This helps preserve intended structure when rendering in the PDF.
        """
        return [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]

    @staticmethod
    def _normalize_heading(text: str) -> str:
        """Drop Markdown heading markers (##, ###, etc.) for PDF display."""
        return re.sub(r"^#{1,3}\s*", "", text.strip())

    @staticmethod
    def _is_heading(text: str) -> bool:
        """detect if a block looks like a heading line.
        Heuristic rules:
        - Must be a single line (no newlines)
        - Starts with 1-3 # followed by space and text, OR starts with numbered list pattern, OR ends with a colon (common in LLM outputs)
        """
        line = text.strip()
        if "\n" in line:
            return False
        if re.match(r"^#{1,3}\s+\S+", line):
            return True
        if re.match(r"^\d+[\).:-]\s+\S+", line):
            return True
        return line.endswith(":")

    def _extract_text(self, state: PaperState) -> PaperState:
        """Read PDF pages and return normalized full text plus inferred title."""
        pdf_path = state["pdf_path"]
        # pypdf extracts text page-by-page; merge into one raw source string.
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = self._clean_text("\n".join(pages))

        if not raw_text:
            raise ValueError("No readable text found in the uploaded PDF")

        # Use filename stem as a practical title proxy.
        paper_title = Path(pdf_path).stem
        return {"raw_text": raw_text, "paper_title": paper_title}

    def _tavily_search(self, query: str, max_results: int = 8) -> list[dict]:
        """Search Tavily for candidate papers relevant to the user topic."""
        # Recommendation features are optional if no Tavily key is available.
        if not self.tavily_api_key:
            return []

        url = f"{self.tavily_base_url}/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
        }
        # Tavily search endpoint requires JSON payload with api_key field.
        response = requests.post(url, json=payload, timeout=60)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Tavily request failed: {response.text}") from exc

        data = response.json()
        # Defensive shape check to avoid downstream type errors.
        results = data.get("results", [])
        if not isinstance(results, list):
            return []
        return results

    @staticmethod
    def _format_tavily_results(results: list[dict]) -> str:
        """Convert Tavily JSON results to compact text for the LLM prompt."""
        if not results:
            return "No external papers found."

        lines = []
        for idx, result in enumerate(results, start=1):
            title = (result.get("title") or "Untitled").strip()
            url = (result.get("url") or "").strip()
            content = (result.get("content") or "").strip()
            snippet = content[:350].strip()
            lines.append(f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}")
        return "\n\n".join(lines)

    def _chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
        """Call Groq chat completion API and return assistant text output."""
        url = f"{self.groq_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

        # Single synchronous request; upstream caller handles exceptions.
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Groq request failed: {response.text}") from exc

        data = response.json()
        try:
            # Groq is OpenAI-compatible for this response shape.
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid response format from Groq") from exc

    def _insights_extractor(self, state: PaperState) -> PaperState:
        """Generate structured insight sections from extracted paper text."""
        source_text = state["raw_text"]
        # Keep prompt bounded to prevent very large payloads.
        text_for_prompt = source_text[:20000]

        system_prompt = (
            "You are an expert research analyst. Extract the most important findings from papers. "
            "Focus on claims, methods, contributions, evidence, and limitations."
        )
        user_prompt = (
            "Read this research paper content and produce detailed important insights.\n"
            "Output requirements:\n"
            "1) Use section headings only (no bullet points).\n"
            "2) Include these sections in order:\n"
            "   ## Core Problem\n"
            "   ## Proposed Method\n"
            "   ## Main Findings\n"
            "   ## Evidence and Results\n"
            "   ## Limitations and Open Questions\n"
            "3) Under each heading, write one rich paragraph of 5-8 sentences.\n"
            "4) Stay faithful to the provided text and avoid generic filler.\n\n"
            f"Paper text:\n{text_for_prompt}"
        )

        # Larger output budget for detailed paragraph-style sections.
        insights = self._chat_completion(system_prompt, user_prompt, max_tokens=1600)
        return {"key_insights": insights}

    def _simplifier(self, state: PaperState) -> PaperState:
        """Rewrite insights according to the chosen audience level."""
        level: Level = state["level"]
        insights = state["key_insights"]

        # Style controls for each reading level.
        level_guide: Dict[Level, str] = {
            "beginner": (
                "Use very simple language, short sentences, and everyday analogies. "
                "Avoid heavy jargon, define terms when needed, and keep concepts approachable."
            ),
            "intermediate": (
                "Use clear technical language with moderate detail. "
                "Keep key terms, but explain them briefly and avoid unnecessary complexity."
            ),
            "advanced": (
                "Use precise technical language and preserve advanced concepts, equations context, "
                "and nuanced tradeoffs. Keep it concise but expert-level."
            ),
        }

        system_prompt = (
            "You are a pedagogy-focused AI that adapts explanations to the learner level while "
            "staying faithful to the source insights."
        )
        user_prompt = (
            f"Target level: {level}\n"
            f"Level instructions: {level_guide[level]}\n\n"
            "Rewrite the following research insights for this level with depth and clarity.\n"
            "Output requirements:\n"
            "1) Use section headings only (no bullet points).\n"
            "2) Include these sections in order:\n"
            "   ## Plain-Language Explanation\n"
            "   ## Key Terms and Meanings\n"
            "   ## Practical Takeaways\n"
            "3) Under each heading, write one rich paragraph of 5-8 sentences.\n"
            "4) Do not over-summarize; preserve important technical meaning.\n\n"
            f"Insights:\n{insights}"
        )

        # Higher token budget supports richer multi-section explanations.
        simplified = self._chat_completion(system_prompt, user_prompt, max_tokens=2200)
        return {"simplified_content": simplified}

    def _topic_fit_and_recommendations(self, state: PaperState) -> PaperState:
        """Assess topic fit and propose better papers using Tavily candidates."""
        user_topic = state.get("user_topic", "").strip()
        # Explicit fallback when topic was not provided.
        if not user_topic:
            return {
                "tavily_results": "No topic was provided.",
                "fit_assessment": "No fit assessment generated because no topic was provided.",
                "recommended_papers": "No recommendations generated because no topic was provided.",
            }

        # Build a focused query around user intent.
        query = f"best research papers about {user_topic}"
        tavily_results = self._tavily_search(query)
        formatted_results = self._format_tavily_results(tavily_results)

        system_prompt = (
            "You are a research advisor. Compare one uploaded paper summary against the user's stated interests, "
            "then recommend better-fit papers from search results."
        )
        user_prompt = (
            f"User topic of interest: {user_topic}\n\n"
            "Uploaded paper insights:\n"
            f"{state.get('key_insights', '')}\n\n"
            "Uploaded paper simplified explanation:\n"
            f"{state.get('simplified_content', '')}\n\n"
            "Candidate papers from web search:\n"
            f"{formatted_results}\n\n"
            "Produce output with exactly these two sections and headings:\n"
            "## Is This Paper a Good Fit?\n"
            "Write a direct judgment (good fit / partial fit / weak fit), then explain why in one detailed paragraph.\n\n"
            "## Better-Fit Paper Recommendations\n"
            "Recommend 3-5 papers from the candidate list. For each recommendation include:\n"
            "Paper title, why it matches the topic, and URL."
        )
        recommendation_text = self._chat_completion(system_prompt, user_prompt, max_tokens=1400)

        # Try to separate response into explicit fit/recommendation sections.
        sections = re.split(r"(?=^##\s+)", recommendation_text, flags=re.MULTILINE)
        fit_assessment = ""
        recommended_papers = recommendation_text.strip()
        for section in sections:
            trimmed = section.strip()
            if not trimmed:
                continue
            normalized = trimmed.lower()
            if normalized.startswith("## is this paper a good fit?"):
                fit_assessment = trimmed
            elif normalized.startswith("## better-fit paper recommendations"):
                recommended_papers = trimmed

        # Fallback if model does not follow expected heading format.
        if not fit_assessment:
            fit_assessment = "## Is This Paper a Good Fit?\nA fit assessment could not be parsed from the model output."

        return {
            "tavily_results": formatted_results,
            "fit_assessment": fit_assessment,
            "recommended_papers": recommended_papers,
        }

    def _build_pdf(self, state: PaperState) -> PaperState:
        """Render assistant outputs into a downloadable, styled PDF report."""
        output_dir = Path(state["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        title = state.get("paper_title", "Research Paper")
        safe_title = self._safe_name(title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{safe_title}_{state['level']}_{timestamp}.pdf"

        # Create custom styles on top of reportlab defaults.
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#0b3a66"),
                spaceAfter=14,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=styles["Heading2"],
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#123c5a"),
                spaceBefore=8,
                spaceAfter=6,
            )
        )
        styles.add(
            ParagraphStyle(
                name="Body",
                parent=styles["BodyText"],
                fontSize=10.5,
                leading=15,
            )
        )

        document = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm,
            title=f"Research Paper Assistant - {title}",
        )

        # Build the report sequentially as a list of Flowable elements.
        story = []
        story.append(Paragraph("Research Paper Assistant Report", styles["ReportTitle"]))
        story.append(Paragraph(f"Paper: {title}", styles["Body"]))
        story.append(Paragraph(f"Difficulty level: {state['level'].capitalize()}", styles["Body"]))
        story.append(Spacer(1, 12))

        # Render "Important Insights" section from model text blocks.
        story.append(Paragraph("Important Insights", styles["SectionTitle"]))
        for block in self._split_blocks(state["key_insights"]):
            if self._is_heading(block):
                story.append(Paragraph(self._normalize_heading(block), styles["SectionTitle"]))
                continue
            story.append(Paragraph(block.replace("\n", "<br/>"), styles["Body"]))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 10))

        # Render level-adapted explanation.
        story.append(Paragraph("Simplified Explanation", styles["SectionTitle"]))
        for block in self._split_blocks(state["simplified_content"]):
            if self._is_heading(block):
                story.append(Paragraph(self._normalize_heading(block), styles["SectionTitle"]))
                continue
            story.append(Paragraph(block.replace("\n", "<br/>"), styles["Body"]))
            story.append(Spacer(1, 6))

        # Render topic fit assessment.
        story.append(Spacer(1, 10))
        story.append(Paragraph("Paper Fit to Your Interest", styles["SectionTitle"]))
        for block in self._split_blocks(state.get("fit_assessment", "")):
            if self._is_heading(block):
                story.append(Paragraph(self._normalize_heading(block), styles["SectionTitle"]))
                continue
            story.append(Paragraph(block.replace("\n", "<br/>"), styles["Body"]))
            story.append(Spacer(1, 6))

        # Render alternative recommendations.
        story.append(Spacer(1, 10))
        story.append(Paragraph("Recommended Papers", styles["SectionTitle"]))
        for block in self._split_blocks(state.get("recommended_papers", "")):
            if self._is_heading(block):
                story.append(Paragraph(self._normalize_heading(block), styles["SectionTitle"]))
                continue
            story.append(Paragraph(block.replace("\n", "<br/>"), styles["Body"]))
            story.append(Spacer(1, 6))

        # Finalize and write PDF to disk.
        document.build(story)
        return {"formatted_pdf_path": str(output_path)}

    def _build_graph(self):
        """Define and compile the LangGraph workflow DAG."""
        graph = StateGraph(PaperState)

        # Register node functions.
        graph.add_node("extract_text", self._extract_text)
        graph.add_node("extract_insights", self._insights_extractor)
        graph.add_node("simplify_by_level", self._simplifier)
        graph.add_node("topic_fit_and_recommendations", self._topic_fit_and_recommendations)
        graph.add_node("format_pdf", self._build_pdf)

        # Wire end-to-end execution order.
        graph.add_edge(START, "extract_text")
        graph.add_edge("extract_text", "extract_insights")
        graph.add_edge("extract_insights", "simplify_by_level")
        graph.add_edge("simplify_by_level", "topic_fit_and_recommendations")
        graph.add_edge("topic_fit_and_recommendations", "format_pdf")
        graph.add_edge("format_pdf", END)

        # Compile graph to an executable object.
        return graph.compile()

    def run(self, pdf_path: str, level: Level, output_dir: str = "outputs/reports", user_topic: str = "") -> PaperState:
        """Run the full workflow for a single uploaded PDF."""
        # Seed state with required inputs only; downstream nodes enrich it.
        initial_state: PaperState = {
            "pdf_path": pdf_path,
            "level": level,
            "output_dir": output_dir,
            "user_topic": user_topic.strip(),
        }
        return self.graph.invoke(initial_state)


def build_assistant_from_env() -> ResearchPaperAssistant:
    """Construct assistant instance from environment variables."""
    # Core provider credentials and optional model/base URL overrides.
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    groq_model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    tavily_base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com")
    return ResearchPaperAssistant(
        groq_api_key=groq_api_key,
        tavily_api_key=tavily_api_key,
        groq_model=groq_model,
        groq_base_url=groq_base_url,
        tavily_base_url=tavily_base_url,
    )
