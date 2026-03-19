"""Streamlit entrypoint for the Research Paper Assistant application.

This module is intentionally UI-focused:
- It loads environment variables.
- It initializes output folders.
- It collects user inputs from the Streamlit interface.
- It executes the LangGraph assistant pipeline.
- It renders generated results and exposes the report as a downloadable PDF.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import Level, build_assistant_from_env


# Load values from a local .env file (if present) into process environment.
load_dotenv()

# Folder where uploaded source PDFs are persisted.
UPLOAD_DIR = Path("outputs/uploads")
# Folder where generated report PDFs are written.
REPORT_DIR = Path("outputs/reports")
# Ensure both folders exist on application startup.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_assistant():
    """Create and cache the assistant instance.

    Streamlit reruns scripts frequently on every interaction. This cache avoids
    rebuilding the LangGraph workflow object and re-reading environment-backed
    model settings on every widget update.
    """
    return build_assistant_from_env()


def save_uploaded_pdf(uploaded_file) -> Path:
    """Persist an uploaded PDF to disk and return its path.

    Args:
        uploaded_file: A Streamlit UploadedFile object from st.file_uploader.

    Returns:
        Path to the saved file under outputs/uploads.
    """
    # Basic path sanitization so nested or absolute-like names cannot escape the
    # intended upload directory.
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    target_path = UPLOAD_DIR / safe_name
    # Streamlit stores uploaded bytes in memory; write them to a file for the
    # downstream pipeline that expects a filesystem path.
    with open(target_path, "wb") as target:
        target.write(uploaded_file.getbuffer())
    return target_path


def main() -> None:
    """Render the Streamlit app and orchestrate one end-to-end run."""
    # Global page metadata shown by Streamlit in browser tab and layout.
    st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
    st.title("Research Paper Assistant")
    st.caption("LangGraph multi-agent workflow: insights extraction -> level-based simplification -> polished PDF report")

    # GROQ_API_KEY is mandatory for all LLM stages; hard-stop if missing.
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Add it to your .env file before running the app.")
        st.stop()
    # Tavily is used for recommendation search. Missing key is non-fatal because
    # the app can still produce paper insights/simplification.
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("TAVILY_API_KEY is not set. Topic-fit recommendations may be limited.")

    # File picker restricted to PDF uploads.
    uploaded_pdf = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    # Free-text topic that drives relevance scoring and alternative paper search.
    user_topic = st.text_input(
        "What topic are you interested in?",
        placeholder="e.g., retrieval-augmented generation for healthcare, graph neural networks, climate modeling",
        help="Used to assess whether the uploaded paper matches your interest and to recommend better-fit papers.",
    )

    level: Level = st.selectbox(
        "Select difficulty level",
        options=["beginner", "intermediate", "advanced"],
        help="Controls wording complexity and depth of concepts.",
    )

    # Keep the primary action disabled until required inputs are available.
    run_button = st.button(
        "Generate Simplified Research Report",
        type="primary",
        disabled=uploaded_pdf is None or not user_topic.strip(),
    )

    # Execute pipeline only when user explicitly clicks and a file exists.
    if run_button and uploaded_pdf is not None:
        # Save uploaded content to disk so backend can parse it via pypdf.
        saved_pdf_path = save_uploaded_pdf(uploaded_pdf)

        # Show a spinner for the whole multi-step workflow.
        with st.spinner("Running LangGraph agents and building your report PDF..."):
            try:
                # Fetch cached assistant and run end-to-end workflow.
                assistant = get_assistant()
                result = assistant.run(
                    pdf_path=str(saved_pdf_path),
                    level=level,
                    output_dir=str(REPORT_DIR),
                    user_topic=user_topic,
                )
            except Exception as exc:
                # Show full traceback in Streamlit for easier debugging.
                st.exception(exc)
                st.stop()

        # Surface all generated artifacts in separate UI sections.
        st.success("Report generated successfully")

        st.subheader("Important Insights")
        st.markdown(result.get("key_insights", "No insights generated."))

        st.subheader(f"Simplified ({level.capitalize()})")
        st.markdown(result.get("simplified_content", "No simplified content generated."))

        st.subheader("Is This Paper a Good Fit for Your Interest?")
        st.markdown(result.get("fit_assessment", "No fit assessment generated."))

        st.subheader("Recommended Papers")
        st.markdown(result.get("recommended_papers", "No recommendations generated."))

        # If backend built a PDF, expose a download button with binary payload.
        generated_pdf_path = result.get("formatted_pdf_path")
        if generated_pdf_path and Path(generated_pdf_path).exists():
            with open(generated_pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download formatted PDF",
                    data=pdf_file.read(),
                    file_name=Path(generated_pdf_path).name,
                    mime="application/pdf",
                )


if __name__ == "__main__":
    # Local script entrypoint.
    main()
