"""Streamlit UI for the Agentic Equity Analyst."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import streamlit as st

from report_store import format_report_label, get_report, list_reports, save_report

LOADING_MESSAGES = [
    "Researching using Perplexity Sonar...",
    "Pulling SEC filings...",
    "Checking Yahoo Finance...",
    "Checking Kalshi prediction markets...",
    "Running DCF model...",
    "Analyzing market sentiment...",
    "Reviewing analyst ratings...",
    "Calculating moving averages...",
    "Structuring the report...",
    "Synthesizing findings...",
]

st.set_page_config(page_title="Agentic Equity Analyst", layout="wide")

# Hide Streamlit's top-right running animation; define CSS spinner
st.markdown(
    """
    <style>
    [data-testid="stStatusWidget"] { visibility: hidden !important; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .eq-spinner {
        width: 52px; height: 52px;
        border: 5px solid #e0e0e0;
        border-top-color: #1f77b4;
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
        margin: 2rem auto 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agentic Equity Analyst")

tab_run, tab_history, tab_current = st.tabs(["Run Analysis", "Report History", "Current Report"])

# ── Tab 1: Run Analysis ──────────────────────────────────────────────────────
with tab_run:
    st.subheader("Generate a New Report")

    col1, col2, col3 = st.columns(3)
    with col1:
        company = st.text_input("Company Name", placeholder="e.g. Apple")
    with col2:
        ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL")
    with col3:
        year = st.text_input("Fiscal Year", placeholder="e.g. 2026", value="2026")

    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        disabled=not (company.strip() and ticker.strip() and year.strip()),
    )

    if run_clicked:
        from reporting_pipeline import generate_financial_report

        result: dict = {"text": None, "error": None, "done": False}

        def _run_analysis() -> None:
            try:
                result["text"] = generate_financial_report(
                    company=company.strip(),
                    ticker=ticker.strip().upper(),
                    year=year.strip(),
                    launch_ui=False,
                )
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                result["done"] = True

        thread = threading.Thread(target=_run_analysis, daemon=True)
        thread.start()

        spinner_slot = st.empty()
        msg_slot = st.empty()

        i = 0
        while not result["done"]:
            spinner_slot.markdown('<div class="eq-spinner"></div>', unsafe_allow_html=True)
            msg_slot.info(LOADING_MESSAGES[i % len(LOADING_MESSAGES)])
            time.sleep(3)
            i += 1

        spinner_slot.empty()
        msg_slot.empty()
        thread.join()

        if result["error"]:
            st.error(f"Analysis failed: {result['error']}")
            st.stop()

        report_text: str = result["text"]

        report_id = save_report(
            report_text=report_text,
            ticker=ticker.strip(),
            company=company.strip(),
            year=year.strip(),
        )

        st.success(f"Report saved — ID: `{report_id}`")
        st.divider()
        st.markdown(report_text)

# ── Tab 2: Report History ────────────────────────────────────────────────────
with tab_history:
    st.subheader("Past Reports")

    if st.button("Refresh", key="refresh_history"):
        st.rerun()

    reports = list_reports()

    if not reports:
        st.info("No past reports yet. Run an analysis to get started.")
    else:
        selected_meta = st.selectbox(
            "Select a report",
            reports,
            format_func=format_report_label,
        )

        if selected_meta:
            content = get_report(selected_meta["id"])
            if content:
                col_meta, col_dl = st.columns([3, 1])
                with col_meta:
                    st.caption(f"ID: `{selected_meta['id']}`")
                with col_dl:
                    st.download_button(
                        label="Download .txt",
                        data=content,
                        file_name=f"{selected_meta['id']}.txt",
                        mime="text/plain",
                    )
                st.divider()
                st.markdown(content)
            else:
                st.warning("Report content not found.")

# ── Tab 3: Current Report ────────────────────────────────────────────────────
with tab_current:
    st.subheader("Current Report (report.txt)")

    current = Path("report.txt")
    if not current.exists() or not current.read_text(encoding="utf-8").strip():
        st.info("No current report found. Run an analysis to generate one.")
    else:
        content = current.read_text(encoding="utf-8")
        st.download_button(
            label="Download report.txt",
            data=content,
            file_name="report.txt",
            mime="text/plain",
        )
        st.divider()
        st.markdown(content)
