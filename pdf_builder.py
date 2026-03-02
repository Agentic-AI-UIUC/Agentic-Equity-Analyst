"""Utilities for persisting generated reports and launching the Streamlit viewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

from langchain.tools import tool
from streamlit.web import cli as stcli

APP_PATH = "streamlit_app.py"

def report(
    content: str,
    *,
    file_path: Union[str, Path] = "report.txt",
    launch_ui: bool = False,
) -> Path:
    """Write the report content to disk and optionally launch the Streamlit UI."""

    path = Path(file_path)
    path.write_text(content, encoding="utf-8")

    if launch_ui:
        run_streamlit_app(APP_PATH)

    return path


def run_streamlit_app(app_path: str) -> None:
    """Run a Streamlit app programmatically using Streamlit's internal CLI."""

    original_argv = sys.argv.copy()

    try:
        sys.argv = ["streamlit", "run", app_path]
        stcli.main()
    finally:
        sys.argv = original_argv


@tool
def report_tool(content: str) -> Path:
    """LangChain tool wrapper for persisting reports (UI launch disabled)."""

    return report(content, launch_ui=False)
