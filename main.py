"""Command-line entry point for generating equity research reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from reporting_pipeline import generate_financial_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Agentic Equity Analyst pipeline and persist the output report.",
    )
    parser.add_argument("--company", required=True, help="Company name to analyze.")
    parser.add_argument(
        "--ticker",
        help="Optional stock ticker symbol to pass to downstream tools (e.g., AAPL, MSFT).",
    )
    parser.add_argument("--year", required=True, help="Fiscal/forecast year for the outlook (e.g., 2026).")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional custom natural-language prompt. Overrides --company/--year wording.",
    )
    parser.add_argument(
        "--file",
        default="report.txt",
        help="Destination file for the generated report (defaults to report.txt).",
    )
    parser.add_argument(
        "--launch-ui",
        action="store_true",
        help="Launch the Streamlit viewer after the report is written.",
    )
    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="Skip the Market Regime Analysis section (included by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_text = generate_financial_report(
        company=args.company,
        ticker=args.ticker,
        year=args.year,
        custom_prompt=args.prompt,
        file_path=args.file,
        launch_ui=args.launch_ui,
        include_market_regime=not args.no_regime,
    )
    print("\nGenerated report saved to", Path(args.file).resolve())
    print("\nPreview:\n" + report_text[:1000])


if __name__ == "__main__":
    main()
