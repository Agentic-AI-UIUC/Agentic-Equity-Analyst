"""Generate the Markdown 'Market Regime Analysis' section for equity reports."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from market_regime import Regime, RegimeAssessment

logger = logging.getLogger(__name__)

_REGIME_EMOJI = {
    Regime.STRONG_BULL: "\U0001f402 Strongly Bullish",   # ox
    Regime.BULL: "\U0001f402 Moderately Bullish",
    Regime.NEUTRAL: "\u2696\ufe0f Neutral",
    Regime.BEAR: "\U0001f43b Moderately Bearish",
    Regime.STRONG_BEAR: "\U0001f43b Strongly Bearish",
}

_DIRECTION_ICON = {
    "bullish": "\U0001f7e2 Bullish",
    "neutral": "\U0001f7e1 Neutral",
    "bearish": "\U0001f534 Bearish",
}

# Sector-sensitivity heuristics (sector name -> brief note on macro exposure).
_SECTOR_NOTES: dict[str, str] = {
    "technology": (
        "Technology companies are sensitive to interest-rate expectations "
        "(higher rates compress growth multiples) and GDP growth (which drives "
        "enterprise and consumer spending on tech products)."
    ),
    "financials": (
        "Financials benefit from a steeper yield curve and moderate growth, "
        "but face headwinds from recession risk and credit deterioration."
    ),
    "healthcare": (
        "Healthcare is generally defensive, but biotech sub-sectors can be "
        "sensitive to risk appetite and rate expectations."
    ),
    "consumer discretionary": (
        "Consumer discretionary names are closely tied to GDP growth and "
        "consumer confidence; recession risk is a key headwind."
    ),
    "consumer staples": (
        "Consumer staples tend to be defensive, outperforming in bearish "
        "regimes and underperforming when risk appetite is strong."
    ),
    "energy": (
        "Energy companies are influenced by global growth expectations "
        "and inflation trends — higher inflation often supports commodity prices."
    ),
    "industrials": (
        "Industrials are cyclical and closely correlated with GDP growth "
        "expectations and infrastructure spending."
    ),
    "real estate": (
        "Real estate is highly rate-sensitive; Fed rate expectations and "
        "inflation directly impact cap rates and financing costs."
    ),
    "utilities": (
        "Utilities are defensive, rate-sensitive plays that tend to "
        "outperform in risk-off environments."
    ),
    "materials": (
        "Materials companies are cyclical, benefiting from strong GDP "
        "growth and sometimes from inflationary pricing power."
    ),
    "communication services": (
        "Communication services span defensive telecoms and growth-oriented "
        "digital media — macro sensitivity varies by sub-sector."
    ),
}


def _sector_commentary(sector: str | None, assessment: RegimeAssessment) -> str:
    """Return 1-2 sentences of sector-specific macro commentary."""
    if not sector:
        return ""
    note = _SECTOR_NOTES.get(sector.lower(), "")
    if note:
        return f"\n\n{note}"
    return ""


def _implications_paragraphs(
    assessment: RegimeAssessment,
    company_name: str,
    company_ticker: str,
    sector: str | None,
) -> str:
    """Build the 'Implications for {Company}' prose."""
    regime = assessment.regime
    signals = assessment.signals

    # Gather key data points for prose
    recession_sig = next((s for s in signals if "recession" in s.name.lower()), None)
    gdp_sig = next((s for s in signals if "gdp" in s.name.lower()), None)
    fed_sig = next((s for s in signals if "fed" in s.name.lower()), None)
    cpi_sig = next((s for s in signals if "inflation" in s.name.lower() or "cpi" in s.name.lower()), None)

    parts: list[str] = []

    # Recession + growth paragraph
    if recession_sig or gdp_sig:
        lines = []
        if recession_sig:
            pct = f"{recession_sig.implied_probability:.0%}"
            if recession_sig.signal_direction == "bullish":
                lines.append(
                    f"With recession probability sitting at only {pct}, "
                    f"the macro backdrop is supportive for {company_name}'s business."
                )
            elif recession_sig.signal_direction == "bearish":
                lines.append(
                    f"Recession probability at {pct} represents a meaningful headwind "
                    f"for {company_name} and warrants caution."
                )
            else:
                lines.append(
                    f"Recession probability at {pct} is moderate — not a tailwind but "
                    f"not yet a significant drag on {company_name}."
                )
        if gdp_sig:
            lines.append(gdp_sig.explanation)
        parts.append(" ".join(lines))

    # Rate + inflation paragraph
    if fed_sig or cpi_sig:
        lines = []
        if fed_sig:
            lines.append(fed_sig.explanation)
        if cpi_sig:
            lines.append(cpi_sig.explanation)
        parts.append(" ".join(lines))

    # Sector note
    sector_note = _sector_commentary(sector, assessment)
    if sector_note:
        parts.append(sector_note.strip())

    # Overall takeaway
    label = _REGIME_EMOJI.get(regime, str(regime.value))
    if regime in (Regime.STRONG_BULL, Regime.BULL):
        parts.append(
            f"On balance, prediction markets suggest a {label.split(maxsplit=1)[1].lower()} "
            f"environment, which should be a net positive for {company_name} ({company_ticker})."
        )
    elif regime in (Regime.STRONG_BEAR, Regime.BEAR):
        parts.append(
            f"On balance, prediction markets point to a {label.split(maxsplit=1)[1].lower()} "
            f"environment, presenting macro headwinds for {company_name} ({company_ticker})."
        )
    else:
        parts.append(
            f"Prediction markets paint a mixed picture — {company_name} ({company_ticker}) "
            f"should be evaluated primarily on company-specific fundamentals in this environment."
        )

    return "\n\n".join(parts)


def generate_regime_section(
    regime_assessment: RegimeAssessment,
    company_ticker: str,
    company_name: str,
    sector: str | None = None,
) -> str:
    """Return a Markdown string for the 'Market Regime Analysis' report section.

    Parameters
    ----------
    regime_assessment : RegimeAssessment
        Output of ``market_regime.assess_regime()``.
    company_ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    company_name : str
        Human-readable company name.
    sector : str, optional
        GICS sector for tailored commentary.

    Returns
    -------
    str
        Complete Markdown section ready to append to a report.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    regime_label = _REGIME_EMOJI.get(regime_assessment.regime, regime_assessment.regime.value)

    # -- Header
    lines: list[str] = [
        "",
        "## Market Regime Analysis",
        "",
        f"**Data sourced from [Kalshi](https://kalshi.com) prediction markets as of {now_utc}**",
        "",
    ]

    # -- Current regime
    lines.append(f"### Current Regime: {regime_label}")
    lines.append(regime_assessment.summary)
    lines.append("")

    # -- Signals table
    if regime_assessment.signals:
        lines.append("### Key Market Signals")
        lines.append("")
        lines.append("| Signal | Kalshi Market | Implied Prob. | Direction |")
        lines.append("|--------|--------------|---------------|-----------|")
        for sig in regime_assessment.signals:
            icon = _DIRECTION_ICON.get(sig.signal_direction, sig.signal_direction)
            prob_str = f"{sig.implied_probability:.0%}"
            lines.append(f"| {sig.name} | {sig.kalshi_ticker} | {prob_str} | {icon} |")
        lines.append("")

    # -- Implications
    implications = _implications_paragraphs(
        regime_assessment, company_name, company_ticker, sector,
    )
    lines.append(f"### Implications for {company_name} ({company_ticker})")
    lines.append(implications)
    lines.append("")

    # -- Methodology note
    lines.append("### Methodology Note")
    lines.append(
        "This regime assessment is derived from Kalshi prediction market prices, which "
        "reflect the aggregated probabilistic views of market participants. These are "
        "indicators of market sentiment, not forecasts or guarantees. "
        f"Data was retrieved on {now_utc}."
    )
    lines.append("")

    return "\n".join(lines)
