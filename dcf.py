import os, re, math
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from typing import List, Dict, Any
import yfinance as yf

#pratibaa was here lol. just checking to make sure this thing is connected to main or whatevs. HI person reading this!
# ---------------------- ENV + EMBEDDINGS ---------------------- #
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

_COLLECTION = None

def get_parser_collection():
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    
    db = os.getenv("CHROMADB")
    api_key = os.getenv("CHROMADB_API_KEY")
    tenant = os.getenv("CHROMADB_TENANT")
    
    if not all([db, api_key, tenant]):
        logging.warning("ChromaDB credentials missing. Vector search in dcf.py will be unavailable.")
        return None

    from langchain_chroma import Chroma
    _COLLECTION = Chroma(
        database=db,
        collection_name="parser_data",
        embedding_function=embeddings,
        chroma_cloud_api_key=api_key,
        tenant=tenant,
    )
    return _COLLECTION


# ---------------------- DCF FUNCTION ---------------------- #
def calculate_dcf(
    free_cash_flows: List[float],
    discount_rate: float,
    terminal_growth_rate: float,
    current_price: float,
    shares_outstanding: float = None
) -> Dict[str, float]:
    """
    Calculate intrinsic value using the Discounted Cash Flow model.

    If shares_outstanding is provided, returns per-share valuation.
    """
    n = len(free_cash_flows)

    # Step 1: Discount each year's FCF
    discounted_fcf = [
        fcf / ((1 + discount_rate) ** (i + 1))
        for i, fcf in enumerate(free_cash_flows)
    ]

    # Step 2: Compute terminal value
    terminal_value = (
        free_cash_flows[-1] * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)

    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** n)
    intrinsic_value_total = sum(discounted_fcf) + discounted_terminal_value

    # Step 3: Adjust for per-share valuation
    if shares_outstanding and shares_outstanding > 0:
        intrinsic_value = intrinsic_value_total / shares_outstanding
    else:
        intrinsic_value = intrinsic_value_total

    undervaluation_percent = ((intrinsic_value - current_price) / current_price) * 100

    return {
        "intrinsic_value": round(intrinsic_value, 2),
        "current_price": round(current_price, 2),
        "undervaluation_percent": round(undervaluation_percent, 2),
        "terminal_value": round(terminal_value, 2),
        "shares_outstanding": shares_outstanding,
    }



# ---------------------- HELPERS ---------------------- #
def extract_number_with_unit(text: str, context: str = "") -> List[float]:
    """
    Extract numbers with unit detection and contextual filtering.
    Example: '$27.9 billion' -> 27900000000.0
    """
    # Capture the original text span so we can check for $ directly
    pattern = r"(\$?\d+(?:\.\d+)?)\s*(billion|million|thousand|%)?"
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    values = []

    for match in matches:
        num_str, unit = match.groups()

        # Skip years (e.g., 2020–2030)
        if len(num_str) == 4 and num_str.startswith(("19", "20")):
            continue

        # --- 1️⃣ Handle price: must have a $ in front of this specific match
        if "price" in context.lower():
            if not num_str.strip().startswith("$"):
                continue  # reject 51 million, 14, etc.

        # --- 2️⃣ Convert to numeric
        num = float(num_str.replace("$", ""))

        # --- 3️⃣ Apply units
        if unit:
            unit = unit.lower()
            if "billion" in unit:
                num *= 1e9
            elif "million" in unit:
                num *= 1e6
            elif "thousand" in unit:
                num *= 1e3
            elif "%" in unit:
                if "growth" in context.lower() or "discount" in context.lower():
                    num = num / 100
                else:
                    continue
        else:
            # context-specific small filters
            if "cash flow" in context.lower() and num < 1e8:
                continue
            if "growth" in context.lower() and num > 1:
                num = num / 100
            if "discount" in context.lower() and num > 1:
                num = num / 100

        values.append(num)

    return values

def query_chunks(company: str, year: str, query: str, k: int = 5) -> str:
    """Query relevant text from parser_data."""
    collection = get_parser_collection()
    if not collection:
        return ""
    results = collection.similarity_search(f"{company} {year} {query}", k=k)
    return " ".join([r.page_content for r in results])

# ---------------------- NEW: YAHOO-BASED DCF HELPER ---------------------- #
def get_dcf_inputs_from_yahoo(ticker: str, years: int = 5) -> Dict:
    """
    Build DCF input parameters using only Yahoo Finance data:
      - Free cash flows (last `years` from cashflow statement)
      - Current price
      - Terminal growth rate (FCF CAGR)
      - Discount rate (CAPM-style from beta)
      - Shares outstanding
    """
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    # Current price
    current_price = info.get("currentPrice")
    if current_price is None:
        hist = yf_ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not determine current price for {ticker}")
        current_price = float(hist["Close"].iloc[-1])

    # Shares outstanding
    shares_outstanding = info.get("sharesOutstanding")

    # Free cash flows from cashflow statement
    cashflow = yf_ticker.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError(f"No cash flow statement available for {ticker}")

    fcf_row = None
    for name in ["Free Cash Flow", "FreeCashFlow", "FreeCashFlowToFirm"]:
        if name in cashflow.index:
            fcf_row = cashflow.loc[name]
            break

    if fcf_row is None:
        raise ValueError(f"Could not find Free Cash Flow row for {ticker}")

    fcf_row = fcf_row.dropna().iloc[:years]
    if fcf_row.empty:
        raise ValueError(f"Not enough FCF data for {ticker}")

    free_cash_flows = [float(v) for v in reversed(fcf_row.values)]

    # Growth: FCF CAGR
    if len(free_cash_flows) >= 2 and free_cash_flows[0] > 0:
        first, last = free_cash_flows[0], free_cash_flows[-1]
        n = len(free_cash_flows)
        cagr = (last / first) ** (1 / (n - 1)) - 1
        terminal_growth_rate = max(min(cagr, 0.15), -0.10)
    else:
        terminal_growth_rate = 0.025

    # Discount rate: CAPM-ish from beta
    beta = info.get("beta", None)
    risk_free = 0.04
    market_return = 0.09
    equity_risk_premium = market_return - risk_free

    if beta is not None:
        discount_rate = risk_free + beta * equity_risk_premium
    else:
        discount_rate = 0.09

    # Ensure terminal growth < discount rate
    if terminal_growth_rate >= discount_rate:
        terminal_growth_rate = discount_rate - 0.01

    return {
        "free_cash_flows": free_cash_flows,
        "current_price": current_price,
        "terminal_growth_rate": terminal_growth_rate,
        "discount_rate": discount_rate,
        "shares_outstanding": shares_outstanding,
    }

# ---------------------- MAIN ---------------------- #

def find_dcf(company: str, year: str) -> str:
    symbol_text = query_chunks(company, year, "stock symbol ticker symbol company symbol")
    ticker_match = re.search(r"\b[A-Z]{1,5}\b", symbol_text)
    ticker = ticker_match.group(0) if ticker_match else None
    yahoo_result = None

    if ticker:            
        try:
            yahoo_inputs = get_dcf_inputs_from_yahoo(ticker, years=5)
            yahoo_result = calculate_dcf(
                    free_cash_flows=yahoo_inputs["free_cash_flows"],
                    discount_rate=yahoo_inputs["discount_rate"],
                    terminal_growth_rate=yahoo_inputs["terminal_growth_rate"],
                    current_price=yahoo_inputs["current_price"],
                    shares_outstanding=yahoo_inputs["shares_outstanding"],
                )
            
            return str(yahoo_result)
        except Exception as e:
            return f" Error during Yahoo-based DCF: {e}"
    else:
        return "Skipping Yahoo-based DCF — no ticker symbol found from filings."
    

@tool 
def find_dcf_tool(company: str, year: str):
    """
    Perform a Discounted Cash Flow analysis of a given company in a specific year.  
    Takes two strings as its arguments, first for the company and secondly for the year (formatted like XXXX)
    Returns result in form of string 
    """
    return find_dcf(company, year)

def get_normalized_valuation_score(ticker: str) -> Dict[str, Any]:
    """
    Returns a normalized valuation score (0.0-1.0) using sensitivity analysis and sigmoid mapping.
    Includes anomaly penalties and uncertainty-adjusted confidence.
    """
    try:
        inputs = get_dcf_inputs_from_yahoo(ticker)
        
        # 1. Sensitivity Analysis (3x3 grid)
        base_r = inputs["discount_rate"]
        base_g = inputs["terminal_growth_rate"]
        
        discount_rates = [base_r - 0.01, base_r, base_r + 0.01]
        growth_rates = [base_g - 0.005, base_g, base_g + 0.005]
        
        uv_results = []
        intrinsic_values = []
        
        for r in discount_rates:
            for g in growth_rates:
                # Ensure g < r to avoid division by zero or negative terminal value
                actual_g = min(g, r - 0.005)
                res = calculate_dcf(
                    free_cash_flows=inputs["free_cash_flows"],
                    discount_rate=r,
                    terminal_growth_rate=actual_g,
                    current_price=inputs["current_price"],
                    shares_outstanding=inputs["shares_outstanding"],
                )
                uv_results.append(res["undervaluation_percent"])
                intrinsic_values.append(res["intrinsic_value"])
        
        # 2. Distribution Stats
        uv_pct = np.mean(uv_results)
        std_uv = np.std(uv_results)
        
        # 3. Log-scale Transformation (optional but recommended to dampen extremes)
        # uv_transformed = np.sign(uv_pct) * np.log1p(abs(uv_pct))
        
        # 4. Sigmoid Normalization
        # Map: 0% -> 0.5, +15% -> 0.73, -15% -> 0.27
        normalized = 1 / (1 + math.exp(-uv_pct / 15.0))
        
        # 5. Confidence Adjustment
        # Confidence penalized by uncertainty (std_uv)
        # 1 / (1 + std_uv / 20) * base 0.8
        confidence = (1 / (1 + std_uv / 20.0)) * 0.8
        
        # 6. Anomaly Penalty
        if abs(uv_pct) > 100:
            confidence *= 0.5
            
        return {
            "score": round(normalized, 3),
            "confidence": round(confidence, 3),
            "expected_uv": round(float(uv_pct), 2),
            "uncertainty_std": round(float(std_uv), 2),
            "intrinsic_value_mean": round(float(np.mean(intrinsic_values)), 2),
            "current_price": inputs["current_price"]
        }
    except Exception as e:
        return {"score": 0.5, "confidence": 0.0, "error": str(e)}
