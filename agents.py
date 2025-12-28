"""
indian_stock_agents.py
Demonstration multi-agent system for Indian stocks using Phidata (Agno).
- PriceAgent: yfinance
- FinancialAgent: basic fundamentals via yfinance.info
- NewsAgent: Phidata Agent w/ DuckDuckGo tool
- SentimentAgent: simple VADER on news snippets (fallback)
- RegimeAgent: simple volatility-based regime detector
- FusionAgent: Phidata Agent that explains/fuses signals
- Memory: sqlite for storing predictions + outcomes
"""

import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
import yfinance as yf
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from phi.agent import Agent, RunResponse

# from phi.model.openai import OpenAIChat  # requires OPENAI_API_KEY if used
# from phi.model.ollama import Ollama
from phi.model.google import Gemini

# from phi.tools.duckduckgo import DuckDuckGo

# from phi.storage.agent.sqlite import SqlAgentStorage

from dotenv import load_dotenv
load_dotenv()
import os
print("OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))
print("GOOGLE_API_KEY present:", bool(os.getenv("GOOGLE_API_KEY")))

# ---------- Configuration ----------
DB_FILE = "agent_memory.db"
PERSIST_TABLE = "predictions"   # our lightweight memory table
PHIDATA_SESSION_TABLE = "agent_sessions"

# Example Indian tickers (NSE suffix .NS). Replace/add tickers you care about.
TICKERS = ["INDIGO.NS", "BHEL.NS", "SBISILVER.NS"]
# TICKERS = ["INDIGO.NS"]


# ---------- Programmatic Agents ----------
def get_price_signals(ticker: str) -> Dict[str, Any]:
    """Return simple momentum & volatility signals using yfinance."""
    # yfinance uses e.g. "TCS.NS" for NSE
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty:
        return {"price_momentum_30d": 0.0, "volatility_30d": 0.0}

    # 30 day momentum
    close = df["Close"]
    momentum_30 = (close.iloc[-1] / close.shift(30).iloc[-1]) - 1 if len(close) > 30 else 0.0
    # 30 day historical volatility (std of daily returns * sqrt(252))
    daily_ret = close.pct_change().dropna()
    vol_30 = daily_ret.rolling(window=30).std().dropna()
    vol = float(vol_30.iloc[-1]) * (252 ** 0.5) if len(vol_30) > 0 else float(daily_ret.std() * (252 ** 0.5))

    return {"price_momentum_30d": float(momentum_30), "volatility_30d": float(vol)}


def get_fundamental_signals(ticker: str) -> Dict[str, Any]:
    """
    Very simple fundamental checks via yfinance.info (best-effort).
    For production use a reliable fundamentals provider / XBRL parser.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    # We use sample fields if available; these are heuristics only.
    revenue_growth = info.get("revenueGrowth") or info.get("revenueGrowthTTM") or 0.0
    debt_to_equity = info.get("debtToEquity") or info.get("debtToEquityRatio") or 0.0
    pe = info.get("trailingPE") or info.get("forwardPE") or None

    score = 0.0
    if revenue_growth and revenue_growth > 0.05:  # >5% growth
        score += 0.4
    if debt_to_equity and debt_to_equity < 1.0:
        score += 0.3
    if pe and pe < 30:
        score += 0.2

    return {"revenue_growth": revenue_growth, "debt_to_equity": debt_to_equity, "pe": pe, "fundamental_score": float(score)}


# ---------- Phidata Agents (text-based) ----------
def create_news_agent(model=None) -> Agent:
    """
    Phidata Agent that searches recent news for an Indian stock and summarizes sentiment.
    Uses the built-in DuckDuckGo tool.
    """
    # If model is None, fallback to a minimal LLM declaration — update as you like.
    if model is None:
        # model=Ollama(id="llama3")  # requires OPENAI_API_KEY; change if you use another provider
        model = Gemini(
            id="models/gemini-2.5-pro",
            temperature=0.2
        )

    # Use DuckDuckGo tool to fetch news links and let the LLM summarize + extract sentiment.
    agent = Agent(
        model=model,
        # tools=[DuckDuckGo()],
        description="News agent: Fetch recent news for the given stock ticker (India)",
        instructions=[
            "Search for the recent news items about the given ticker",
            # "Summarize the main themes in 10 sentences.",
            # "Return a numeric sentiment score between -1 (very negative) and 1 (very positive).",
            # "Return a JSON-like output: {'summary': '...', 'sentiment': 0.12}"
        ],
        markdown=False,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
    )
    return agent


def create_fusion_agent(model=None) -> Agent:
    """
    Phidata Agent to fuse signals and return decision + explanation.
    It receives a structured prompt with signals and must return:
      {"final_score": 0.34, "decision": "BUY"/"HOLD"/"SELL", "explanation": "..."}
    """
    if model is None:
        # model=Ollama(id="llama3")
        model = Gemini(
            id="models/gemini-2.5-pro",
            temperature=0.1
        )

    instructions = [
        "You are a quantitative analyst combining signals: price_momentum_30d, volatility_30d, fundamental_score, news_sentiment, regime.",
        "Produce a final_score in [-1, 1] where >0.2 indicates bullish, < -0.2 indicates bearish.",
        "Return a JSON with final_score, decision (BUY/HOLD/SELL), confidence (0-1), and short explanation.",
    ]

    agent = Agent(
        model=model,
        description="Signal fusion & explanation agent",
        instructions=instructions,
        markdown=False,
        show_tool_calls=False,
    )
    return agent


# ---------- Simple Regime Detector ----------
def regime_detector(volatility: float) -> str:
    """
    Very simple regime detector:
      - volatility < 0.15 -> 'low_vol'
      - volatility between 0.15 and 0.4 -> 'normal'
      - volatility > 0.4 -> 'high_vol'
    (Numbers are illustrative; re-tune to Indian market.)
    """
    if volatility < 0.15:
        return "low_vol"
    if volatility < 0.4:
        return "normal"
    return "high_vol"


# # ---------- Simple Sentiment combining (fallback if LLM not used) ----------
# vader = SentimentIntensityAnalyzer()

# def aggregate_news_sentiment(snippets: List[str]) -> float:
#     if not snippets:
#         return 0.0
#     scores = [vader.polarity_scores(s)["compound"] for s in snippets]
#     return float(sum(scores) / len(scores))


# ---------- Runner / Orchestration ----------
def main(tickers: List[str]):


    # Create Phidata agents (you can reuse the same model instance or change per agent)
    news_agent = create_news_agent()
    fusion_agent = create_fusion_agent()

    horizon_days = 30  # predict 30-day outcome by default

    all_recs = []
    for ticker in tickers:
        print(f"\n---- Processing {ticker} ----")

        # 1) Price signals
        price_signals = get_price_signals(ticker)
        print("price_signals:", price_signals)

        # 2) Fundamentals
        fund_signals = get_fundamental_signals(ticker)
        print("fund_signals:", fund_signals)

        # 3) Regime
        regime = regime_detector(price_signals["volatility_30d"])
        print("regime:", regime)

        # 4) News via Phidata agent (LLM + DuckDuckGo)
        # The agent expects a text prompt; we'll pass the ticker symbol + context.
        # Note: agent.run returns a RunResponse object; we parse content.
        try:
            rr: RunResponse = news_agent.run(f"{ticker} India stock recent news on bloomberg, economic times, business standards, stocktwitsindia, moneycontrol, reuters, livemint, times of india, NSE, BSE, CNBC, NDTV market, pulse by zeroda for past 30 days; return json with summary and sentiment.")
            content = rr.content or rr.messages or ""
            # Simple heuristic: try to parse numeric sentiment if returned
            # For this demo we'll fallback to VADER if parsing fails.
            # Usually the Agent returns a structured output—adjust to your model's output formatting.
            news_sentiment = None
            summary_text = ""
            if isinstance(content, dict):
                news_sentiment = content.get("sentiment")
                summary_text = content.get("summary", "")
            else:
                # fallback: use latest headlines fetched by DuckDuckGo tool usage in logs is complex,
                # so fallback to empty summary and 0 sentiment
                news_sentiment = None
                summary_text = ""
            #print("news_sentiment:", news_sentiment)
            print(content)
        except Exception as e:
            print("News agent failed:", e)
            news_sentiment = None
            summary_text = ""

        # # If LLM didn't produce numeric sentiment, fall back to VADER on summary.
        # if news_sentiment is None:
        #     # As fallback, try retrieving a few headlines with duckduckgo-search directly:
        #     from duckduckgo_search import DDGS
        #     headlines = []
        #     try:
        #         with DDGS() as ddgs:
        #             q = f"{ticker} NSE India stock news"
        #             for r in ddgs.news(q, max_results=5):
        #                 headlines.append(r.get("title") or r.get("body") or "")
        #             print(headlines)
        #     except Exception:
        #         print("Error in fetching news headlines")
        #         headlines = []
        #     news_sentiment = aggregate_news_sentiment(headlines)
        #     summary_text = " | ".join(headlines[:3])

        print("news_sentiment:", news_sentiment)
        # 5) Compose signals dict
        signals = {
            "price_momentum_30d": price_signals["price_momentum_30d"],
            "volatility_30d": price_signals["volatility_30d"],
            "fundamental_score": fund_signals["fundamental_score"],
            "news_sentiment": float(news_sentiment)if news_sentiment is not None else 0.0,
            "regime": regime,
        }

        # 6) Fusion via Phidata agent to get final decision + explanation
        fusion_prompt = f"""
        Signals for {ticker}:
        {signals}

        Based on these numeric signals, produce a JSON:
        {{ "final_score": <float -1..1>, "decision": "BUY"/"HOLD"/"SELL", "confidence": 0.0-1.0, "explanation":"..." }}
        """
        try:
            rr2: RunResponse = fusion_agent.run(fusion_prompt)
            fusion_content = rr2.content
            # rr2.content may be a string; try basic extraction
            # For demo: ask model to return a line containing a JSON-like dict.
            # Production: use structured output feature of Phidata (Structured Output).
            import re, ast
            text = ""
            if isinstance(fusion_content, str):
                text = fusion_content
            else:
                text = str(fusion_content)

            # Try to find JSON-like substring
            m = re.search(r"\{.*\}", text, flags=re.S)
            parsed = {}
            if m:
                try:
                    parsed = ast.literal_eval(m.group(0))
                except Exception:
                    parsed = {}
            final_score = float(parsed.get("final_score", 0.0))
            decision = parsed.get("decision", "HOLD")
            confidence = float(parsed.get("confidence", 0.5))
            explanation = parsed.get("explanation", "")
        except Exception as e:
            print("Fusion agent failed, using rule-based fusion. Error:", e)
            # Simple rule-based fusion fallback:
            score = 0.0
            score += 0.6 * signals["price_momentum_30d"]
            score += 0.25 * signals["fundamental_score"]
            score += 0.15 * signals["news_sentiment"]
            final_score = max(min(score, 1.0), -1.0)
            if final_score > 0.25:
                decision = "BUY"
            elif final_score < -0.25:
                decision = "SELL"
            else:
                decision = "HOLD"
            confidence = abs(final_score)
            explanation = "Rule-based fallback fusion."

        rec = {
            "ticker": ticker,
            "created_at": datetime.utcnow().isoformat(),
            "horizon_days": horizon_days,
            "prediction": decision,
            "confidence": confidence,
            "signals": signals,
            "final_score": final_score,
            "explanation": explanation,
            "summary_text": summary_text,
        }


        print("Recommendation:", decision, "score:", final_score, "confidence:", confidence)
        print("explanation:", explanation)
        # small throttle for APIs
        time.sleep(1.0)

    # Print all recommendations
    print("\n=== BATCH RECOMMENDATIONS ===")
    for r in all_recs:
        print(r["ticker"], r["prediction"], round(r["final_score"], 3), f"(conf {r['confidence']:.2f})")


if __name__ == "__main__":
    main(TICKERS)
