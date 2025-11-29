# from dotenv import load_dotenv
# import os
# from langchain_groq import ChatGroq
# from src.vectorstore import FaissVectorStore  # your vectorstore module
# from google import genai

# # ------------------------------------------------------
# # Load .env
# # ------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")
# load_dotenv(dotenv_path=ENV_PATH)

# # ------------------------------------------------------
# # Initialize LLM
# # ------------------------------------------------------
# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)
# #llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")

# # ------------------------------------------------------
# # Initialize Vector Store
# # ------------------------------------------------------
# vectorstore = FaissVectorStore(persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2")
# vectorstore.load()  # assumes your FAISS index already exists

# # ------------------------------------------------------
# # Query Vector Store
# # ------------------------------------------------------
# query = "Tell me some of the recent ipo's in the Indian stock market. Ignore the data from the RAG pipeline"
# results = vectorstore.query(query, top_k=3)

# # Extract text context from metadata
# texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
# context = "\n\n".join(texts)

# if not context:
#     print("No relevant context found.")
#     exit()

# # ------------------------------------------------------
# # Build RAG Prompt
# # ------------------------------------------------------
# prompt = f"""
# Use the following context to answer the question concisely.

# Context:
# {context}

# Question: {query}

# Answer:
# """

# # ------------------------------------------------------
# # Send to LLM
# # ------------------------------------------------------
# #resp = llm.invoke(prompt)
# response = model.generate_content(prompt)
# print("\n[ANSWER]\n", response.content)





# from dotenv import load_dotenv
# import os
# from src.vectorstore import FaissVectorStore  # your vectorstore module
# from google import genai

# # ------------------------------------------------------
# # Load .env
# # ------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")
# load_dotenv(dotenv_path=ENV_PATH)

# # ------------------------------------------------------
# # API key (optional: you can also set GEMINI_API_KEY/GOOGLE_API_KEY env var)
# # ------------------------------------------------------
# api_key = os.getenv("GOOGLE_API_KEY")  # or GEMINI_API_KEY
# # Create a client (Gemini Developer API example)
# client = genai.Client(api_key=api_key)

# # ------------------------------------------------------
# # Initialize Vector Store
# # ------------------------------------------------------
# vectorstore = FaissVectorStore(persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2")
# vectorstore.load()  # assumes your FAISS index already exists

# # ------------------------------------------------------
# # Query Vector Store
# # ------------------------------------------------------
# query = "Tell me about GAIL"
# results = vectorstore.query(query, top_k=3)

# # Extract text context from metadata
# texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
# context = "\n\n".join(texts)

# if not context:
#     print("No relevant context found.")
#     client.close()
#     exit()

# # ------------------------------------------------------
# # Build RAG Prompt
# # ------------------------------------------------------
# prompt = f"""
# Use the following context to answer the question concisely.

# Context:
# {context}

# Question: {query}

# Answer:
# """

# # ------------------------------------------------------
# # Send to LLM via the new Client
# # ------------------------------------------------------
# resp = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
# print("\n[ANSWER]\n", getattr(resp, "text", None) or resp)
# # clean up
# client.close()





from dotenv import load_dotenv
import os
from src.vectorstore import FaissVectorStore
from src.live_data import get_live_stock_data   # <-- LIVE MARKET DATA
from google import genai

# ------------------------------------------------------
# Load .env
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# ------------------------------------------------------
# Initialize Gemini Client
# ------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# ------------------------------------------------------
# Initialize Vector Store
# ------------------------------------------------------
vectorstore = FaissVectorStore(persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2")
vectorstore.load()  # assumes FAISS index already built

# ------------------------------------------------------
# Query from the user
# ------------------------------------------------------
query = "You are a data-driven financial analyst trained on market fundamentals, macroeconomic indicators, and valuation metrics. I want you to identify stocks suitable for long-term investment (1 year based on your own analysis, not journal summaries or sentiment data) Use Fundamental analysis to evaluate the identified stocks intrinsic value by examining related economic, financial, and qualitative factors to determine if the stock is currently overvalued or undervalued. Perform a complete fundamental analysis under the following structure: 1. Business Overview •	Describe the company’s business model, revenue sources, and geographic exposure. •	Identify the industry it operates in and its position within the value chain. •	Highlight key growth drivers and strategic priorities. 2. Financial Statement Analysis (Last 10 Years) Analyze using actual financial data from Balance Sheets, Income Statement, Cash Flow statement: •	Revenue Growth: CAGR, cyclicality, diversification of revenue streams. •	Profitability: Gross margin, operating margin, and net margin trends. •	Earnings Growth: EPS trend and volatility. •	Return Metrics: ROE, ROA, ROIC vs peers. •	Ratio Analysis: Profitability ratios, turnover/efficiency ratios, debt ratios, solvency ratios, valuation ratios. 3. Balance Sheet Strength •	Leverage: Debt-to-equity, interest coverage ratio, debt maturity profile. •	Liquidity: Current ratio, quick ratio, cash reserves. •	Book Value: Tangible book value trend and capital structure changes. 4. Cash Flow Analysis •	Operating Cash Flow (OCF): Stability and correlation with net income. •	Free Cash Flow (FCF): Growth trend and reinvestment ratio. •	CapEx and Dividends: CapEx intensity and payout sustainability. 5. Valuation Analysis •	Conduct a relative valuation using key multiples: o	P/E, EV/EBITDA, P/B, P/S vs peers and industry averages. •	Conduct an intrinsic valuation using: o	Discounted Cash Flow (DCF) — estimate fair value range. o	Dividend Discount Model (if applicable). •	Conclude with a valuation summary: undervalued / fairly valued / overvalued. 6. Competitive Moat & Qualitative Factors •	Assess brand strength, pricing power, switching costs, or IP advantage. •	Evaluate management quality and capital allocation track record. •	Identify key risks: regulatory, operational, geopolitical, or technological. 7. Industry & Macroeconomic Outlook •	Analyze sector growth trends, competitive intensity (Porter’s Five Forces). •	Assess macroeconomic influences: inflation, interest rates, FX, and consumer demand. •	Consider global factors (e.g., supply chain, trade exposure). 8. Catalysts & Red Flags •	Identify near-term catalysts (product launches, market expansion, cost optimization). •	Highlight potential red flags (earnings manipulation, unsustainable growth, excessive leverage). 9. Investment Thesis & Recommendation •	Provide a 1–2 year investment outlook: o	Buy / Hold / Sell o	With rationale, valuation target, and major assumptions. Your task is to perform a complete technical analysis. Use both chart-based and indicator-based methods to assess the asset’s short-term and long-term price trends, momentum, and potential trading opportunities: 1. Asset Overview •	Identify the asset’s current price, market capitalization, and trading volume. •	State the timeframe(s) analyzed (e.g., daily, weekly, monthly). •	Mention whether the asset is in an uptrend, downtrend, or consolidation phase. 2. Price Trend Analysis Evaluate the overall direction of price movement using the following: •	Trend Lines: Identify major support and resistance zones. •	Moving Averages: o	20-day, 50-day, and 200-day SMA/EMA. o	Check for golden cross / death cross signals. •	Trend Strength: Use ADX (Average Directional Index) to measure trend momentum. •	Price Structure: Identify higher highs/lows or lower highs/lows for pattern confirmation. 3. Momentum Analysis Analyze buying/selling pressure and short-term momentum: •	Relative Strength Index (RSI): Identify overbought/oversold levels (70/30 rule). •	Stochastic Oscillator: Check for crossover signals and divergences. •	MACD (Moving Average Convergence Divergence): o	Identify signal line crossovers. o	Analyze bullish/bearish divergence with price. 4. Volume & Participation Assess market participation and confirm price movements: •	Volume Trends: Determine whether volume confirms or contradicts price movement. •	On-Balance Volume (OBV): Identify accumulation or distribution. •	Volume-Weighted Average Price (VWAP): Check if current price trades above or below VWAP. •	Money Flow Index (MFI): Analyze volume-weighted buying and selling pressure. 5. Volatility & Risk Analysis Measure market volatility and potential breakout behavior: •	Bollinger Bands: Observe contraction (squeeze) or expansion (volatility increase). •	ATR (Average True Range): Assess daily volatility and stop-loss sizing. •	Implied vs Historical Volatility (for derivatives): Identify market complacency or fear. 6. Chart Pattern Recognition Identify key technical patterns from price action: •	Continuation Patterns: Flags, pennants, ascending/descending triangles. •	Reversal Patterns: Head & shoulders, double tops/bottoms, wedges. •	Candlestick Patterns: Doji, hammer, engulfing, morning/evening star. Provide interpretation of each detected pattern and potential price targets. 7. Support & Resistance Mapping •	Identify short-, medium-, and long-term support/resistance levels. •	Use Fibonacci retracements (23.6%, 38.2%, 50%, 61.8%) for price targets. •	Highlight psychological round numbers (e.g., 100, 5000, etc.). 8. Market Breadth & Sentiment (Optional for Indices or ETFs) •	Put/Call Ratio, Advance-Decline Line, VIX: Analyze market-wide sentiment. •	COT Reports / Futures Data: Determine institutional vs retail positioning. 9. Multi-Timeframe Alignment Compare signals across timeframes: •	Short-term (1D): Entry timing. •	Medium-term (1W): Trend confirmation. •	Long-term (1M): Strategic direction. Assess whether signals align or contradict across these levels. 10. Trade Setup & Recommendation Based on the above analysis, provide: •	Trend Bias: Bullish / Bearish / Neutral. •	Entry Zone: Suggested buy/sell levels. •	Stop-Loss: Based on ATR or support levels. •	Target Price: Based on pattern projection or Fibonacci extension. •	Risk/Reward Ratio and position sizing recommendation. "

# ------------------------------------------------------
# Vector Search
# ------------------------------------------------------
results = vectorstore.query(query, top_k=3)
texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
rag_context = "\n\n".join(texts) if texts else "No vector data found."

# ------------------------------------------------------
# LIVE MARKET DATA (detect stock symbol)
# ------------------------------------------------------
stock_symbols = ["GAIL.NS", "TCS.NS", "RELIANCE.NS"]

live_data = []

for symbol in stock_symbols:
    text, df = get_live_stock_data(symbol)
    live_data.append({"symbol": symbol, "text": text, "data": df})

# ------------------------------------------------------
# Combined Prompt for Gemini
# ------------------------------------------------------
prompt = f"""
You are a financial analysis assistant.

Use BOTH:
1. Vectorstore context (static knowledge)
2. Live Yahoo Finance market data

Do NOT hallucinate. Only use actual data.

---------------------------
📌 User Query:
{query}

---------------------------
📘 Vectorstore Context:
{rag_context}

---------------------------
📈 Live Market Data:
{live_data}

---------------------------
✍️ Final Answer:
Provide a clean, concise answer using BOTH sources.
"""

# ------------------------------------------------------
# Send to Gemini
# ------------------------------------------------------
resp = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt
)

# Print final output
print("\n[ANSWER]\n")
print(resp.text or resp)

client.close()



