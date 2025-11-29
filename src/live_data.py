import yfinance as yf

def get_live_stock_data(symbol: str) -> str:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    data = ticker.history(
    period="1y",        
    interval="1d"       
    )

    # convert dict to readable text
    text = "\n".join([f"{k}: {v}" for k, v in info.items()])
    return text,data


