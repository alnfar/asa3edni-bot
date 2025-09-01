import os, logging, requests, yfinance as yf, pandas as pd, numpy as np, telebot
from datetime import date, timedelta

# ========= إعداد =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
logging.basicConfig(level=logging.INFO)

ALIASES = {
    "US100": "^NDX", "US500": "^GSPC", "US30": "^DJI",
    "GOLD": "GC=F", "XAUUSD": "XAUUSD=X",
    "SILVER": "SI=F", "XAGUSD": "XAGUSD=X",
    "OIL": "CL=F", "BRENT": "BZ=F"
}
def normalize_symbol(s): return ALIASES.get(s.upper(), s.upper())

def get_yf_price(symbol):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="1d", interval="1m")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception as e:
        logging.warning(f"yf error {e}")
    return None

def compute_ta(symbol):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="30d", interval="1d")
        if df is None or df.empty: return None
        close = df["Close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
        rsi = 100 - (100/(1+rs)).iloc[-1]
        return {"sma20": float(sma20), "rsi14": float(rsi)}
    except Exception as e:
        logging.warning(f"ta error {e}")
    return None

def openai_summary(prompt):
    if not OPENAI_API_KEY: return "⚠️ OpenAI API key missing"
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        data = {"model": "gpt-4o-mini", "messages":[{"role":"user","content":prompt}]}
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data).json()
        return r["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ OpenAI error: {e}"

def get_news(symbol):
    if not FINNHUB_API_KEY: return ["⚠️ Finnhub API key missing"]
    try:
        today = date.today(); start = (today - timedelta(days=7)).isoformat()
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={today.isoformat()}&token={FINNHUB_API_KEY}"
        r = requests.get(url).json()
        if not r: return ["No news available"]
        return [f"- {x.get('headline')}
{x.get('url')}" for x in r[:5]]
    except Exception as e:
        return [f"❌ News error: {e}"]

# ========= أوامر =========
@bot.message_handler(commands=["start","help"])
def start(message):
    bot.reply_to(message, "Welcome!\nCommands:\n/price SYMBOL\n/ai SYMBOL\n/news SYMBOL")

@bot.message_handler(commands=["price"])
def price(message):
    parts = message.text.split()
    if len(parts)<2: return bot.reply_to(message, "Usage: /price SYMBOL")
    symbol = normalize_symbol(parts[1])
    val = get_yf_price(symbol)
    if val: bot.reply_to(message, f"💰 {symbol}: {val}")
    else: bot.reply_to(message, "⚠️ No data")

@bot.message_handler(commands=["ai"])
def ai(message):
    parts = message.text.split()
    if len(parts)<2: return bot.reply_to(message, "Usage: /ai SYMBOL")
    symbol = normalize_symbol(parts[1])
    price = get_yf_price(symbol)
    ta = compute_ta(symbol)
    prompt = f"حلل {symbol}. السعر {price}. المؤشرات {ta}."
    reply = openai_summary(prompt)
    bot.reply_to(message, reply)

@bot.message_handler(commands=["news"])
def news(message):
    parts = message.text.split()
    if len(parts)<2: return bot.reply_to(message, "Usage: /news SYMBOL")
    symbol = normalize_symbol(parts[1])
    items = get_news(symbol)
    bot.reply_to(message, "\n\n".join(items))

if __name__ == "__main__":
    bot.infinity_polling()
