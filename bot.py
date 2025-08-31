import os, io, re, logging, requests, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ========= Config =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ضع_التوكن_هنا")

# OCR (image-only)
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "helloworld")
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# Trading-news providers (set your preferred provider key; priority order: FINNHUB > POLYGON > ALPHAVANTAGE)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")          # https://finnhub.io/ (company-news)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")          # https://polygon.io/ (reference/news)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")# https://www.alphavantage.co/ (news-sentiment)

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ========= OCR helpers =========
TICKER_RE = re.compile(r"\b[A-Z]{1,6}\b")
TIME_RE = re.compile(r"\b(1m|3m|5m|15m|30m|45m|1h|2h|4h|60m|90m|1d|1D|D|1wk|1mo)\b", re.IGNORECASE)
BLACKLIST = {"OPEN","HIGH","LOW","CLOSE","BUY","SELL","CALL","PUT","USD","EUR","GOLD","INDEX","TIME","PRICE","VOLUME"}

def ocr_space_extract_text(image_bytes: bytes) -> str:
    files = {'filename': ('image.jpg', image_bytes)}
    data = {'apikey': OCR_SPACE_API_KEY, 'language': 'eng', 'isOverlayRequired': False}
    r = requests.post(OCR_SPACE_URL, files=files, data=data, timeout=60)
    r.raise_for_status()
    j = r.json()
    if not j.get("IsErroredOnProcessing") and j.get("ParsedResults"):
        return j["ParsedResults"][0].get("ParsedText", "")
    raise ValueError(f"OCR error: {j}")

def extract_ticker_and_tf(text: str):
    up = text.upper()
    tickers = [t for t in TICKER_RE.findall(up) if t not in BLACKLIST]
    tickers = list(dict.fromkeys(tickers))
    m = TIME_RE.search(text)
    tf = (m.group(1).lower() if m else None) or "1d"
    sym = tickers[0] if tickers else None
    return sym, tf

# ========= Technicals =========
def ema(series: pd.Series, span: int) -> pd.Series: return series.ewm(span=span, adjust=False).mean()
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9); return 100 - (100/(1+rs))
def macd(series: pd.Series):
    ema12, ema26 = ema(series,12), ema(series,26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_patterns(df: pd.DataFrame):
    out = []
    o,h,l,c = df["Open"].iloc[-2:], df["High"].iloc[-2:], df["Low"].iloc[-2:], df["Close"].iloc[-2:]
    body = (c - o).abs().iloc[-1]
    rng = (h - l).iloc[-1]
    upper_w = (h - max(c.iloc[-1], o.iloc[-1])).iloc[-1]
    lower_w = (min(c.iloc[-1], o.iloc[-1]) - l).iloc[-1]
    if rng > 0 and body / rng < 0.1: out.append("Doji")
    if lower_w > 2*body and upper_w < body: out.append("Hammer")
    if upper_w > 2*body and lower_w < body: out.append("Shooting Star")
    prev_body = (c.iloc[0] - o.iloc[0]); last_body = (c.iloc[1] - o.iloc[1])
    if last_body>0 and prev_body<0 and c.iloc[1]>o.iloc[0] and o.iloc[1]<c.iloc[0]: out.append("Bullish Engulfing")
    if last_body<0 and prev_body>0 and o.iloc[1]>c.iloc[0] and c.iloc[1]<o.iloc[0]: out.append("Bearish Engulfing")
    return out[:3]

# ========= Data & News =========
def fetch_history(symbol: str, interval: str = "1d", lookback: str = "6mo"):
    lookback = "60d" if interval in ["1h","2h","4h","60m","90m"] else "6mo"
    return yf.download(symbol, period=lookback, interval=interval, progress=False)

def news_from_finnhub(symbol: str, n=3):
    if not FINNHUB_API_KEY: return None
    # company-news needs from/to dates (last 7 days)
    today = dt.date.today()
    frm = today - dt.timedelta(days=7)
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={frm}&to={today}&token={FINNHUB_API_KEY}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    items = r.json()
    if not isinstance(items, list): return None
    items = sorted(items, key=lambda x: x.get("datetime",0), reverse=True)[:n]
    out = []
    for it in items:
        headline = it.get("headline","")
        src = it.get("source","")
        link = it.get("url","")
        out.append(f"• {headline} — {src}\n{link}")
    return "\n".join(out) if out else None

def news_from_polygon(symbol: str, n=3):
    if not POLYGON_API_KEY: return None
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit={n}&apiKey={POLYGON_API_KEY}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    js = r.json()
    results = js.get("results", [])
    out = []
    for it in results[:n]:
        title = it.get("title","")
        src = it.get("publisher",{}).get("name","")
        link = it.get("article_url","")
        out.append(f"• {title} — {src}\n{link}")
    return "\n".join(out) if out else None

def news_from_alphavantage(symbol: str, n=3):
    if not ALPHAVANTAGE_API_KEY: return None
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&sort=LATEST&apikey={ALPHAVANTAGE_API_KEY}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    js = r.json()
    feed = js.get("feed", [])[:n]
    out = []
    for it in feed:
        title = it.get("title","")
        src = it.get("source","")
        link = it.get("url","")
        out.append(f"• {title} — {src}\n{link}")
    return "\n".join(out) if out else None

def fetch_trading_news(symbol: str, n=3):
    # Priority: Finnhub -> Polygon -> AlphaVantage
    for fn in (news_from_finnhub, news_from_polygon, news_from_alphavantage):
        try:
            res = fn(symbol, n) if fn != news_from_alphavantage else fn(symbol, n)
            if res: return res
        except Exception as e:
            logging.warning(f"News provider error: {e}")
    return "لا توجد أخبار حديثة من مزودي الأخبار المخصّصين."

# ========= Chart image =========
def make_chart(df: pd.DataFrame, symbol: str):
    fig = plt.figure(figsize=(8,4.2))
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(df.index, ema(df["Close"],20), label="EMA20")
    plt.plot(df.index, ema(df["Close"],50), label="EMA50")
    plt.title(f"{symbol} – Close & EMAs")
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    return buf

# ========= Core analysis =========
def analyze_symbol(symbol: str, interval: str):
    df = fetch_history(symbol, interval=interval)
    if df.empty: raise ValueError("لا توجد بيانات للسهم/الإطار المختار.")
    df = df.dropna().copy()
    price = float(df["Close"].iloc[-1])
    ema20 = float(ema(df["Close"],20).iloc[-1])
    ema50 = float(ema(df["Close"],50).iloc[-1])
    rsi_v = float(rsi(df["Close"],14).iloc[-1])
    macd_line, signal, hist = macd(df["Close"])
    macd_v, signal_v, hist_v = float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    atr_v = float(atr(df,14).iloc[-1] if not np.isnan(atr(df,14).iloc[-1]) else df["Close"].diff().abs().rolling(14).mean().iloc[-1])
    patterns = detect_patterns(df)

    # Clear public-friendly signal
    if ema20 > ema50 and rsi_v < 70 and macd_v > signal_v:
        signal_txt = "🟢 شراء محتمل"
    elif ema20 < ema50 and rsi_v > 30 and macd_v < signal_v:
        signal_txt = "🔴 بيع محتمل"
    else:
        signal_txt = "🟡 انتظار"

    trend_txt = "📈 صاعد" if ema20>ema50 else ("📉 هابط" if ema20<ema50 else "➡️ جانبي")
    stop = price - 1.5*atr_v if ema20>=ema50 else price + 1.5*atr_v
    target = price + 2.5*atr_v if ema20>=ema50 else price - 2.5*atr_v

    # Liquidity (relative volume)
    vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
    vol_ratio = float(df["Volume"].iloc[-1] / (vol_ma20 + 1e-9)) if not np.isnan(vol_ma20) else 1.0
    liq_txt = "قوية" if vol_ratio>=1.5 else ("متوسطة" if vol_ratio>=0.8 else "ضعيفة")

    text = f"""🔎 تحليل آلي واضح – {symbol}
السعر: {price:.4f}
الاتجاه: {trend_txt}
الإشارة: {signal_txt}
RSI(14): {rsi_v:.1f} | MACD: {macd_v:.3f}/{signal_v:.3f}
📌 شموع: {", ".join(patterns) if patterns else "لا توجد إشارة واضحة"}
💵 السيولة: {liq_txt} (x{vol_ratio:.2f} من متوسط 20 يوم)

دخول افتراضي: قريب من السعر الحالي
🛑 وقف: {stop:.3f}
🎯 هدف: {target:.3f}
⏱️ الإطار: {interval}
(تنبيه: هذا مساعد تداول—not توصية ملزمة)
"""
    chart = make_chart(df, symbol)
    news = fetch_trading_news(symbol, 3)
    return text, chart, news

# ========= Telegram handlers =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("أرسل صورة للتشارت (بدون كتابة). سأستخرج الرمز والإطار تلقائيًا، أحلل الشموع والمؤشرات، وأجيب آخر أخبار من مزوّدات تداول.\nبديل: /stock NVDA 1h")

async def stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /stock NVDA 1h")
        return
    symbol = context.args[0].upper()
    interval = context.args[1] if len(context.args)>=2 else "1d"
    try:
        text, chart, news = analyze_symbol(symbol, interval)
        await update.message.reply_photo(photo=InputFile(chart, filename="chart.png"), caption=text)
        await update.message.reply_text("📰 أخبار تداول:\n" + news)
    except Exception as e:
        await update.message.reply_text(f"خطأ: {e}")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.photo[-1].get_file()
        img_bytes = await file.download_as_bytearray()
        text_raw = ocr_space_extract_text(img_bytes)
        symbol, interval = extract_ticker_and_tf(text_raw)
        if not symbol:
            await update.message.reply_text("ما قدرتش نحدد الرمز من الصورة. حاول بصورة أوضح أو استخدم /stock SYMBOL")
            return
        text, chart, news = analyze_symbol(symbol, interval)
        await update.message.reply_photo(photo=InputFile(chart, filename="chart.png"), caption=f"📷 استخرجت: {symbol} ({interval})\n"+text)
        await update.message.reply_text("📰 أخبار تداول:\n" + news)
    except Exception as e:
        await update.message.reply_text(f"خطأ في تحليل الصورة: {e}")

def main():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "ضع_التوكن_هنا":
        raise SystemExit("⚠️ ضع TELEGRAM_TOKEN في المتغيرات أولًا.")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stock", stock))
    app.add_handler(MessageHandler(filters.PHOTO | (filters.Document.IMAGE), photo_handler))
    app.run_polling()

if __name__ == "__main__":
    main()
