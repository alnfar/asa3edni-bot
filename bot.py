
import os
import io
import base64
import math
import datetime as dt
import logging
import asyncio
from typing import Optional, Tuple, Dict, Any, List

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ------------------------- Config & Logging -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("asa3edni-bot")

if not TELEGRAM_TOKEN:
    log.error("Missing TELEGRAM_TOKEN env var.")
if not FINNHUB_API_KEY:
    log.warning("Missing FINNHUB_API_KEY (analyst & news will be limited).")
if not OPENAI_API_KEY:
    log.warning("Missing OPENAI_API_KEY (AI summaries & image analysis disabled).")

# ------------------------- Helpers -------------------------

SYMBOL_MAP_PRICE = {
    # Indices (use ETFs for stability)
    "US100": "QQQ",  # NASDAQ100 ETF
    "US500": "SPY",  # S&P500 ETF
    "US30": "DIA",   # Dow Jones ETF
    "GOLD": "XAUUSD=X",  # Spot gold. Alternative "GC=F" futures, or ETF GLD.
}

SYMBOL_MAP_ANALYST = {
    "US100": "QQQ",
    "US500": "SPY",
    "US30": "DIA",
    "GOLD": "GLD",
}

def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    return SYMBOL_MAP_PRICE.get(s, s)

def analyst_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    return SYMBOL_MAP_ANALYST.get(s, s)

def yf_history(symbol: str, period="6mo", interval="1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.dropna()
        return df
    except Exception as e:
        log.exception(f"yfinance error: {e}")
        return None

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(window).mean()
    roll_down = pd.Series(loss).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.index = series.index
    return rsi

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_tech(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['RSI14'] = rsi(df['Close'], 14)
    df['ATR14'] = atr(df, 14)
    last = df.iloc[-1]
    trend = "صاعد ⬆️" if last['EMA20'] > last['EMA50'] else "هابط ⬇️"
    signal = "شراء مبدئي" if last['EMA20'] > last['EMA50'] and last['RSI14'] < 70 else (
             "بيع/حذر" if last['EMA20'] < last['EMA50'] and last['RSI14'] > 30 else "محايد")
    entry = float(last['Close'])
    stop = round(entry - 1.5 * float(last['ATR14']), 4) if not np.isnan(last['ATR14']) else None
    tp = round(entry + 2.0 * float(last['ATR14']), 4) if not np.isnan(last['ATR14']) else None
    return {
        "price": float(last['Close']),
        "ema20": float(last['EMA20']),
        "ema50": float(last['EMA50']),
        "rsi14": float(last['RSI14']),
        "atr14": float(last['ATR14'] if not np.isnan(last['ATR14']) else 0),
        "trend": trend,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "tp": tp,
    }

# ------------------------- Finnhub -------------------------

def _fh(path, params=None):
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing")
    base = "https://finnhub.io/api/v1"
    params = params or {}
    params["token"] = FINNHUB_API_KEY
    r = requests.get(f"{base}/{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def finnhub_company_news(symbol: str, days: int = 3) -> List[Dict[str, Any]]:
    """آخر أخبار الشركة (3 أيام افتراضيًا)."""
    to = dt.date.today()
    fr = to - dt.timedelta(days=days)
    try:
        data = _fh("company-news", {"symbol": symbol, "from": fr.isoformat(), "to": to.isoformat()})
        # رجّع أهم 5
        data = sorted(data, key=lambda x: x.get("datetime", 0), reverse=True)[:5]
        return data
    except Exception as e:
        log.warning(f"company-news failed for {symbol}: {e}")
        return []

def finnhub_reco(symbol: str):
    try:
        data = _fh("stock/recommendation", {"symbol": symbol})
        if not data: 
            return None
        last = sorted(data, key=lambda x: (x.get("period","")), reverse=True)[0]
        return {
            "buy": last.get("buy",0),
            "hold": last.get("hold",0),
            "sell": last.get("sell",0),
            "strongBuy": last.get("strongBuy",0),
            "strongSell": last.get("strongSell",0),
            "period": last.get("period","")
        }
    except Exception as e:
        log.warning(f"reco failed for {symbol}: {e}")
        return None

def finnhub_targets(symbol: str):
    try:
        data = _fh("stock/price-target", {"symbol": symbol})
        if not data:
            return None
        return {
            "targetMean": data.get("targetMean"),
            "targetHigh": data.get("targetHigh"),
            "targetLow": data.get("targetLow"),
            "lastUpdated": data.get("lastUpdatedTime")
        }
    except Exception as e:
        log.warning(f"targets failed for {symbol}: {e}")
        return None

# ------------------------- OpenAI -------------------------

def openai_chat(messages: List[Dict[str, Any]], model: str = "gpt-4o-mini", max_tokens: int = 400) -> str:
    if not OPENAI_API_KEY:
        return "لتفعيل الملخص الذكي أضف OPENAI_API_KEY."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"OpenAI response parsing error: {e} | {r.text[:200]}")
        return "تعذّر الحصول على ملخص الذكاء الاصطناعي حالياً."

def ai_summarize(symbol: str, tech: Dict[str, Any], news: List[Dict[str, Any]], analysts: str) -> str:
    news_lines = []
    for n in news[:4]:
        headline = n.get("headline") or n.get("title") or ""
        source = n.get("source", "")
        news_lines.append(f"- {headline} ({source})")
    tech_text = (
        f"السعر: {tech['price']:.2f}, ترند: {tech['trend']}, إشارة: {tech['signal']}, "
        f"RSI14: {tech['rsi14']:.1f}, EMA20: {tech['ema20']:.2f}, EMA50: {tech['ema50']:.2f}, ATR: {tech['atr14']:.3f}. "
        f"مقترح: دخول {tech['entry']:.2f}, وقف {tech['stop']:.2f}، هدف {tech['tp']:.2f}."
    )
    user_text = (
        f"حلّل باختصار {symbol} بلغة عربية مبسطة للجمهور العام.\n"
        f"التحليل الفني: {tech_text}\n"
        f"أهم الأخبار:\n" + ("\n".join(news_lines) if news_lines else "لا توجد أخبار مهمة.") + "\n"
        f"رأي المحللين: {analysts or 'غير متاح'}\n"
        f"اكتب 3-5 نقاط سريعة + خلاصة نهائية (ليست نصيحة استثمارية)."
    )
    messages = [
        {"role": "system", "content": "أنت خبير أسواق مالية تكتب نقاط واضحة ومختصرة بالعربية."},
        {"role": "user", "content": user_text},
    ]
    return openai_chat(messages, model="gpt-4o-mini", max_tokens=350)

def ai_analyze_image(symbol: str, image_b64: str) -> str:
    """يحلل صورة شارت (Base64) مع توجيه عام. يحتاج OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        return "لتفعيل تحليل الصور، أضف OPENAI_API_KEY."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    content = [
        {"type": "text", "text": f"حلّل الشارت المرفق لـ {symbol}. قيّم الاتجاه ومستويات الدعم/المقاومة بإيجاز."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ]
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "أنت خبير تحليل فني. قدّم نقاطًا قصيرة بالعربية."},
            {"role": "user", "content": content}
        ],
        "max_tokens": 300
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    try:
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"OpenAI vision parse error: {e} | {r.text[:200]}")
        return "تعذّر تحليل الصورة حالياً."

# ------------------------- Telegram Handlers -------------------------

HELP_TEXT = (
"أهلاً 👋\n"
"أوامر سريعة:\n"
"• /ai SYM — ملخص ذكي (فني + أخبار + محللين). مثال: /ai AAPL أو /ai US100\n"
"• /news SYM — آخر الأخبار فقط (من Finnhub)\n"
"• ابعث صورة شارت + بعدها اكتب الرمز (مثال: AAPL) ثم استعمل /ai AAPL\n"
"\nالمؤشرات المدعومة: US100→QQQ, US500→SPY, US30→DIA, GOLD→XAUUSD= X (سعر) & GLD (محللين)\n"
"⚠️ ليس نصيحة استثمارية."
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def news_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استعمل: /news AAPL")
        return
    user_sym = context.args[0].upper()
    sym = analyst_symbol(user_sym)
    items = finnhub_company_news(sym, days=3) if FINNHUB_API_KEY else []
    if not items:
        await update.message.reply_text("لا توجد أخبار حالياً أو مفتاح Finnhub مفقود.")
        return
    lines = []
    for it in items[:5]:
        headline = it.get("headline") or it.get("title","")
        src = it.get("source","")
        url = it.get("url","")
        lines.append(f"• {headline} ({src})\n{url}")
    await update.message.reply_text("\n\n".join(lines))

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """استقبال صورة وتخزينها Base64 مؤقتاً للمستخدم."""
    photo = update.message.photo[-1]  # أعلى دقة
    file = await photo.get_file()
    bio = await file.download_as_bytearray()
    b64 = base64.b64encode(bio).decode("utf-8")
    # خزّنها في user_data
    context.user_data['last_image_b64'] = b64
    await update.message.reply_text("تم استلام الصورة ✅\nابعث الرمز الآن (مثال: AAPL أو US100) ثم استخدم /ai SYMBOL.")

def analyst_text_for(symbol: str) -> str:
    sym = analyst_symbol(symbol)
    parts = []
    reco = finnhub_reco(sym) if FINNHUB_API_KEY else None
    tgt = finnhub_targets(sym) if FINNHUB_API_KEY else None
    if reco:
        parts.append(f"({reco['period']}) StrongBuy {reco['strongBuy']}, Buy {reco['buy']}, Hold {reco['hold']}, Sell {reco['sell']}, StrongSell {reco['strongSell']}")
    if tgt and tgt.get("targetMean"):
        parts.append(f"الأهداف: متوسط {tgt['targetMean']}, أعلى {tgt['targetHigh']}, أدنى {tgt['targetLow']}")
    return "\n".join(parts)

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استعمل: /ai AAPL\nويمكنك إرسال صورة الشارت قبل الأمر للحصول على قراءة بصرية.")
        return
    user_sym = context.args[0].upper()
    sym_price = normalize_symbol(user_sym)
    df = yf_history(sym_price, period="6mo", interval="1d")
    if df is None:
        await update.message.reply_text("لا توجد بيانات للسهم/المؤشر المطلوب.")
        return
    tech = compute_tech(df)

    news = finnhub_company_news(analyst_symbol(user_sym), days=3) if FINNHUB_API_KEY else []
    analysts = analyst_text_for(user_sym) if FINNHUB_API_KEY else ""

    ai_summary = ai_summarize(user_sym, tech, news, analysts) if OPENAI_API_KEY else "لتفعيل الملخص الذكي، أضف OPENAI_API_KEY."
    # إن وُجدت صورة محفوظة لهذا المستخدم، حللها
    img_note = ""
    b64 = context.user_data.get('last_image_b64')
    if b64 and OPENAI_API_KEY:
        img_note = "\n\n🖼️ تحليل الشارت:\n" + ai_analyze_image(user_sym, b64)[:1200]

    # رسالة نهائية
    txt = (
        f"📊 {user_sym}\n"
        f"السعر الحالي: {tech['price']:.2f}\n"
        f"الترند: {tech['trend']} | الإشارة: {tech['signal']}\n"
        f"RSI14: {tech['rsi14']:.1f} | EMA20: {tech['ema20']:.2f} | EMA50: {tech['ema50']:.2f}\n"
        f"مقترح: دخول {tech['entry']:.2f} | وقف {tech['stop']:.2f} | هدف {tech['tp']:.2f}\n"
        f"\n🧠 ملخص الذكاء الاصطناعي:\n{ai_summary}"
        f"\n\n📰 أهم الأخبار ({len(news)}): " + ("لا يوجد" if not news else "") +
        "".join([f"\n• {n.get('headline','')} ({n.get('source','')})" for n in news[:4]]) +
        (f"\n\n👥 رأي المحللين:\n{analysts}" if analysts else "") +
        img_note +
        "\n\n⚠️ هذا المحتوى للتثقيف وليس نصيحة استثمارية."
    )
    await update.message.reply_text(txt)

# ------------------------- Main -------------------------
def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_TOKEN غير موجود.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("news", news_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    log.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
