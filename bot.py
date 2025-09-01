
# -*- coding: utf-8 -*-
import os
import io
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple

import telebot
from telebot import types

# ----------------------
# Environment & Clients
# ----------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "").strip()
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY", "").strip()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN env var")
if not FINNHUB_API_KEY:
    raise RuntimeError("Missing FINNHUB_API_KEY env var")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

# OpenAI (new SDK)
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to init OpenAI client: {e}")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")

# ----------------------
# Helpers
# ----------------------

# Friendly aliases -> Finnhub symbols
SYMBOL_MAP = {
    "US100": "^NDX",    # Nasdaq-100
    "US500": "^GSPC",   # S&P 500
    "US30":  "^DJI",    # Dow Jones
    "NAS100": "^NDX",
    "NAS100 CASH": "^NDX",
    "NDX": "^NDX",
    "SPX": "^GSPC",
    "DOW": "^DJI",
    "GOLD": "XAUUSD",
    "XAU": "XAUUSD",
    "XAUUSD": "XAUUSD",
}

def normalize_symbol(sym: str) -> str:
    s = sym.upper().strip().replace("/", "").replace("\\", "")
    return SYMBOL_MAP.get(s, s)

def finnhub_quote(symbol: str) -> Optional[dict]:
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    # data keys: c (current), d (change), dp (percent), h, l, o, pc, t
    if not data or "c" not in data or data.get("c") in (0, None):
        return None
    return data

def format_price_card(symbol: str, q: dict) -> str:
    arrow = "🟢" if q.get("d", 0) >= 0 else "🔴"
    lines = [
        f"<b>{symbol}</b>",
        f"{arrow} السعر الحالي: <b>{q.get('c')}</b>",
        f"التغير: {q.get('d')} | النسبة: {q.get('dp')}%",
        f"الافتتاح: {q.get('o')}  الأعلى: {q.get('h')}  الأدنى: {q.get('l')}",
        f"إغلاق أمس: {q.get('pc')}",
    ]
    return "\n".join(lines)

def finnhub_news(symbol: str) -> list:
    # indices may not have company-news; fall back to general news
    end = datetime.utcnow().date()
    start = end - timedelta(days=3)
    url_company = (
        f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
        f"&from={start}&to={end}&token={FINNHUB_API_KEY}"
    )
    news = []
    try:
        r = requests.get(url_company, timeout=15)
        if r.status_code == 200:
            news = [n for n in r.json() if isinstance(n, dict)]
    except Exception:
        news = []
    if not news:
        url_gen = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        try:
            r = requests.get(url_gen, timeout=15)
            if r.status_code == 200:
                news = [n for n in r.json() if isinstance(n, dict)]
        except Exception:
            news = []
    return news[:5]

def summarize_news_for(symbol: str, news: list) -> str:
    if not news:
        return "❌ لا توجد أخبار متاحة الآن."
    # make a compact prompt for OpenAI to summarize headlines
    items = []
    for n in news:
        title = n.get("headline") or n.get("title") or ""
        source = n.get("source") or ""
        dt = n.get("datetime") or n.get("publishedAt") or ""
        url = n.get("url") or ""
        items.append({"title": title, "source": source, "time": str(dt), "url": url})
    try:
        msg = "\n".join([f"- {x['title']} ({x['source']})" for x in items if x['title']])
        completion = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "لخص الأخبار التالية في نقاط عربية مختصرة واذكر الانحياز (صعود/هبوط) إن وجد."},
                {"role": "user", "content": f"الرمز: {symbol}\nالعناوين:\n{msg}"}
            ],
            temperature=0.2,
        )
        summary = completion.choices[0].message.content.strip()
    except Exception:
        summary = "ملخص قصير للأخبار:\n" + "\n".join([f"• {x['title']}" for x in items if x['title']][:5])
    # attach first 3 links (if any)
    links = "\n".join([f"🔗 {x['url']}" for x in items if x['url']][:3])
    return summary + ("\n\n" + links if links else "")

# Simple user state (memory)
USER_MODE = {}  # user_id -> "image" if expecting symbol for last photo

def main_keyboard() -> types.ReplyKeyboardMarkup:
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("📊 السعر", "📰 الأخبار")
    kb.row("🤖 تحليل ذكي", "📷 تحليل صورة")
    kb.row("❓ مساعدة")
    return kb

# ----------------------
# Handlers
# ----------------------

@bot.message_handler(commands=["start", "menu", "help"])
def start_cmd(message: types.Message):
    text = (
        "أهلاً بك 👋\n"
        "اختر من الأزرار أو اكتب:\n"
        "• /price SYMBOL — السعر الحالي (مثال: /price US100)\n"
        "• /ai SYMBOL — تحليل ذكي بالأخبار والسعر (مثال: /ai AAPL)\n"
        "• أرسل صورة الشارت ثم أرسل الرمز (US100، US500، XAUUSD...) لتحليل الصورة.\n"
    )
    bot.reply_to(message, text, reply_markup=main_keyboard())

@bot.message_handler(commands=["price"])
def price_cmd(message: types.Message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "اكتب الرمز بعد الأمر مثل: /price US100")
            return
        sym = normalize_symbol(parts[1])
        q = finnhub_quote(sym)
        if not q:
            bot.reply_to(message, "❌ لا توجد بيانات لهذا الرمز.")
            return
        bot.reply_to(message, format_price_card(sym, q))
    except Exception as e:
        bot.reply_to(message, f"حدث خطأ: {e}")

@bot.message_handler(commands=["ai"])
def ai_cmd(message: types.Message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "اكتب الرمز بعد الأمر مثل: /ai US100")
            return
        sym = normalize_symbol(parts[1])
        q = finnhub_quote(sym)
        news = finnhub_news(sym)
        price_part = format_price_card(sym, q) if q else "لا توجد بيانات سعرية."
        news_summary = summarize_news_for(sym, news)

        # Small AI summary combining
        prompt = (
            f"حلّل التالي للرمز {sym}:\n"
            f"- بيانات السعر:\n{price_part}\n\n"
            f"- ملخص الأخبار:\n{news_summary}\n\n"
            "أعطني خلاصة تداول واضحة (اتجاه، مناطق دعم/مقاومة تقديرية، سيناريو بديل، وتنبيه مخاطر)."
        )
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        analysis = resp.choices[0].message.content.strip()
        bot.reply_to(message, analysis)
    except Exception as e:
        bot.reply_to(message, f"حدث خطأ: {e}")

@bot.message_handler(content_types=["photo"])
def photo_handler(message: types.Message):
    # save last photo and ask for symbol
    USER_MODE[message.from_user.id] = {
        "mode": "await_symbol",
        "file_id": message.photo[-1].file_id
    }
    bot.reply_to(message, "✅ تم استلام الصورة.\nأرسل الرمز الآن (مثال: US100 أو AAPL)، ثم استخدم: /ai SYMBOL.", reply_markup=main_keyboard())

@bot.message_handler(func=lambda m: True)
def generic_handler(message: types.Message):
    txt = (message.text or "").strip()
    uid = message.from_user.id

    # Quick buttons
    if txt == "📊 السعر":
        bot.reply_to(message, "أرسل الأمر هكذا: /price SYMBOL (مثال: /price US100)")
        return
    if txt == "📰 الأخبار":
        bot.reply_to(message, "أرسل الأمر هكذا: /ai SYMBOL وسيتم تضمين الأخبار.")
        return
    if txt == "🤖 تحليل ذكي":
        bot.reply_to(message, "أرسل: /ai SYMBOL (مثال: /ai XAUUSD)")
        return
    if txt == "📷 تحليل صورة":
        bot.reply_to(message, "أرسل صورة الشارت الآن ثم أرسل الرمز.")
        return
    if txt == "❓ مساعدة":
        start_cmd(message)
        return

    # If we are waiting for symbol after a photo
    state = USER_MODE.get(uid)
    if state and state.get("mode") == "await_symbol":
        symbol = normalize_symbol(txt)
        file_id = state.get("file_id")
        try:
            # Download photo from Telegram
            file_info = bot.get_file(file_id)
            data = bot.download_file(file_info.file_path)
            b64_image = "data:image/jpeg;base64," + (data.encode("base64") if hasattr(data, "encode") else __import__("base64").b64encode(data).decode())

            # Vision analysis
            vision_prompt = f"حلّل الشموع والاتجاه للرمز {symbol}. أعطِ دعم/مقاومة، وجهة عامة، وما إذا كان هناك نموذج واضح."
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "أنت محلل فني مختصر ومباشر."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": b64_image}},
                        ],
                    },
                ],
                temperature=0.2,
            )
            analysis = resp.choices[0].message.content.strip()

            # optional: add price card
            q = finnhub_quote(symbol)
            price_txt = format_price_card(symbol, q) if q else ""

            bot.reply_to(message, (price_txt + "\n\n" if price_txt else "") + analysis, reply_markup=main_keyboard())
        except Exception as e:
            bot.reply_to(message, f"خطأ في تحليل الصورة: {e}")
        finally:
            USER_MODE.pop(uid, None)
        return

    # Otherwise, ignore or guide
    if txt.startswith("/"):
        bot.reply_to(message, "الأمر غير معروف. استخدم /start لرؤية الأوامر.")
    else:
        bot.reply_to(message, "أرسل /start لعرض القائمة أو أرسل صورة الشارت ثم الرمز.")

if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(skip_pending=True, timeout=30)
