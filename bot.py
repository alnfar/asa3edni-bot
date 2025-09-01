# -*- coding: utf-8 -*-
"""
Telegram bot: price lookup, news, and chart (image) analysis.
Env vars required:
  TELEGRAM_TOKEN  -> BotFather token
  FINNHUB_API_KEY -> finnhub.io API key
  OPENAI_API_KEY  -> OpenAI API key
"""

import os
import time
import json
import requests
import telebot

# ---------- ENV ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN env var")
if not FINNHUB_API_KEY:
    print("⚠️ FINNHUB_API_KEY is missing — price/news commands will be limited.")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is missing — AI image analysis will be disabled.")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# Keep last image URL per chat to use with /ai
LAST_IMG_URL = {}

# Symbols mapping for indices & metals used in price/news
SYMBOL_MAP = {
    # User friendly -> Finnhub price symbol
    "US100": "^NDX",
    "US500": "^GSPC",
    "US30": "^DJI",
    "XAUUSD": "XAUUSD",
}

# Mapping for news (ETFs proxy since indices don't have company news)
NEWS_PROXY = {
    "US100": "QQQ",
    "US500": "SPY",
    "US30": "DIA",
    # metals: use general news
}

WELCOME = (
    "مرحبًا 👋\n"
    "أنا بوت يساعدك في:\n"
    "• جلب السعر الفوري: /price SYMBOL (مثال: /price AAPL أو /price US100)\n"
    "• تحليل صورة الشارت بالذكاء الاصطناعي: أرسل الصورة ثم /ai SYMBOL\n"
    "• أحدث الأخبار للسهم/المؤشر: /news SYMBOL (مثال: /news AAPL أو /news US500)\n\n"
    "المؤشرات المدعومة: US100=NASDAQ100, US500=S&P500, US30=Dow Jones, الذهب=XAUUSD\n"
    "نصيحة: أرسل الشارت أولاً ثم الرمز 👍"
)

def _map_symbol(sym: str) -> str:
    s = sym.strip().upper()
    return SYMBOL_MAP.get(s, s)

def _map_news_symbol(sym: str) -> str:
    s = sym.strip().upper()
    return NEWS_PROXY.get(s, s)

def _get_price(symbol: str):
    """Use Finnhub /quote for price. Returns float or None"""
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("c")  # current price
    except Exception as e:
        return None

def _get_news(symbol: str, limit: int = 5):
    """Fetch latest news using Finnhub. If company symbol -> company-news, else general."""
    out = []
    try:
        sym = _map_news_symbol(symbol)
        # Try company news for tickers like AAPL, SPY, QQQ...
        from datetime import datetime, timedelta
        to = datetime.utcnow().date()
        fr = to - timedelta(days=7)
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={fr}&to={to}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        items = r.json()
        if not isinstance(items, list) or len(items) == 0:
            # fallback: general news
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            items = r.json()
        for x in items[:limit]:
            title = x.get("headline") or x.get("title") or "بدون عنوان"
            src = x.get("source") or ""
            dt = x.get("datetime") or x.get("datetime") or ""
            out.append(f"• {title} — {src}")
    except Exception:
        pass
    return out

def _get_file_url_from_telegram(message):
    """Return public file URL (requires bot token)."""
    try:
        file_id = message.photo[-1].file_id
        f = bot.get_file(file_id)
        # Telegram file URL:
        return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{f.file_path}"
    except Exception:
        return None

def _openai_vision(image_url: str, symbol: str):
    """Call OpenAI Vision to analyze the provided chart image."""
    if not OPENAI_API_KEY:
        return "⚠️ لا يمكن تحليل الصورة: مفتاح OpenAI غير مُضاف."
    try:
        # New SDK (1.x) — chat.completions
        import openai  # type: ignore
        openai.api_key = OPENAI_API_KEY

        prompt = (
            "حلل هذا الشارت كمتداول محترف:\n"
            f"- الرمز: {symbol}\n"
            "- الإطار الزمني إن أمكن من الصورة.\n"
            "- الاتجاه الحالي (صاعد/هابط/عرضي) مع سبب (قمم/قيعان، متوسطات، شموع بارزة).\n"
            "- مناطق دعم/مقاومة قريبة بالأرقام الموجودة على المحاور إن ظهرت.\n"
            "- شموع لافتة (ابتلاع، همر، دوجي...) وأثرها المتوقع.\n"
            "- سيناريو هام: متى تفكّر في الشراء؟ ومتى البيع/التخارج؟ مع إدارة مخاطر مبسطة.\n"
            "اختصر بدون مبالغة واذكر ملاحظة عدم اعتباره نصيحة مالية."
        )

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.2,
        )
        txt = resp["choices"][0]["message"]["content"].strip()
        return txt
    except Exception as e:
        return f"حدث خطأ أثناء تحليل الصورة: {e}"

# ---------- Handlers ----------

@bot.message_handler(commands=["start"])
def start_cmd(message):
    bot.reply_to(message, WELCOME)

@bot.message_handler(commands=["price"])
def price_cmd(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "اكتب بالشكل: /price SYMBOL (مثال: /price AAPL أو /price US500)")
            return
        user_sym = parts[1]
        sym = _map_symbol(user_sym)
        price = _get_price(sym)
        if price:
            bot.reply_to(message, f"سعر {user_sym.upper()}: {price}")
        else:
            bot.reply_to(message, "⚠️ لا توجد بيانات لهذا الرمز/المؤشر الآن.")
    except Exception as e:
        bot.reply_to(message, f"خطأ: {e}")

@bot.message_handler(commands=["news"])
def news_cmd(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "اكتب: /news SYMBOL (مثال: /news AAPL أو /news US100)")
            return
        user_sym = parts[1]
        items = _get_news(user_sym, limit=5)
        if not items:
            bot.reply_to(message, f"📰 أحدث الأخبار عن {user_sym}: لا توجد أخبار متاحة الآن.")
            return
        text = "📰 أحدث الأخبار:\n" + "\n".join(items)
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"خطأ في جلب الأخبار: {e}")

@bot.message_handler(commands=["ai"])
def ai_cmd(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "اكتب: /ai SYMBOL بعد إرسال صورة الشارت. مثال: /ai US100")
            return
        user_sym = parts[1].upper()
        img_url = LAST_IMG_URL.get(message.chat.id)
        if not img_url:
            bot.reply_to(message, "أرسل صورة الشارت أولاً، ثم اكتب /ai SYMBOL.")
            return
        res = _openai_vision(img_url, user_sym)
        bot.reply_to(message, res)
    except Exception as e:
        bot.reply_to(message, f"خطأ في تحليل الصورة: {e}")

@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    url = _get_file_url_from_telegram(message)
    if url:
        LAST_IMG_URL[message.chat.id] = url
        bot.reply_to(message, "تم استلام الصورة ✅\nابعث الرمز الآن (مثال: AAPL أو US100) ثم استخدم /ai SYMBOL.")
    else:
        bot.reply_to(message, "تعذر قراءة الصورة من تيليجرام. حاول مرة أخرى.")

@bot.message_handler(func=lambda m: True)
def fallback(message):
    # Natural fallback: try interpret as symbol for price shortcut
    raw = (message.text or "").strip().upper()
    if 2 <= len(raw) <= 6:
        sym = _map_symbol(raw)
        price = _get_price(sym)
        if price:
            bot.reply_to(message, f"سعر {raw}: {price}\n(لتحليل الشارت: أرسل صورة ثم /ai {raw})")
            return
    bot.reply_to(message, "استخدم /price أو /news أو أرسل صورة الشارت ثم /ai SYMBOL.")

if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(skip_pending=True, timeout=30)
