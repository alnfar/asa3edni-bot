import os
import telebot
import requests
from telebot import types

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

symbols_map = {
    "ناسداك": "US100",
    "اس اند بي": "US500",
    "الداو": "US30",
    "الذهب": "XAUUSD",
    "يورو دولار": "EURUSD",
}

@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn1 = types.KeyboardButton("📊 الأسعار")
    btn2 = types.KeyboardButton("📰 الأخبار")
    btn3 = types.KeyboardButton("🤖 تحليل الذكاء")
    btn4 = types.KeyboardButton("🖼️ تحليل صورة")
    markup.add(btn1, btn2, btn3, btn4)
    bot.reply_to(message, "أهلاً 👋 اختر الخدمة:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == "📊 الأسعار")
def price_menu(message):
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    for key in symbols_map:
        markup.add(types.KeyboardButton(key))
    bot.reply_to(message, "اختر الأداة:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text in symbols_map)
def get_price(message):
    symbol = symbols_map[message.text]
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        r = requests.get(url).json()
        current = r.get("c")
        if current:
            bot.reply_to(message, f"💹 السعر الحالي {symbol}: {current}")
        else:
            bot.reply_to(message, "⚠️ لا توجد بيانات متاحة الآن.")
    except Exception as e:
        bot.reply_to(message, f"❌ خطأ: {e}")

@bot.message_handler(func=lambda m: m.text == "📰 الأخبار")
def get_news(message):
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
    try:
        r = requests.get(url).json()[:3]
        news = "\n".join([f"📰 {n['headline']}" for n in r])
        bot.reply_to(message, f"آخر الأخبار:\n{news}")
    except Exception as e:
        bot.reply_to(message, f"❌ خطأ في جلب الأخبار: {e}")

bot.polling()
