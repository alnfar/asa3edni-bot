import os
import telebot
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "أهلاً 👋 هذا البوت جاهز لتحليل الأسهم والذهب والعملات.
استخدم:
/price SYMBOL للحصول على السعر
/ai SYMBOL لتحليل ذكي")

@bot.message_handler(commands=['price'])
def price(message):
    try:
        symbol = message.text.split()[1]
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        r = requests.get(url).json()
        current = r.get("c")
        if current:
            bot.reply_to(message, f"💰 سعر {symbol}: {current}")
        else:
            bot.reply_to(message, "⚠️ لا توجد بيانات لهذا الرمز.")
    except Exception as e:
        bot.reply_to(message, f"خطأ: {e}")

@bot.message_handler(commands=['ai'])
def ai_analysis(message):
    try:
        symbol = message.text.split()[1]
        bot.reply_to(message, f"🤖 تحليل ذكي لـ {symbol} (ميزة تجريبية).")
    except:
        bot.reply_to(message, "⚠️ صيغة الأمر: /ai SYMBOL")

print("🤖 Bot running...")
bot.infinity_polling()
