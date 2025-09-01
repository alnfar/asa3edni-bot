import os
import telebot
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "أهلاً 👋\nالبوت جاهز لتحليل الأسهم والذهب والعملات.\n\n"
                          "استخدم:\n"
                          "/price SYMBOL للحصول على السعر\n"
                          "/ai SYMBOL لتحليل المؤشر.")

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
def ai(message):
    try:
        symbol = message.text.split()[1]
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2023-08-01&to=2023-09-01&token={FINNHUB_API_KEY}"
        r = requests.get(url).json()
        headlines = [f"- {x.get('headline')}" for x in r[:5]]
        reply = "\n".join(headlines) if headlines else "❌ لا توجد أخبار متاحة الآن."
        bot.reply_to(message, f"📰 أحدث الأخبار عن {symbol}:\n{reply}")
    except Exception as e:
        bot.reply_to(message, f"خطأ: {e}")

print("✅ Bot is running...")
bot.polling()
