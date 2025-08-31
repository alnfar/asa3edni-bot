# Asa3edni Bot – Full Auto (Trading-news providers)
- Photo-only via OCR.Space (no caption needed)
- Clear public signals: Buy/Sell/Wait + Entry/Stop/Target
- Indicators: EMA20/50, RSI, MACD, ATR, simple candle patterns
- News priority: Finnhub -> Polygon -> Alpha Vantage (set API keys via Variables)

### Deploy (Railway/GitHub)
Files required: bot.py, requirements.txt, Procfile, runtime.txt

Environment Variables:
- TELEGRAM_TOKEN=<BotFather token>  [required]
- OCR_SPACE_API_KEY=helloworld      [or your own OCR key]
- FINNHUB_API_KEY=<optional>        [preferred]
- POLYGON_API_KEY=<optional>
- ALPHAVANTAGE_API_KEY=<optional>

Start command (Procfile):
worker: python bot.py
