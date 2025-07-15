from telegram import Bot

def send_telegram_alert(message):
    token = '8119271612:AAHXYHS4Z5aCGBdoaO-H5E5FhX3ScL4gQCc'  # Your bot token
    chat_id = '5426954808'  # Your chat ID

    bot = Bot(token=token)
    bot.send_message(chat_id=chat_id, text=message)
