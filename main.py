import logging
import os
import ffmpeg
from telegram.ext import Updater
from telegram.ext import CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)



def start(update, context):
    button_list = [
       [InlineKeyboardButton("в камере 1", callback_data="stream1")],
       [InlineKeyboardButton("в камере 2", callback_data="stream2")],
       [InlineKeyboardButton("в камере 3", callback_data="stream3")],
    ]
    reply_markup = InlineKeyboardMarkup(button_list)
    context.bot.send_message(chat_id=update.effective_chat.id, text="Cмотреть на пену", reply_markup=reply_markup)

    # context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


def watch_froth(update, context):
    f = open('/home/nofate/work/private/nnhack-21/bot/video/1.mp4', 'rb')

    context.bot.send_video(chat_id=update.effective_chat.id, supports_streaming=True, video=f)

if __name__ == '__main__':
    updater = Updater(token='1618035678:AAGe4gJWjEp17QQ3_pNTcGBmJEpqszNeFng', use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(CallbackQueryHandler(watch_froth))

    updater.start_polling()