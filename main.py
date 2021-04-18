import logging
import os
import ffmpeg
from telegram.ext import Updater
from telegram.ext import CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import utils

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
    stream_name = update.callback_query.data
    video_dir = 'video'
    files = {
        "stream1" : "F1_1_1_1.ts",
        "stream2" : "F1_2_2_1.ts",
        "stream3" : "F2_2_3_2.ts"
    }
    source_file = files[stream_name]
    processor = None
    utils.emulate_stream(f"{video_dir}/{source_file}", f"{video_dir}/{stream_name}.mp4", processor)
    f = open(f"{video_dir}/{stream_name}.mp4", 'rb')
    context.bot.send_video(chat_id=update.effective_chat.id, supports_streaming=True, video=f)


if __name__ == '__main__':
    updater = Updater(token='1618035678:AAGe4gJWjEp17QQ3_pNTcGBmJEpqszNeFng', use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(CallbackQueryHandler(watch_froth))

    updater.start_polling()