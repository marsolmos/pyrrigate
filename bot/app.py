#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler
from telegram.ext import MessageHandler, Filters, ConversationHandler
from telegram.ext import CallbackQueryHandler, CallbackContext


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_KEY = '1983590680:AAFNogAyTokWlrEws_GMq_opNR_ysW1Xi0M'

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext) -> None:
    """Sends initial message with three inline buttons attached."""
    keyboard = [
        [
            InlineKeyboardButton("What plant is this?", callback_data='1'),
            InlineKeyboardButton("Recommended cares for plant species", callback_data='2'),
        ],
        [InlineKeyboardButton("Help", callback_data='3')],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text(
        'Hi! Welcome to Planta-Bit Bot. How can I help you?',
        reply_markup=reply_markup
        )


def button(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    selected_option = query.data
    if selected_option == '1':
        query.edit_message_text(
            text="Alright! First I will need an image of your plant. Could you please send me one?",
            )

    elif selected_option == '2':
        query.edit_message_text(
            text="Awesome!",
            )
        plant_cares()

    elif selected_option == '3':
        help_command(update, context)

    elif selected_option == '4':
        start()

    else:
        query.edit_message_text(text=f"There was an unexpected error. Please, try again by typing /start")


def plant_cares():
    update.message.reply_text("Please, give me the name of your plant species")


def text_message(update, context):
    """Send message in case user sends a text message without a command."""
    update.message.reply_text(
        'Hmmmm... I am not sure what you mean. '
        'Please, use /start to use this bot or /help to see user guide',
        )


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def help_command(update: Update, context: CallbackContext) -> None:
    """Displays info on how to use the bot."""
    update.message.reply_text("Use /start to test this bot.")


def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(API_KEY)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # On different commands - answeer in Telegram
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CallbackQueryHandler(button))
    dp.add_handler(CommandHandler('help', help_command))

    # On non-command i.e. message - return special message
    dp.add_handler(MessageHandler(Filters.text, text_message))

    # Log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
