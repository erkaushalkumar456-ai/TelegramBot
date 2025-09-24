import os
import logging
from datetime import datetime, timedelta
import asyncio
from io import BytesIO
import base64

# Third-party libraries
import openai
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

# Improved logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("TelegramBot")  # Use a custom logger name

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

MAX_PDF_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
CONTEXT_TIMEOUT_MINUTES = 10

user_context = {}


async def get_user_language(update: Update) -> str:
    if update.message and update.message.from_user:
        return update.message.from_user.language_code or "en"
    return "en"


def get_translation(lang: str, key: str) -> str:
    translations = {
        "en": {
            "welcome": "Welcome! I'm an AI assistant. I can help you with text, images, and PDFs. How can I assist you?",
            "help": (
                "Here's what I can do:\n\n"
                "- *Text Messages*: Chat with me in any language.\n"
                "- *Image Uploads*: Send an image, and I'll analyze it.\n"
                "- *PDF Uploads*: Send a PDF, and I'll answer questions about it.\n\n"
                "Use /reset to clear our conversation history."
            ),
            "reset": "Our conversation history has been cleared.",
            "pdf_received": "Thank you for the PDF. I'm processing it now. Please ask me a question about its content.",
            "pdf_too_large": f"The PDF file is too large. Please send a file smaller than {MAX_PDF_SIZE / 1024 / 1024} MB.",
            "image_too_large": f"The image file is too large. Please send a file smaller than {MAX_IMAGE_SIZE / 1024 / 1024} MB.",
            "unsupported_file": "Sorry, I only support PDF and image files.",
            "error_processing": "Sorry, I encountered an error while processing your request. Please try again.",
            "thinking": "Thinking...",
            "image_received": "I've received your image. What would you like to know about it?",
        },
    }
    return translations.get(lang, translations["en"]).get(key, "")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user_context[user_id] = {
        "history": [],
        "last_seen": datetime.now(),
        "pdf_text": None,
        "image_data": None,
    }
    lang = await get_user_language(update)
    await update.message.reply_text(get_translation(lang, "welcome"))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = await get_user_language(update)
    await update.message.reply_text(
        get_translation(lang, "help"), parse_mode=ParseMode.MARKDOWN
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id in user_context:
        user_context[user_id] = {
            "history": [],
            "last_seen": datetime.now(),
            "pdf_text": None,
            "image_data": None,
        }
    lang = await get_user_language(update)
    await update.message.reply_text(get_translation(lang, "reset"))


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    lang = await get_user_language(update)

    if user_id in user_context and (
        datetime.now() - user_context[user_id]["last_seen"]
    ) > timedelta(minutes=CONTEXT_TIMEOUT_MINUTES):
        await reset_command(update, context)

    if user_id not in user_context:
        user_context[user_id] = {
            "history": [], "last_seen": datetime.now(), "pdf_text": None, "image_data": None
        }

    user_context[user_id]["last_seen"] = datetime.now()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    prompt = update.message.text
    history = user_context[user_id]["history"]

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(history)
    
    if user_context[user_id]["image_data"]:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_context[user_id]['image_data']}"}},
            ],
        })
        user_context[user_id]["image_data"] = None 
    elif user_context[user_id]["pdf_text"]:
        pdf_context = f"Use the following content from a PDF to answer: {user_context[user_id]['pdf_text'][:4000]}"
        messages.append({"role": "system", "content": pdf_context})
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append({"role": "user", "content": prompt})

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create, 
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        full_response = ""
        message = await update.message.reply_text("...")
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id, message_id=message.message_id, text=full_response
                )

        user_context[user_id]["history"].append({"role": "user", "content": prompt})
        user_context[user_id]["history"].append({"role": "assistant", "content": full_response})

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await update.message.reply_text(get_translation(lang, "error_processing"))


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    lang = await get_user_language(update)
    document = update.message.document

    if document.mime_type != "application/pdf":
        await update.message.reply_text(get_translation(lang, "unsupported_file"))
        return

    if document.file_size > MAX_PDF_SIZE:
        await update.message.reply_text(get_translation(lang, "pdf_too_large"))
        return

    try:
        file = await context.bot.get_file(document.file_id)
        file_bytes = await file.download_as_bytearray()
        pdf_text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
        if user_id not in user_context:
            user_context[user_id] = {"history": [], "last_seen": datetime.now(), "pdf_text": None, "image_data": None}
        user_context[user_id]["pdf_text"] = pdf_text
        user_context[user_id]["image_data"] = None
        await update.message.reply_text(get_translation(lang, "pdf_received"))
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        await update.message.reply_text(get_translation(lang, "error_processing"))


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    lang = await get_user_language(update)
    photo = update.message.photo[-1]

    if photo.file_size > MAX_IMAGE_SIZE:
        await update.message.reply_text(get_translation(lang, "image_too_large"))
        return

    try:
        file = await context.bot.get_file(photo.file_id)
        file_bytes = await file.download_as_bytearray()
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")
        if user_id not in user_context:
            user_context[user_id] = {"history": [], "last_seen": datetime.now(), "pdf_text": None, "image_data": None}
        user_context[user_id]["image_data"] = image_base64
        user_context[user_id]["pdf_text"] = None
        await update.message.reply_text(get_translation(lang, "image_received"))
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text(get_translation(lang, "error_processing"))


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Get help on how to use the bot"),
        BotCommand("reset", "Reset the conversation"),
    ])

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
        logger.error("API keys not found. Please set them in the .env file.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.run_polling()


if __name__ == "__main__":
    main()

