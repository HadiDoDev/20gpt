import io
import logging
import asyncio
import traceback
import html
import json
from tempfile import NamedTemporaryFile
from PIL import Image

from datetime import datetime
import openai

import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    error as telegram_error
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils
import langchain_utils
import eboo_utils

# setup
db = database.Database()
logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """کلیدهای میانبر:
⚪ /new – آغاز گفتگو جدید
⚪ /mode – انتخاب درس
⚪ /retry – تکرار پاسخ سوال قبلی
⚪ /purchase – خرید اعتبار دروس
⚪ /balance – نمایش اعتبار
⚪ /help – راهنما
قبل از پرسیدن سوال، مطمئن شوید که درس مورد نظر را به درستی انتخاب کرده باشید
 سوال خود را کامل بپرسید
 میتوانید سوال خود را تایپ کنید، وویس بفرستید و یا عکس نمونه سوال خود را ارسال کنید
 برای دریافت پاسخ خود صبر کنید، به دلیل دریافت پیام های زیاد ممکن است کمی طول بکشد
"""

HELP_GROUP_CHAT_MESSAGE = """You can add bot to any <b>group chat</b> to help and entertain its participants!

Instructions (see <b>video</b> below):
1. Add the bot to the group chat
2. Make it an <b>admin</b>, so that it can see messages (all other rights can be restricted)
3. You're awesome!

To get a reply from the bot in the chat – @ <b>tag</b> it or <b>reply</b> to its message.
For example: "{bot_username} write a poem about Telegram"
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int) or isinstance(n_used_tokens, float):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)

    # image generation
    if db.get_user_attribute(user.id, "n_generated_images") is None:
        db.set_user_attribute(user.id, "n_generated_images", 0)


async def is_bot_mentioned(update: Update, context: CallbackContext):
     try:
         message = update.message

         if message.chat.type == "private":
             return True

         if message.text is not None and ("@" + context.bot.username) in message.text:
             return True

         if message.reply_to_message is not None:
             if message.reply_to_message.from_user.id == context.bot.id:
                 return True
     except:
         return True
     else:
         return False


async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)


async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def help_group_chat_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update, context, update.message.from_user)
     user_id = update.message.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

     await update.message.reply_text(text, parse_mode=ParseMode.HTML)
     await update.message.reply_video(config.help_group_chat_video_path)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)


async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")

        try:
            # Check user credit
            db.check_if_user_has_credit(user_id, chat_mode, raise_exception=True)

            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            # print("Dialog Messages:", dialog_messages, flush=True)

            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            # print(type(_message), _message, flush=True)
            langchain_instance=langchain_utils.LANGCHAIN("gpt-4-1106-preview")
            answer, n_input_tokens, n_output_tokens, n_first_dialog_messages_removed, cost = langchain_instance(_message, [], chat_mode)
            
            # chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            # if config.enable_message_streaming:
            #     gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            # else:
            #     answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
            #         _message,
            #         dialog_messages=dialog_messages,
            #         chat_mode=chat_mode
            #     )

            #     async def fake_gen():
            #         yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

            #     gen = fake_gen()

            # prev_answer = ""
            # prev_answer = langchain_response

            # async for gen_item in gen:
                # status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

            answer = answer[:4096]  # telegram message limit

            # update only when 100 new symbols are ready
            # if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
            #     continue

            try:
                await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
            except telegram.error.BadRequest as e:
                if str(e).startswith("Message is not modified"):
                    pass
                else:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

            await asyncio.sleep(0.01)  # wait a bit to avoid flooding

            # prev_answer = answer

            # update user data
            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            print("NDM:", new_dialog_message, flush=True)
        
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

            # Update n Used Rials of User
            print("COSTTTTTTTTTTTT:", cost, flush=True)
            db.decrease_user_credit(user_id, cost)

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(message_handle_fn())
        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]


async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)
    
    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)


async def vision_message_handle(update: Update, context: CallbackContext, use_new_dialog_timeout: bool = True):
        # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    # current_model = db.get_user_attribute(user_id, "current_model")

    # if current_model != "gpt-4-vision-preview":
    #     await update.message.reply_text(
    #         "🥲 Images processing is only available for <b>gpt-4-vision-preview</b> model. Please change your settings in /settings",
    #         parse_mode=ParseMode.HTML,
    #     )
    #     return

    # new dialog timeout
    if use_new_dialog_timeout:
        if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and \
            len(await db.get_dialog_messages(user_id)) > 0:
            await db.start_new_dialog(user_id)
            await update.message.reply_text(
                f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅",
                parse_mode=ParseMode.HTML,
            )

    photo = update.message.effective_attachment[-1]
    photo_file = await context.bot.get_file(photo.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await photo_file.download_to_memory(buf)

    # buf.name = "image.jpg"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer
    # Open the image using Pillow
    # image = Image.open(buf)

    # Save the image to a file
    # image.save("media/image.jpg")

    image = NamedTemporaryFile(
        dir='media/',
        prefix=str(user_id)+'_',
        suffix='.jpg',
        delete=False
    )
    image.write(buf.read())
    image.close()

    # in case of CancelledError
    # n_input_tokens, n_output_tokens = 0, 0
    print("In Vision HANDLE!!!!!", image.name, image, '<=filename', flush=True)

    # send placeholder message to user
    placeholder_message = await update.message.reply_text("درحال تبدیل عکس به متن...")

    # send typing action
    await update.message.chat.send_action(action="typing")

    filelink = f"http://51.89.156.250:8095/{image.name.split('/')[-1]}"
    added_image = eboo_utils.addfile(filelink)
    extracted_text = eboo_utils.convert(added_image['FileToken'])

    # Edit placeholder message
    placeholder_message = await context.bot.edit_message_text("درحال استخراج سوال از متن و ارسال پاسخ، لطفا تا دریافت کامل اطلاعات صبر کنید...",
                                                              chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)


    try:
        message = update.message.caption

        # if message is None or len(message) == 0:
        #     await update.message.reply_text(
        #         "🥲 You sent <b>empty message</b>. Please, try again!",
        #         parse_mode=ParseMode.HTML,
        #     )
        #     return
        if message:
            extracted_text = f"{message}\n {extracted_text}"
        
        

        langchain_instance = langchain_utils.LANGCHAIN("gpt-4-0613")
        step_size = 500
        question_list = []
        for i in range(0, len(extracted_text), step_size):
            extracted_question, cost = langchain_instance.parse_text(extracted_text[i:i+step_size])
            question_list.extend(extracted_question)

            # Update used_rials user attr.
            db.decrease_user_credit(user_id, cost)

        # Delete placeholder message
        await context.bot.delete_message(chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

        # print("Question List:", question_list, flush=True)
        for question in question_list:
            placeholder_message = await update.message.reply_text(question)
            # await update.message.chat.send_action(action="typing")
            await message_handle(update, context, message=question)

        # dialog_messages = db.get_dialog_messages(user_id)
        # parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
        #     config.chat_modes[chat_mode]["parse_mode"]
        # ]

        # langchain_instance=langchain_utils.LANGCHAIN(current_model)
        # answer, n_input_tokens, n_output_tokens, n_first_dialog_messages_removed = langchain_instance(_message, [], chat_mode)

        # chatgpt_instance = openai_utils.ChatGPT(model=current_model)
        # if config.enable_message_streaming:
        #     gen = chatgpt_instance.send_vision_message_stream(
        #         message,
        #         dialog_messages=dialog_messages,
        #         image_buffer=buf,
        #         chat_mode=chat_mode,
        #     )
        # else:
        #     (
        #         answer,
        #         (n_input_tokens, n_output_tokens),
        #         n_first_dialog_messages_removed,
        #     ) = await chatgpt_instance.send_vision_message(
        #         message,
        #         dialog_messages=dialog_messages,
        #         image_buffer=buf,
        #         chat_mode=chat_mode,
        #     )

        #     async def fake_gen():
        #         yield "finished", answer, (
        #             n_input_tokens,
        #             n_output_tokens,
        #         ), n_first_dialog_messages_removed

        #     gen = fake_gen()

        # prev_answer = ""
        # async for gen_item in gen:
        #     (
        #         status,
        #         answer,
        #         (n_input_tokens, n_output_tokens),
        #         n_first_dialog_messages_removed,
        #     ) = gen_item

        #     answer = answer[:4096]  # telegram message limit

        #     # update only when 100 new symbols are ready
        #     if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
        #         continue

        #     try:
        #         await context.bot.edit_message_text(
        #             answer,
        #             chat_id=placeholder_message.chat_id,
        #             message_id=placeholder_message.message_id,
        #             parse_mode=parse_mode,
        #         )
        #     except telegram_error.BadRequest as e:
        #         if str(e).startswith("Message is not modified"):
        #             continue
        #         else:
        #             await context.bot.edit_message_text(
        #                 answer,
        #                 chat_id=placeholder_message.chat_id,
        #                 message_id=placeholder_message.message_id,
        #             )

        #     await asyncio.sleep(0.01)  # wait a bit to avoid flooding

        #     prev_answer = answer

        # # update user data
        # new_dialog_message = {
        #     "user": message,
        #     "bot": answer,
        #     "date": datetime.now(),
        # }

        # await db.set_dialog_messages(
        #     user_id, await db.get_dialog_messages(user_id) + [new_dialog_message]
        # )

        # await db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
        
    # except asyncio.CancelledError:
    #     # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
    #     await db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
    #     raise
    except Exception as e:
        error_text = f"Something went wrong during completion. Reason: {e}"
        logger.error(error_text)
        await update.message.reply_text(error_text)
        return


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(message, n_images=config.return_n_generated_images, size=config.image_size)
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    db.set_user_attribute(user_id, "n_generated_images", config.return_n_generated_images + db.get_user_attribute(user_id, "n_generated_images"))

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.reply_text("<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML)


def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
     if await is_previous_message_not_answered_yet(update.callback_query, context): return

     user_id = update.callback_query.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     query = update.callback_query
     await query.answer()

     page_index = int(query.data.split("|")[1])
     if page_index < 0:
         return

     text, reply_markup = get_chat_mode_menu(page_index)
     try:
         await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
     except telegram.error.BadRequest as e:
         if str(e).startswith("Message is not modified"):
             pass


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def _show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_generated_images = db.get_user_attribute(user_id, "n_generated_images")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = config.models["info"]["dalle-2"]["price_per_1_image"] * n_generated_images
    if n_generated_images != 0:
        details_text += f"- DALL·E 2 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars


    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    user_credit = db.get_user_attribute(user_id, "credit")

    details_text = "🏷️ Details:\n"

    total_rials = user_credit['total_rials'] - user_credit['used_rials']


    text = f"You'r total credit: <b>{total_rials:.03f} Rials</b>\n"
    text += f"You'r available chat modes: <b>{user_credit['chat_modes']}</b>\n\n"
    text += details_text
    text += f"Is on trial mode: <b>{user_credit['is_trial']}</b>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")


async def purchase_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    option = query.data

    # ارسال پست با تصویر، لینک و متن
    if option == 'dini8':
        caption = "دینی پایه هشتم"
        photo_url = "static/photos/dini.jpg"
        link_url = "https://google.com"
    elif option == 'dini9':
        caption = "دینی پایه نهم"
        photo_url = "static/photos/dini.jpg"
        link_url = "https://google.com"
    elif option == 'dini10':
        caption = "دینی پایه دهم"
        photo_url = "static/photos/dini.jpg"
        link_url = "https://google.com"
    elif option == 'dastoorzabaan11':
        caption = "دستور زبان پایه یازدهم"
        photo_url = "static/photos/dastoorzabaan.jpg"
        link_url = "https://google.com"
    else:
        # اگر گزینه معتبر نباشد، هیچکاری انجام نده
        return

    keyboard = [
        [InlineKeyboardButton("بازکردن لینک و خرید", url=link_url)],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    with open(photo_url, 'rb') as photo_file:
        await query.message.reply_photo(photo=photo_file, caption=caption, reply_markup=reply_markup)


async def purchase(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message with three inline buttons attached."""
    keyboard = [
        [
            InlineKeyboardButton("دینی: هشتم", callback_data="dini8"),
            InlineKeyboardButton("دینی: نهم", callback_data="dini9"),
            InlineKeyboardButton("دینی: دهم", callback_data="dini10"),
        ],
        [InlineKeyboardButton("دستور زبان فارسی: یازدهم", callback_data="dastoorzabaan11")],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text("لطفا گزینه درس مورد نظر خود را برای خرید انتخاب کنید:", reply_markup=reply_markup)


# تابع برای نمایش منوی کاربری
async def user_menu(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    await update.message.reply_text(f"سلام {user.first_name}! چه کاری برای شما انجام بدهم؟", reply_markup=get_user_menu_markup())


# تابع برای ایجاد محتوای منوی کاربری
def get_user_menu_markup():
    return {
        "keyboard": [["گزینه 1", "گزینه 2"], ["گزینه 3"]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "گفتگوی جدید"),
        BotCommand("/mode", "انتخاب درس"),
        BotCommand("/retry", "تکرار پاسخ سوال قبلی"),
        BotCommand("/purchase", "خرید اعتبار دروس"),
        BotCommand("/balance", "نمایش اعتبار"),
        BotCommand("/help", "راهنما"),
    ])

        # BotCommand("/settings", "Show settings"),
        # BotCommand("/balance", "Show balance"),
        # BotCommand("/menu", "Show subscriptions"),

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids) | filters.Chat(chat_id=group_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    application.add_handler(MessageHandler(filters.PHOTO & user_filter, vision_message_handle))

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))

    application.add_error_handler(error_handle)

    application.add_handler(CommandHandler("purchase", purchase))
    application.add_handler(CallbackQueryHandler(purchase_button))

    application.add_handler(CommandHandler("menu", user_menu))

    # start the bot
    application.run_polling()

if __name__ == "__main__":
    run_bot()