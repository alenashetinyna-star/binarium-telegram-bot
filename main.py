import asyncio
import hmac
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from telegram import Bot, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("binarium-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
USER_COOLDOWN_SECONDS = int(os.getenv("USER_COOLDOWN_SECONDS", "15"))
MAX_DAILY_USER_REQUESTS = int(os.getenv("MAX_DAILY_USER_REQUESTS", "30"))
ADMIN_USER_IDS = {
    int(x.strip()) for x in os.getenv("ADMIN_USER_IDS", "").split(",")
    if x.strip().isdigit()
}

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не заданий. Додайте його в Environment Variables.")
if not PUBLIC_BASE_URL:
    log.warning("PUBLIC_BASE_URL не заданий. Автоматичне встановлення webhook вимкнено.")
if not WEBHOOK_SECRET:
    log.warning("WEBHOOK_SECRET не заданий. Рекомендується встановити випадковий секрет.")

app = FastAPI(title="Binarium Screenshot Analyzer", version="1.0.0")
telegram_app: Application | None = None
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# In-memory rate limiting: resets on restart and is not shared between instances.
last_request: dict[int, float] = {}
daily_requests: dict[int, deque] = defaultdict(deque)


def cv_image_report(image_bytes: bytes) -> dict[str, Any]:
    """Only image-quality/structure checks. This does NOT calculate RSI/MACD from pixels."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"valid": False, "message": "Не вдалося прочитати зображення."}

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
    warnings = []
    if width < 500 or height < 300:
        warnings.append("Низька роздільність: свічки та підписи можуть бути нерозбірливими.")
    if contrast < 22:
        warnings.append("Низький контраст зображення.")
    if brightness < 25 or brightness > 238:
        warnings.append("Зображення надто темне або пересвічене.")
    if edge_density < 0.008:
        warnings.append("Мало помітних деталей; графік може бути обрізаний або нечіткий.")
    return {
        "valid": True,
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "edge_density": round(edge_density, 4),
        "warnings": warnings,
    }


ANALYSIS_PROMPT = """
Ти — обережний помічник з технічного аналізу скриншотів торгових графіків.
Проаналізуй наданий скриншот, але НЕ вигадуй показники, значення, ціни, актив або таймфрейм,
яких неможливо чітко прочитати. Це бінарні опціони з дуже високим ризиком; не обіцяй прибутку,
не називай оцінку ймовірності гарантованою і не радь збільшувати ставку чи відіграватися.

Мета: детально описати видимий тренд, структуру максимумів/мінімумів, підтримку/опір,
свічкові патерни, імпульс, волатильність, а також RSI/MACD/EMA/MA/Bollinger лише якщо вони
справді видимі на скриншоті. Якщо індикатор не показаний або його значення нечітке — напиши це.
Не обчислюй RSI/MACD з одного скриншота ціни. Таймфрейм може бути 1, 2, 3, 5, 10, 15, 30 або
60 хвилин, але вкажи «не вдалося визначити», якщо доказів немає.

Поверни ВИКЛЮЧНО коректний JSON з ключами:
{
 "asset": "назва активу або Не визначено",
 "timeframe": "визначений таймфрейм або Не визначено",
 "trend": "висхідний/низхідний/боковий/неясний",
 "signal": "CALL" або "PUT" або "НЕ ВХОДИТИ",
 "signal_strength": "низька/помірна/відносно висока/немає",
 "expiry_minutes": число від 1 до 60 або null,
 "summary": "короткий висновок",
 "evidence_for": ["спостереження, що підтримують напрямок"],
 "evidence_against": ["суперечливі ознаки та ризики"],
 "indicators": [{"name":"...", "observation":"...", "status":"видно/не видно/нечитабельно"}],
 "levels": {"support":"видимий рівень або невідомо","resistance":"видимий рівень або невідомо"},
 "chart_quality": "добра/посередня/погана",
 "no_trade_reasons": ["причини не входити"],
 "caveat": "коротке попередження про невизначеність"
}

Правила сигналу:
- Якщо скриншот розмитий, обрізаний, актив/таймфрейм неясні, мало свічок, ознаки суперечливі,
  індикатори не підтверджують рух або немає чіткого сетапу — signal = "НЕ ВХОДИТИ".
- Не вигадуй експірацію: пропонуй її лише як умовний горизонт аналізу, коли структура достатньо зрозуміла.
- Для дуже коротких таймфреймів будь особливо консервативним.
- signal_strength — якісна оцінка узгодженості видимих ознак, НЕ математична ймовірність виграшу.
- Не стверджуй, що майбутній рух можна достовірно передбачити за одним скриншотом.
Відповідай українською.
"""


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Модель не повернула JSON.")
    return json.loads(cleaned[start:end + 1])


async def analyze_with_gemini(image_bytes: bytes, mime_type: str, cv_report: dict) -> dict[str, Any]:
    if genai_client is None:
        return {
            "signal": "НЕ ВХОДИТИ",
            "signal_strength": "немає",
            "summary": "AI-аналіз недоступний: не задано GEMINI_API_KEY.",
            "asset": "Не визначено",
            "timeframe": "Не визначено",
            "trend": "неясний",
            "expiry_minutes": None,
            "evidence_for": [],
            "evidence_against": ["Візуальні перевірки OpenCV не замінюють аналіз ринкових даних."],
            "indicators": [],
            "levels": {"support": "невідомо", "resistance": "невідомо"},
            "chart_quality": "посередня",
            "no_trade_reasons": ["Не налаштовано AI-модель; сигнал не формується."],
            "caveat": "Не відкривайте угоду на основі цього результату.",
        }

    prompt = ANALYSIS_PROMPT + "\nДодаткові технічні характеристики зображення (не ринкові індикатори): " + json.dumps(cv_report, ensure_ascii=False)
    response = await genai_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
    )
    data = parse_json_response(response.text or "")
    if data.get("signal") not in {"CALL", "PUT", "НЕ ВХОДИТИ"}:
        data["signal"] = "НЕ ВХОДИТИ"
    try:
        expiry = data.get("expiry_minutes")
        if expiry is not None:
            expiry = int(expiry)
            if not 1 <= expiry <= 60:
                expiry = None
        data["expiry_minutes"] = expiry
    except (ValueError, TypeError):
        data["expiry_minutes"] = None
    if cv_report.get("warnings"):
        data.setdefault("evidence_against", []).extend(cv_report["warnings"])
        # Conservative override for weak screenshots.
        if cv_report.get("width", 0) < 500 or cv_report.get("height", 0) < 300:
            data["signal"] = "НЕ ВХОДИТИ"
            data["signal_strength"] = "немає"
            data["no_trade_reasons"] = list(data.get("no_trade_reasons", [])) + [
                "Зображення має низьку роздільність."
            ]
    return data


def format_analysis(data: dict[str, Any], cv_report: dict[str, Any]) -> str:
    signal = data.get("signal", "НЕ ВХОДИТИ")
    emoji = {"CALL": "🟢", "PUT": "🔴", "НЕ ВХОДИТИ": "🟡"}.get(signal, "🟡")
    expiry = data.get("expiry_minutes")
    expiry_text = f"{expiry} хв (умовний горизонт, не гарантія)" if expiry else "не рекомендовано / не визначено"
    lines = [
        "📊 *BINARIUM — АНАЛІЗ СКРИНШОТА*",
        "",
        f"{emoji} *Висновок: {signal}*",
        f"*Актив:* {data.get('asset', 'Не визначено')}",
        f"*Таймфрейм:* {data.get('timeframe', 'Не визначено')}",
        f"*Видимий тренд:* {data.get('trend', 'неясний')}",
        f"*Узгодженість ознак:* {data.get('signal_strength', 'немає')} (це НЕ ймовірність виграшу)",
        f"*Експірація:* {expiry_text}",
        "",
        "*Короткий висновок*",
        str(data.get("summary", "Недостатньо даних.")),
        "",
        "*Аргументи за напрямок*",
    ]
    lines += [f"• {x}" for x in (data.get("evidence_for") or ["Немає достатніх підтверджень."])[:6]]
    lines += ["", "*Ризики та аргументи проти*"]
    lines += [f"• {x}" for x in (data.get("evidence_against") or ["Не виявлено / не визначено."])[:6]]
    lines += ["", "*Індикатори*"]
    indicators = data.get("indicators") or []
    if indicators:
        for item in indicators[:8]:
            lines.append(f"• *{item.get('name', 'Індикатор')}:* {item.get('observation', 'невідомо')} [{item.get('status', 'не визначено')}]")
    else:
        lines.append("• На скриншоті не вдалося надійно прочитати індикатори.")
    levels = data.get("levels") or {}
    lines += [
        "",
        "*Рівні*",
        f"• Підтримка: {levels.get('support', 'невідомо')}",
        f"• Опір: {levels.get('resistance', 'невідомо')}",
        "",
        f"*Якість графіка:* {data.get('chart_quality', 'не визначено')}",
    ]
    reasons = data.get("no_trade_reasons") or []
    if reasons:
        lines += ["", "*Причини не входити*"] + [f"• {x}" for x in reasons[:6]]
    warnings = cv_report.get("warnings") or []
    if warnings:
        lines += ["", "*Перевірка зображення*"] + [f"• {x}" for x in warnings[:4]]
    lines += [
        "",
        f"_{data.get('caveat', 'Ринок непередбачуваний; результат не гарантований.')}_",
        "",
        "⚠️ Це інформаційний технічний аналіз, не фінансова порада. Бінарні опціони можуть призвести до втрати всієї суми ставки. Не ризикуйте коштами, які не можете дозволити собі втратити.",
    ]
    # Telegram Markdown legacy escaping is intentionally avoided for model-generated text;
    # parse_mode=None is used by send_message.
    # Plain text avoids parse errors caused by arbitrary model-generated characters.
    return "\n".join(lines).replace("*", "").replace("_", "")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Вітаю! Я аналізую скриншоти торгових графіків і пояснюю видимі технічні ознаки.\n\n"
        "📷 Надішліть скриншот графіка Binarium як фото або файл-зображення.\n"
        "Підтримуваний діапазон таймфреймів: 1–60 хв, якщо таймфрейм можна прочитати.\n\n"
        "Я можу видати CALL, PUT або НЕ ВХОДИТИ. Висновок не є гарантією та не замінює незалежну перевірку.\n\n"
        "Команди: /help — інструкція, /privacy — приватність."
    )
    await update.effective_message.reply_text(msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Як отримати кращий аналіз:\n"
        "1. Зробіть повний скриншот графіка без обрізаних свічок.\n"
        "2. Залиште видимими назву активу, таймфрейм і панелі індикаторів.\n"
        "3. Не закривайте шкали ціни та часу.\n"
        "4. Не надсилайте скриншоти з особистими даними.\n\n"
        "OpenCV перевіряє лише якість/структуру зображення; AI інтерпретує те, що видно. "
        "RSI/MACD не обчислюються точно з одного скриншота, якщо їх значення не видно."
    )


async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Приватність: скриншот завантажується в пам'ять процесу для аналізу та передається до Google Gemini API, "
        "якщо власник бота налаштував GEMINI_API_KEY. Не надсилайте персональні дані. "
        "Бот не зберігає зображення на диск навмисно, але хостинг/провайдери можуть мати технічні журнали. "
        "Власник бота має ознайомитися з політиками Telegram і Google."
    )


def user_is_limited(user_id: int) -> tuple[bool, str]:
    now = time.time()
    previous = last_request.get(user_id, 0)
    if now - previous < USER_COOLDOWN_SECONDS:
        return True, f"Зачекайте {int(USER_COOLDOWN_SECONDS - (now - previous)) + 1} с перед наступним аналізом."
    q = daily_requests[user_id]
    while q and now - q[0] > 86400:
        q.popleft()
    if len(q) >= MAX_DAILY_USER_REQUESTS and user_id not in ADMIN_USER_IDS:
        return True, f"Досягнуто ліміту {MAX_DAILY_USER_REQUESTS} аналізів за 24 години."
    last_request[user_id] = now
    q.append(now)
    return False, ""


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if not message or not user:
        return
    limited, reason = user_is_limited(user.id)
    if limited:
        await message.reply_text(reason)
        return

    photo = message.photo[-1] if message.photo else None
    document = message.document if message.document and (message.document.mime_type or "").startswith("image/") else None
    if not photo and not document:
        await message.reply_text("Надішліть зображення як фото або файл PNG/JPEG/WebP.")
        return
    file_id = photo.file_id if photo else document.file_id
    file_obj = await context.bot.get_file(file_id)
    image_bytes = bytes(await file_obj.download_as_bytearray())
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        await message.reply_text(f"Файл завеликий. Максимум: {MAX_IMAGE_MB} МБ.")
        return

    mime = (document.mime_type if document else "image/jpeg") or "image/jpeg"
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        mime = "image/jpeg"
    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    cv_report = cv_image_report(image_bytes)
    if not cv_report.get("valid"):
        await message.reply_text("Не вдалося прочитати файл. Спробуйте PNG або JPEG.")
        return
    try:
        result = await analyze_with_gemini(image_bytes, mime, cv_report)
        await message.reply_text(format_analysis(result, cv_report), disable_web_page_preview=True)
    except Exception as exc:
        log.exception("Image analysis failed: %s", exc)
        await message.reply_text(
            "Не вдалося завершити AI-аналіз (можливо, перевищено безкоштовну квоту або модель недоступна). "
            "Зараз безпечний висновок: НЕ ВХОДИТИ. Спробуйте пізніше."
        )


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text("Надішліть скриншот графіка або скористайтеся /help.")


@app.get("/")
async def health():
    return {"status": "ok", "service": "binarium-screenshot-analyzer"}


@app.post("/telegram")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if WEBHOOK_SECRET and not hmac.compare_digest(x_telegram_bot_api_secret_token or "", WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Bot is starting")
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.update_queue.put(update)
    except Exception:
        log.exception("Invalid Telegram update")
        raise HTTPException(status_code=400, detail="Invalid update")
    return JSONResponse({"ok": True})


@app.on_event("startup")
async def startup():
    global telegram_app
    telegram_app = Application.builder().token(BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("privacy", privacy_command))
    telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, process_image))
    telegram_app.add_handler(MessageHandler(filters.ALL, unknown_message))
    await telegram_app.initialize()
    await telegram_app.start()
    if PUBLIC_BASE_URL:
        webhook_url = PUBLIC_BASE_URL + "/telegram"
        await telegram_app.bot.set_webhook(
            url=webhook_url,
            secret_token=WEBHOOK_SECRET or None,
            allowed_updates=["message"],
            drop_pending_updates=False,
        )
        log.info("Telegram webhook configured: %s", webhook_url)
    else:
        log.warning("Webhook not configured; set PUBLIC_BASE_URL to the public HTTPS service URL.")


@app.on_event("shutdown")
async def shutdown():
    global telegram_app
    if telegram_app:
        try:
            await telegram_app.bot.delete_webhook()
            await telegram_app.stop()
            await telegram_app.shutdown()
        except Exception:
            log.exception("Error during shutdown")
        telegram_app = None
