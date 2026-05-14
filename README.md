import os
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo
import discord
from discord.ext import commands
from google import genai
from google.genai import types
from flask import Flask
from threading import Thread

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")
MAX_HISTORY = 20
RATE_LIMIT = 10
RATE_WINDOW = 60
app = Flask("")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN secret is not set.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY secret is not set.")

client_ai = genai.Client(api_key=GEMINI_API_KEY)
KST = ZoneInfo("Asia/Seoul")
SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

user_request_times: dict[str, deque] = defaultdict(deque)


def get_kst_now() -> str:
    now = datetime.now(KST)
    return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S (KST)")


def is_rate_limited(user_id: str) -> bool:
    now = time.monotonic()
    times = user_request_times[user_id]
    while times and now - times[0] > RATE_WINDOW:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        return True
    times.append(now)
    return False


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_history(history: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_user_history(user_id: str) -> list:
    history = load_history()
    return history.get(user_id, [])


def append_user_history(user_id: str, role: str, text: str) -> None:
    history = load_history()
    messages = history.get(user_id, [])
    messages.append({"role": role, "text": text})
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]
    history[user_id] = messages
    save_history(history)


def clear_user_history(user_id: str) -> None:
    history = load_history()
    history.pop(user_id, None)
    save_history(history)


def build_contents(past_messages: list, new_prompt: str) -> list:
    contents = []
    for msg in past_messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["text"])]))
    stamped_prompt = f"[현재 한국 시각: {get_kst_now()}]\n{new_prompt}"
    contents.append(types.Content(role="user", parts=[types.Part(text=stamped_prompt)]))
    return contents


async def send_long(ctx, text: str) -> None:
    if len(text) <= 2000:
        await ctx.send(text)
    else:
        for i in range(0, len(text), 2000):
            await ctx.send(text[i : i + 2000])


@bot.event
async def on_ready():
    print(f"{bot.user} 로그인 완료")


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    raise error


@bot.command()
async def ai(ctx, *, prompt):
    user_id = str(ctx.author.id)

    if is_rate_limited(user_id):
        await ctx.send(
            "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요. (1분에 최대 10회)"
        )
        return

    try:
        async with ctx.typing():
            past = get_user_history(user_id)
            contents = build_contents(past, prompt)
            response = client_ai.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[SEARCH_TOOL],
                    system_instruction=(
                        "최신 정보나 실시간 데이터가 필요한 질문에는 Google 검색 도구를 사용해서 답하세요. "
                        "답변 시 절대 서론, 안내문, '현재 시각 기준', '검색 결과에 따르면' 같은 도입부를 쓰지 마세요. "
                        "사용자가 물어본 내용의 본론만 바로 출력하세요. "
                        "답변은 핵심만 담아 500자 이내로 간결하게 작성하세요."
                    ),
                ),
            )
            result = response.text if response.text else "응답 없음"

        append_user_history(user_id, "user", prompt)
        append_user_history(user_id, "model", result)

        await send_long(ctx, result)

    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            await ctx.send("현재 AI 사용량이 많아 잠시 후에 다시 시도해주세요.")
        else:
            await ctx.send(f"에러 발생: {e}")
        print(e)


@bot.command()
async def reset(ctx):
    clear_user_history(str(ctx.author.id))
    await ctx.send("대화 기록이 초기화되었습니다.")


@app.route("/")
def home():
    return "I'm alive"


def run():
    app.run(host="0.0.0.0", port=9000)


def keep_alive():
    t = Thread(target=run)
    t.start()


keep_alive()
bot.run(DISCORD_TOKEN)
