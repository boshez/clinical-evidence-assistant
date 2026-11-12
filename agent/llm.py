"""One small function that sends a prompt to Groq and returns the text."""
import json
import os
from functools import lru_cache
from typing import Callable

from dotenv import load_dotenv
from groq import Groq

# Chosen from Groq's live model list on 2026-11-12: the largest chat model on
# the free tier. Groq retires models, so GROQ_MODEL in .env can override it.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# WHY a type alias: nodes accept any function of this shape, so the tests can
# pass a scripted stand-in and check the wiring without calling Groq.
LLM = Callable[[str, str], str]


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Build the Groq client once, using the key in .env."""
    load_dotenv()
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def chat(system: str, user: str) -> str:
    """Send one system + user message pair and return the reply text."""
    response = get_client().chat.completions.create(
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # WHY 0: the same question should give the same route and verdict.
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a reply; {} if there is none."""
    # WHY tolerant: models sometimes wrap JSON in prose or code fences.
    # Returning {} lets each caller choose its own safe default.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
