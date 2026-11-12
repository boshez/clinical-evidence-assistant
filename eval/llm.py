"""LLM helpers for evaluation: rate-limit retries and the judge model."""
import re
import time

from groq import RateLimitError

from agent.llm import LLM, chat, get_client

# WHY a different model family from the answerer (openai/gpt-oss-120b): a
# model marking its own answers tends to agree with itself.
JUDGE_MODEL = "qwen/qwen3.8-27b"
MAX_TRIES = 6


def patient(llm: LLM) -> LLM:
    """Wrap an LLM so a rate-limit error waits and retries instead of failing."""
    # WHY: the free tier limits requests per minute, and a 40-question run
    # makes a few hundred calls.
    def call(system: str, user: str) -> str:
        for attempt in range(1, MAX_TRIES + 1):
            try:
                return llm(system, user)
            except RateLimitError:
                if attempt == MAX_TRIES:
                    raise
                time.sleep(15 * attempt)
        return ""

    return call


def judge_chat(system: str, user: str) -> str:
    """Send one prompt to the judge model and return its reply text."""
    response = get_client().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    # WHY: this model can print its reasoning inside <think> tags first.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


agent_llm = patient(chat)
judge_llm = patient(judge_chat)
