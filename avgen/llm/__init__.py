import json
import re

from .base import LLM
from .openai_llm import OpenAILLM

PROVIDERS = {"openai": OpenAILLM}


def register_llm(name, cls):
    PROVIDERS[name] = cls


def create_llm(cfg):
    """cfg: {"provider": "openai", ...provider kwargs}"""
    cfg = dict(cfg)
    return PROVIDERS[cfg.pop("provider", "openai")](**cfg)


def extract_json(text):
    """Parse JSON from an LLM reply, tolerating markdown code fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def wrap_prompt(prompt, prepend="", append=""):
    return "\n\n".join(p for p in (prepend, prompt, append) if p)
