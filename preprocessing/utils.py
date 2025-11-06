 # preprocessing/utils.py

import re
from langdetect import detect_langs

def detect_language_mix(text: str) -> str:
    """Detects language mix in the text (e.g., ru:0.8, kk:0.2)"""
    try:
        langs = detect_langs(text)
        top = sorted([(l.lang, l.prob) for l in langs], key=lambda x: -x[1])[:2]
        return ", ".join([f"{l}:{p:.2f}" for l, p in top])
    except Exception:
        return "unknown"

def need_llm_normalization(text: str, lang_mix: str) -> bool:
    """Automatically determine if LLM normalization is needed"""
    if "ru:" in lang_mix and "kk:" in lang_mix:
        return True
    if len(re.findall(r"[^a-zA-Zа-яА-Яәіңғүұқөһ\s\d]", text)) > 5:
        return True
    if len(text.split()) > 25:
        return True
    return False
