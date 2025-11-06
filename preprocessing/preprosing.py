# preprocessing/preprosing.py

import re
import unicodedata
from symspellpy import SymSpell, Verbosity
from transformers import pipeline
from .utils import detect_language_mix, need_llm_normalization

# --- LLM моделін инициализациялау ---
normalizer_llm = pipeline(
    "text2text-generation",
    model="t5-small",  # Используем модель BART
    max_new_tokens=128
)

# --- SymSpell инициализация ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)  # қажет болса

def normalize_text(s: str) -> str:
    """Нормализация: артық символдар мен регистрді түзету"""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("№", " № ")
    s = re.sub(r"[^a-zA-Zа-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ\s#№\-\:\./]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def correct_token(token: str) -> str:
    """Бір токеннің орфографиясын түзету"""
    if len(token) < 3 or token.isdigit():
        return token
    suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else token


def soft_correct(s: str) -> str:
    """Барлық токендердің орфографиясын түзету"""
    return " ".join(correct_token(t) for t in s.split())


def llm_refine(text: str) -> str:
    """LLM арқылы грамматикалық түзету"""
    prompt = (
    f"Мәтінді қазақ тілінде грамматикалық және орфографиялық тұрғыдан дұрыс жаз:\n\n{text}\n\n"
    f"Тек түзетілген нұсқасын қайтар."
)
    return normalizer_llm(prompt)[0]["generated_text"].strip()


def normalize_complaint(text: str) -> str:
    """Негізгі нормализация pipeline"""
    step1 = normalize_text(text)
    step2 = soft_correct(step1)
    lang_mix = detect_language_mix(step2)
    print(f"[Language detected]: {lang_mix}")

    use_llm = need_llm_normalization(step2, lang_mix)
    print(f"[Use LLM Normalization?] {use_llm}")

    if use_llm:
        step4 = llm_refine(step2)
        print(f"[LLM refined]: {step4}")
        return step4

    return step2
