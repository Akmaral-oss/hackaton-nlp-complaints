# nlp/llm_utils.py

import json
from transformers import pipeline

# Инициализация модели LLM (используется одна модель для всего)
normalizer_llm = pipeline("text2text-generation", model="google/mt5-small", max_new_tokens=256)

def llm_fill_entities(text: str, missing_entities: list):
    """Использует LLM для дополнения недостающих сущностей."""
    prompt = f"Мәтін: {text}\nКерек мәндер: {', '.join(missing_entities)}. JSON форматында қайтар."
    response = normalizer_llm(prompt)[0]["generated_text"]

    try:
        json_data = json.loads(response)
        return json_data
    except Exception:
        return {}

def complete_with_llm(result: dict, missing_entities: list):
    """Дополняет недостающие сущности с использованием LLM."""
    fill_result = llm_fill_entities(result["text"], missing_entities)

    for key in missing_entities:
        if key in fill_result:
            result[key] = fill_result[key]
            result["confidence"][key] = 0.8  # Понижаем уверенность

    return result

def generate_solution_with_llm(complaint_text: str, route: str, time: str, place: str, aspect: str, priority: str):
    """Генерация решения для города на основе анализа жалобы."""
    prompt = f"""
    Төменде берілген шағымды тексеріп, қала әкімшілігіне қандай шешімдер қабылдау қажет екенін қазақ тілінде жазыңыз:
    
    Жалоба: "{complaint_text}"
    Маршрут: {route}
    Уақыт: {time}
    Орын: {place}
    Аспект: {aspect}
    Уровень жалобы: {priority}
    
    Рекомендация:
    """
    result = normalizer_llm(prompt)
    solution = result[0]['generated_text'].strip()  # Убираем лишнюю точку
    return solution
