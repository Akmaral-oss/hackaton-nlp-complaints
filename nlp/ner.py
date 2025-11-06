# nlp/ner.py

import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# --- Инициализация модели NER ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased")
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# --- Функция для извлечения времени из текста ---
def extract_time_from_text(text: str) -> str:
    """Извлекает время из текста, используя регулярные выражения."""
    # Регулярные выражения для извлечения времени
    time_pattern = r"(\d{1,2}[:\s]?\d{2}\s?[APap][Mm]|\d{1,2}[\s]?(AM|PM|am|pm)|\d{1,2}([:]\d{2})?)"
    time_match = re.search(time_pattern, text)

    # Возвращаем время, если найдено, иначе None
    if time_match:
        return time_match.group(0)
    return None

# --- Извлечение сущностей (место, маршруты, время и т. д.) ---
def extract_entities_with_ner(text: str):
    """Извлекает маршруты, места, время и объекты из текста с помощью NER и регулярных выражений."""
    ner_results = nlp_ner(text)
    entities = {"places": [], "objects": [], "routes": [], "times": []}

    # Извлечение сущностей с помощью NER
    for entity in ner_results:
        label = entity.get("entity_group", "")
        value = entity.get("word", "")

        if label == "LOC":  # Местоположение (улица, остановка)
            entities["places"].append((value, entity["score"]))

        elif label == "ORG":  # Организация (часто — транспорт)
            entities["objects"].append((value, entity["score"]))

        elif label == "MISC":  # Разное — иногда маршруты
            if re.match(r"№\d+", value):
                entities["routes"].append((value, entity["score"]))
            else:
                entities["objects"].append((value, entity["score"]))

    # Извлечение времени
    time = extract_time_from_text(text)
    if time:
        entities["times"].append((time, 1.0))  # добавляем уверенность (1.0, так как мы ищем конкретное время)

    return entities



def full_ner_pipeline(text: str, submitted_ts: str):
    """
    Полный пайплайн NER + Aspect классификация.
    Возвращает все сущности и классифицированный аспект.
    """
    # 1️⃣ Нормализация текста
    normalized_text = normalize_complaint(text)
    print(f"[Normalized Complaint]: {normalized_text}")

    # 2️⃣ Извлечение сущностей
    ner_results = extract_entities_with_ner(normalized_text)
    print(f"[NER Results]: {ner_results}")

    # 3️⃣ Классификация аспекта
    aspect, aspect_confidence = classify_aspect(normalized_text)
    print(f"[Aspect Classification]: {aspect} (confidence: {aspect_confidence:.2f})")

    # 4️⃣ Сборка финального результата
    result = {
        "text": normalized_text,
        "route": ner_results["routes"][0][0] if ner_results["routes"] else None,
        "place": ner_results["places"][0][0] if ner_results["places"] else None,
        "object": ner_results["objects"][0][0] if ner_results["objects"] else None,
        "time": submitted_ts,
        "aspect": aspect,
        "confidence": {
            "route": ner_results["routes"][0][1] if ner_results["routes"] else 0.0,
            "place": ner_results["places"][0][1] if ner_results["places"] else 0.0,
            "object": ner_results["objects"][0][1] if ner_results["objects"] else 0.0,
            "aspect": aspect_confidence,
        },
    }

    return result
