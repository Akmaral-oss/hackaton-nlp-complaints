# nlp/priority_classifier.py

from transformers import pipeline

# --- Zero-shot Classifier for Complaint Levels ---
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_complaint_level(text: str) -> str:
    """
    Classifies the complaint level based on the text.
    Uses both rules and zero-shot classification.
    """
    low_level_keywords = ["замечание", "неудобство", "редкое событие", "незначительная проблема"]
    medium_level_keywords = ["повторяющаяся проблема", "неудобства часто", "сбой в сервисе"]
    high_level_keywords = ["системное нарушение", "влияние на многих пассажиров", "массовые задержки"]
    critical_level_keywords = ["опасность", "травма", "угроза безопасности"]

    if any(keyword in text.lower() for keyword in critical_level_keywords):
        return "критический"
    elif any(keyword in text.lower() for keyword in high_level_keywords):
        return "высокий"
    elif any(keyword in text.lower() for keyword in medium_level_keywords):
        return "средний"
    elif any(keyword in text.lower() for keyword in low_level_keywords):
        return "низкий"

    candidate_labels = ["низкий", "средний", "высокий", "критический"]
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # Return the most likely level
