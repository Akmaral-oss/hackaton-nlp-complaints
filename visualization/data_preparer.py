# data_preparer.py

def collect_complaints(complaint_data, complaint_output):
    """
    Собирает все данные жалоб для последующей визуализации.
    """
    complaint_data.append(complaint_output)
    
    # Визуализация при накоплении 10 жалоб
    if len(complaint_data) >= 10:
        visualize_complaints(complaint_data)

    return complaint_data  # Возвращаем обновленный список

def prepare_complaint_data(complaint_text, ner_results, aspect, priority, solution):
    """
    Подготавливает структурированные данные жалобы для визуализации.
    """
    complaint_output = {
        "complaint_id": str(hash(complaint_text)),  # Генерация уникального ID жалобы
        "route": ner_results["routes"][0][0] if ner_results["routes"] else "Не указан",  # Если маршрут не найден, ставим "Не указан"
        "time": ner_results["times"][0][0] if ner_results["times"] else "Не указано",  # Время
        "place": ner_results["places"][0][0] if ner_results["places"] else "Не указано",  # Место
        "aspect": aspect,  # Аспект
        "priority": priority,  # Уровень жалобы
        "recommendation": solution  # Рекомендация для сити-менеджера
    }

    return complaint_output
