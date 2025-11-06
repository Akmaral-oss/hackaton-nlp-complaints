# visualization/visualizer.py

import matplotlib.pyplot as plt  # Импорт для графиков
import pandas as pd  # Импорт для работы с данными

def visualize_complaints(data):
    """
    Visualizes the most problematic routes, complaint distribution by levels, and aspect frequencies.
    """
    df = pd.DataFrame(data)

    # Visualize most problematic routes (by number of complaints)
    route_complaints = df['route'].value_counts()
    plt.figure(figsize=(10,6))
    route_complaints.plot(kind='bar', title="Самые проблемные маршруты")
    plt.xlabel("Маршрут")
    plt.ylabel("Количество жалоб")
    plt.show()

    # Visualize complaint distribution by levels
    level_complaints = df['priority'].value_counts()
    plt.figure(figsize=(10,6))
    level_complaints.plot(kind='bar', title="Распределение жалоб по уровням")
    plt.xlabel("Уровень жалобы")
    plt.ylabel("Количество жалоб")
    plt.show()

    # Visualize aspect frequencies
    aspect_complaints = df['aspect'].value_counts()  # Now we visualize dynamic aspects
    plt.figure(figsize=(10,6))
    aspect_complaints.plot(kind='bar', title="Частота аспектов жалоб")
    plt.xlabel("Аспект")
    plt.ylabel("Количество жалоб")
    plt.show()
