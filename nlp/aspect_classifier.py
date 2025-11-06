# nlp/aspect_classifier.py

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Инициализация модели для генерации эмбеддингов
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts: list):
    """Генерация эмбеддингов для списка жалоб."""
    return model.encode(texts, convert_to_tensor=True)

# --- Unsupervised clustering (KMeans) for aspects ---
def cluster_complaints(texts: list, n_clusters=1):
    """
    Uses KMeans clustering to classify complaints into aspects (topics).
    
    Args:
        texts (list): List of complaint texts.
        n_clusters (int): Number of clusters to identify (can be tuned).
    
    Returns:
        cluster_labels (list): Cluster labels for each complaint (the predicted aspect).
    """
    embeddings = generate_embeddings(texts)

    # ✅ Переводим с MPS (GPU) на CPU и в numpy
    if hasattr(embeddings, "cpu"):
        embeddings = embeddings.cpu().numpy()

    # Clustering with KMeans
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    except ValueError:
        print("⚠️ Not enough samples for clustering, skipping...")

    
    cluster_labels = kmeans.labels_
    
    return cluster_labels


# --- Function to classify aspects based on cluster labels ---
def classify_aspect_unsupervised(texts: list):
    """
    Classifies complaint aspects using unsupervised clustering (KMeans).
    
    Args:
    texts (list): List of complaint texts.
    
    Returns:
    aspect_labels (list): Predicted aspect labels for each complaint.
    """
    cluster_labels = cluster_complaints(texts)

    # Define aspect labels for the clusters (can be adjusted based on domain knowledge)
    aspect_labels = [
        "пунктуальность", "заполненность", "безопасность", "поведение персонала", "состояние автобуса", "оплата"
    ]
    
    # Assign aspect labels to clusters
    clustered_aspects = [aspect_labels[label] for label in cluster_labels]
    
    return clustered_aspects
