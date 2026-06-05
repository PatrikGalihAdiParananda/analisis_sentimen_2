# sentimen/ml_utils/model_loader.py

import os
import pickle
from django.conf import settings # Untuk mengakses BASE_DIR

# Path ke folder ml_utils relatif dari BASE_DIR proyek Anda
MODEL_DIR = os.path.join(settings.BASE_DIR, 'sentimen', 'ml_utils')

MODEL_PATH = os.path.join(MODEL_DIR, 'model_random_forest_final.pkl')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer_final.pkl')

_model = None
_tfidf_vectorizer = None

def load_model_and_vectorizer():
    global _model, _tfidf_vectorizer
    if _model is None or _tfidf_vectorizer is None:
        try:
            with open(MODEL_PATH, 'rb') as model_file:
                _model = pickle.load(model_file)
            with open(TFIDF_PATH, 'rb') as vectorizer_file:
                _tfidf_vectorizer = pickle.load(vectorizer_file)
            print("ML Model and TF-IDF Vectorizer loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model or TF-IDF file not found at {MODEL_PATH} or {TFIDF_PATH}")
            _model = None
            _tfidf_vectorizer = None
        except Exception as e:
            print(f"Error loading model or vectorizer: {e}")
            _model = None
            _tfidf_vectorizer = None
    return _model, _tfidf_vectorizer

# Panggil fungsi ini saat Django app dimulai (opsional, bisa juga di views)
# Namun, memanggilnya sekali di sini akan membuat model siap pakai
# dari sentimen/apps.py