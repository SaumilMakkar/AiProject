import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Categories
CATEGORIES = [
    'business',
    'entertainment',
    'food',
    'graphics',
    'historical',
    'medical',
    'politics',
    'space',
    'sport',
    'technology'  # Fixed typo from 'technologie'
]

# Text preprocessing settings
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'this', 'that', 'these', 'those', 'they', 'their', 'them',
    'there', 'then', 'than', 'what', 'when', 'where', 'which', 'who', 'whom',
    'why', 'how', 'if', 'else', 'because', 'while', 'until', 'since', 'about',
    'above', 'below', 'between', 'into', 'through', 'during', 'before', 'after',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

# Feature extraction settings
MIN_WORD_FREQUENCY = 2
MAX_FEATURES = 2000  # Increased for more categories

# Model settings
SMOOTHING_ALPHA = 1.0  # Laplace smoothing parameter

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True) 