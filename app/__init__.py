"""
Vietnamese Sentiment Analysis Application
Package chứa các modules chính cho dự án phân tích cảm xúc tiếng Việt
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import các modules chính
from .preprocess import preprocess_text, load_stopwords
from .predict import SentimentPredictor

__all__ = [
    'preprocess_text',
    'load_stopwords', 
    'SentimentPredictor'
]

