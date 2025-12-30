"""
Preprocessing Module
Module tiền xử lý dữ liệu cho dự án phân tích cảm xúc tiếng Việt
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import string
from typing import List, Set, Optional

try:
    from underthesea import word_tokenize
    USE_UNDERTHESEA = True
except ImportError:
    USE_UNDERTHESEA = False
    print("Warning: underthesea không được cài đặt. Sử dụng tokenizer đơn giản.")


def load_stopwords(filepath: str = 'vietnamese-stopwords.txt') -> Set[str]:
    """
    Tải danh sách stopwords từ file
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn tới file stopwords
    
    Returns:
    --------
    Set[str] : Tập hợp các stopwords
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        print(f"✅ Đã tải {len(stopwords)} stopwords từ {filepath}")
        return stopwords
    except FileNotFoundError:
        print(f"⚠️  Không tìm thấy file {filepath}. Sử dụng tập rỗng.")
        return set()


def remove_punctuation(text: str) -> str:
    """Xóa dấu câu"""
    return text.translate(str.maketrans('', '', string.punctuation))


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode tiếng Việt"""
    return unicodedata.normalize('NFC', text)


def remove_duplicate_characters(text: str) -> str:
    """Xóa các ký tự lặp liên tiếp (vd: 'haaay' -> 'hay')"""
    return re.sub(r'(.)\1+', r'\1', text)


def remove_emoji(text: str) -> str:
    """Xóa emoji khỏi văn bản"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_urls(text: str) -> str:
    """Xóa URLs khỏi văn bản"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_stopwords(text: str, stopwords: Set[str]) -> str:
    """Loại bỏ stopwords"""
    tokens = text.split()
    clean_tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(clean_tokens)


def preprocess_text(text: str, stopwords: Optional[Set[str]] = None) -> str:
    """
    Hàm tiền xử lý chính cho văn bản tiếng Việt.
    
    Pipeline:
    1. Chuyển về chữ thường
    2. Chuẩn hóa Unicode
    3. Xóa URLs và emoji
    4. Xóa ký tự lặp
    5. Xóa dấu câu
    6. Tokenize (tách từ tiếng Việt)
    7. Loại bỏ stopwords
    
    Parameters:
    -----------
    text : str
        Văn bản cần xử lý
    stopwords : Set[str], optional
        Tập hợp các stopwords cần loại bỏ
    
    Returns:
    --------
    str : Văn bản đã được xử lý
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # 1. Chuyển về chữ thường và loại bỏ khoảng trắng thừa
    text = text.lower().strip()
    
    # 2. Chuẩn hóa Unicode
    text = normalize_unicode(text)
    
    # 3. Xóa URLs và emoji
    text = remove_urls(text)
    text = remove_emoji(text)
    
    # 4. Xóa ký tự lặp
    text = remove_duplicate_characters(text)
    
    # 5. Xóa dấu câu
    text = remove_punctuation(text)
    
    # 6. Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    # 7. Tokenize (Tách từ tiếng Việt)
    if USE_UNDERTHESEA:
        text = word_tokenize(text, format="text")
    
    # 8. Loại bỏ stopwords
    if stopwords is not None:
        text = remove_stopwords(text, stopwords)
    
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, 
                        text_column: str = 'sentence',
                        stopwords: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Tiền xử lý toàn bộ DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    text_column : str
        Tên cột chứa văn bản cần xử lý
    stopwords : Set[str], optional
        Tập hợp các stopwords
    
    Returns:
    --------
    pd.DataFrame : DataFrame với cột mới chứa văn bản đã xử lý
    """
    df = df.copy()
    df['sentence_processed'] = df[text_column].apply(
        lambda x: preprocess_text(x, stopwords)
    )
    
    # Loại bỏ các dòng có văn bản rỗng sau khi xử lý
    df = df[df['sentence_processed'].str.len() > 0]
    df.reset_index(drop=True, inplace=True)
    
    return df


if __name__ == "__main__":
    # Test
    sample_texts = [
        "Thầy giảng bài rất hay và dễ hiểu!!! 😊",
        "Giảng hơi buồn ngủ, cần cải thiện thêm...",
        "Cơ sở vật chất rất tuyệt vời!!!!"
    ]
    
    stopwords = load_stopwords()
    
    print("\n" + "="*50)
    print("KIỂM TRA TIỀN XỬ LÝ")
    print("="*50)
    
    for text in sample_texts:
        processed = preprocess_text(text, stopwords)
        print(f"\nGốc: {text}")
        print(f"Xử lý: {processed}")

