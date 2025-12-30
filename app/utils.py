import re
import unicodedata
import string

# Hàm tiền xử lý text (phiên bản không cần underthesea nếu chạy trên môi trường đơn giản)
try:
    from underthesea import word_tokenize
    USE_UNDERTHESEA = True
except ImportError:
    USE_UNDERTHESEA = False
    print("Warning: underthesea không được cài đặt. Sử dụng tokenizer đơn giản.")

def remove_punctuation(text: str) -> str:
    """Xóa dấu câu"""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode tiếng Việt"""
    return unicodedata.normalize('NFC', text)

def remove_duplicate_characters(text: str) -> str:
    """Xóa các ký tự lặp liên tiếp (vd: 'haaay' -> 'hay')"""
    return re.sub(r'(.)\1+', r'\1', text)

def normalize_stopwords(text: str, stopwords: set) -> str:
    """Loại bỏ stopwords"""
    tokens = text.split()
    clean_tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(clean_tokens)

def preprocess_text(text: str, stopwords: set = None) -> str:
    """
    Hàm tiền xử lý chính cho văn bản tiếng Việt.
    
    Parameters:
    -----------
    text : str
        Văn bản cần xử lý
    stopwords : set
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
    
    # 3. Xóa ký tự lặp
    text = remove_duplicate_characters(text)
    
    # 4. Xóa dấu câu
    text = remove_punctuation(text)
    
    # 5. Tokenize (Tách từ tiếng Việt)
    if USE_UNDERTHESEA:
        text = word_tokenize(text, format="text")
    
    # 6. Loại bỏ stopwords
    if stopwords is not None:
        text = normalize_stopwords(text, stopwords)
    
    return text
