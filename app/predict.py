"""
Prediction/Inference Module
Module để dự đoán cảm xúc cho văn bản mới
"""

import joblib
import numpy as np
from typing import List, Tuple, Dict
from utils import preprocess_text


class SentimentPredictor:
    """
    Class để thực hiện dự đoán cảm xúc
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Khởi tạo predictor
        
        Parameters:
        -----------
        model_dir : str
            Đường dẫn tới thư mục chứa models
        """
        self.model_dir = model_dir
        self.pipeline = None
        self.label_encoder = None
        self.stopwords = None
        self.metadata = None
        
        self._load_models()
    
    def _load_models(self):
        """Load tất cả models và components cần thiết"""
        try:
            self.pipeline = joblib.load(f'{self.model_dir}/sentiment_pipeline.pkl')
            self.label_encoder = joblib.load(f'{self.model_dir}/label_encoder.pkl')
            self.metadata = joblib.load(f'{self.model_dir}/model_metadata.pkl')
            
            # Load stopwords nếu có
            try:
                self.stopwords = joblib.load(f'{self.model_dir}/stopwords.pkl')
            except:
                self.stopwords = set()
            
            print(f"✅ Đã tải model: {self.metadata.get('model_name', 'N/A')}")
            print(f"   Threshold: {self.metadata.get('optimal_threshold', 0.5):.4f}")
            print(f"   F1-Score (Test): {self.metadata.get('f1_score', 0):.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Không thể tải model: {e}")
    
    def predict_single(self, text: str) -> Dict:
        """
        Dự đoán cảm xúc cho một câu
        
        Parameters:
        -----------
        text : str
            Văn bản cần phân tích
        
        Returns:
        --------
        Dict : Dictionary chứa kết quả dự đoán
            - text: văn bản gốc
            - processed: văn bản sau xử lý
            - sentiment: nhãn cảm xúc (positive/negative)
            - sentiment_encoded: nhãn encode (0/1)
            - probability: xác suất của lớp dự đoán
            - prob_negative: xác suất lớp negative
            - prob_positive: xác suất lớp positive
        """
        # 1. Tiền xử lý
        processed_text = preprocess_text(text, self.stopwords)
        
        if not processed_text:
            return {
                'text': text,
                'processed': '',
                'sentiment': 'unknown',
                'sentiment_encoded': -1,
                'probability': 0.0,
                'prob_negative': 0.0,
                'prob_positive': 0.0
            }
        
        # 2. Lấy threshold
        threshold = self.metadata.get('optimal_threshold', 0.5)
        
        # 3. Dự đoán xác suất
        if hasattr(self.pipeline, 'decision_function'):
            # Cho SVM
            decision = self.pipeline.decision_function([processed_text])[0]
            prob_positive = 1 / (1 + np.exp(-decision))
            prob_negative = 1 - prob_positive
        else:
            # Cho LR và NB
            probs = self.pipeline.predict_proba([processed_text])[0]
            prob_negative = probs[0]
            prob_positive = probs[1]
        
        # 4. Áp dụng threshold
        if prob_positive >= threshold:
            sentiment_idx = 1
        else:
            sentiment_idx = 0
        
        sentiment_label = self.label_encoder.inverse_transform([sentiment_idx])[0]
        probability = prob_positive if sentiment_idx == 1 else prob_negative
        
        return {
            'text': text,
            'processed': processed_text,
            'sentiment': sentiment_label,
            'sentiment_encoded': sentiment_idx,
            'probability': probability,
            'prob_negative': prob_negative,
            'prob_positive': prob_positive
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Dự đoán cảm xúc cho nhiều câu
        
        Parameters:
        -----------
        texts : List[str]
            Danh sách các văn bản cần phân tích
        
        Returns:
        --------
        List[Dict] : Danh sách kết quả dự đoán
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """
        Lấy thông tin về model
        
        Returns:
        --------
        Dict : Thông tin model
        """
        return self.metadata


def predict_from_cli():
    """
    Hàm để chạy prediction từ command line
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Dự đoán cảm xúc cho văn bản tiếng Việt')
    parser.add_argument('--text', type=str, help='Văn bản cần phân tích')
    parser.add_argument('--file', type=str, help='File chứa danh sách văn bản (mỗi dòng một câu)')
    parser.add_argument('--model_dir', type=str, default='models', help='Thư mục chứa models')
    
    args = parser.parse_args()
    
    # Khởi tạo predictor
    predictor = SentimentPredictor(model_dir=args.model_dir)
    
    print("\n" + "="*70)
    print("🎭 PHÂN TÍCH CẢM XÚC TIẾNG VIỆT")
    print("="*70)
    
    # Xử lý input
    if args.text:
        # Phân tích một câu
        texts = [args.text]
    elif args.file:
        # Phân tích từ file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {args.file}")
            return
    else:
        # Interactive mode
        print("\n💬 Nhập văn bản cần phân tích (để trống để thoát):\n")
        texts = []
        while True:
            text = input("📝 Văn bản: ").strip()
            if not text:
                break
            texts.append(text)
    
    # Dự đoán
    if texts:
        results = predictor.predict_batch(texts)
        
        print("\n" + "="*70)
        print("📊 KẾT QUẢ PHÂN TÍCH")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            sentiment = result['sentiment']
            emoji = "😊" if sentiment == "positive" else "😔"
            prob = result['probability']
            
            print(f"\n[{i}] {result['text']}")
            print(f"    ➜ {emoji} {sentiment.upper()} ({prob:.1%})")
            print(f"       Tích cực: {result['prob_positive']:.1%} | Tiêu cực: {result['prob_negative']:.1%}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    predict_from_cli()

