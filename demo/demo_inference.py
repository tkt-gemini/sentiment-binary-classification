"""
Demo Inference Script
Script demo nhanh để test model đã train

Chạy: python demo_inference.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predict import SentimentPredictor


def main():
    """Demo inference với các câu mẫu"""
    
    print("="*70)
    print("🎭 DEMO: PHÂN TÍCH CẢM XÚC TIẾNG VIỆT")
    print("="*70)
    
    # Khởi tạo predictor
    print("\n📦 Đang tải model...")
    try:
        predictor = SentimentPredictor(model_dir='../app/models')
    except:
        # Thử đường dẫn khác
        try:
            predictor = SentimentPredictor(model_dir='app/models')
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            print("\n💡 Hướng dẫn:")
            print("   1. Đảm bảo đã chạy training: cd app && python train.py")
            print("   2. Hoặc chạy notebook main.ipynb để export model")
            return
    
    # Các câu test mẫu
    test_sentences = [
        "Thầy giảng bài rất hay và dễ hiểu",
        "Giảng hơi buồn ngủ, cần cải thiện thêm",
        "Cơ sở vật chất rất tuyệt vời",
        "Thường xuyên đi muộn và không có trách nhiệm",
        "Giáo viên nhiệt tình, luôn giúp đỡ sinh viên",
        "Bài giảng khô khan, không sinh động",
        "Phòng học sạch sẽ, thoáng mát",
        "Thiết bị cũ kỹ, không hoạt động tốt"
    ]
    
    print("\n" + "="*70)
    print("📝 CÁC CÂU TEST MẪU")
    print("="*70)
    
    # Dự đoán
    results = predictor.predict_batch(test_sentences)
    
    # Hiển thị kết quả
    for i, (text, result) in enumerate(zip(test_sentences, results), 1):
        sentiment = result['sentiment']
        emoji = "😊" if sentiment == "positive" else "😔"
        prob_pos = result['prob_positive']
        prob_neg = result['prob_negative']
        
        print(f"\n[{i}] {text}")
        print(f"    ➜ {emoji} {sentiment.upper()}")
        print(f"       Tích cực: {prob_pos:6.1%} | Tiêu cực: {prob_neg:6.1%}")
    
    # Interactive mode
    print("\n" + "="*70)
    print("💬 CHẾ ĐỘ TƯƠNG TÁC")
    print("="*70)
    print("Nhập văn bản để phân tích (Enter để thoát)\n")
    
    while True:
        try:
            user_input = input("📝 Nhập câu: ").strip()
            if not user_input:
                break
            
            result = predictor.predict_single(user_input)
            sentiment = result['sentiment']
            emoji = "😊" if sentiment == "positive" else "😔"
            
            print(f"    ➜ {emoji} {sentiment.upper()}")
            print(f"       Tích cực: {result['prob_positive']:6.1%} | Tiêu cực: {result['prob_negative']:6.1%}\n")
            
        except KeyboardInterrupt:
            break
    
    print("\n" + "="*70)
    print("👋 Cảm ơn đã sử dụng!")
    print("="*70)


if __name__ == "__main__":
    main()

