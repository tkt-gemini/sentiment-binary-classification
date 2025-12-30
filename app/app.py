import streamlit as st
import joblib
import numpy as np
from utils import preprocess_text

# CẤU HÌNH TRANG
st.set_page_config(
    page_title="Phân tích Cảm xúc Tiếng Việt",
    page_icon="🎭",
    layout="centered"
)

# TẢI MODEL VÀ CÁC THÀNH PHẦN
@st.cache_resource
def load_model():
    """Tải model và các thành phần cần thiết"""
    pipeline = joblib.load('models/sentiment_pipeline.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    
    # Tải stopwords nếu có
    try:
        stopwords = joblib.load('models/stopwords.pkl')
    except:
        stopwords = set()
    
    return pipeline, label_encoder, metadata, stopwords

# Load tất cả
try:
    pipeline, label_encoder, metadata, stopwords = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Không thể tải model: {e}")
    st.info("Vui lòng chạy notebook để xuất model trước khi sử dụng demo này.")

# GIAO DIỆN CHÍNH
st.title("🎭 Phân tích Cảm xúc Tiếng Việt")
st.markdown("*Ứng dụng nhận dạng cảm xúc (Tích cực / Tiêu cực) cho văn bản tiếng Việt*")

st.divider()

# Hiển thị thông tin model
if MODEL_LOADED:
    with st.expander("ℹ️ Thông tin Model", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", metadata.get('model_name', 'N/A'))
        col2.metric("F1-Score", f"{metadata.get('f1_score', 0):.2%}")
        col3.metric("Threshold", f"{metadata.get('optimal_threshold', 0.5):.4f}")

# NHẬP VĂN BẢN
st.subheader("📝 Nhập văn bản cần phân tích")

user_input = st.text_input(
    "Nhập câu cần phân tích:",
    placeholder="Ví dụ: Thầy giảng bài rất hay và dễ hiểu"
)
texts_to_analyze = [user_input] if user_input else []

# NÚT PHÂN TÍCH
analyze_button = st.button("🔍 Phân tích cảm xúc", type="primary", use_container_width=True)

# XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ
if analyze_button and MODEL_LOADED and len(texts_to_analyze) > 0:
    st.divider()
    st.subheader("📊 Kết quả phân tích")
    
    # Lấy threshold từ metadata
    threshold = metadata.get('optimal_threshold', 0.5)
    
    for i, text in enumerate(texts_to_analyze):
        # 1. Tiền xử lý
        processed = preprocess_text(text, stopwords)
        
        # 2. Dự đoán xác suất
        if hasattr(pipeline, 'predict_proba'):
            prob = pipeline.predict_proba([processed])[0]
            prob_positive = prob[1]
            prob_negative = prob[0]
        else:
            # Cho SVM
            d = pipeline.decision_function([processed])[0]
            prob_positive = 1 / (1 + np.exp(-d))
            prob_negative = 1 - prob_positive
        
        # 3. Áp dụng threshold
        if prob_positive >= threshold:
            sentiment_idx = 1
            sentiment = "Positive"
            emoji = "😊"
            color = "green"
        else:
            sentiment_idx = 0
            sentiment = "Negative"
            emoji = "😔"
            color = "red"
        
        # 4. Hiển thị kết quả
        with st.container():
            st.markdown(f"**Câu {i+1}:** {text}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(
                    f"<h2 style='color:{color}; text-align:center;'>{emoji} {sentiment}</h2>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.progress(prob_positive, text=f"Tích cực: {prob_positive:.1%}")
                st.progress(prob_negative, text=f"Tiêu cực: {prob_negative:.1%}")
            
            # Hiển thị câu đã xử lý (Debug)
            with st.expander("🔧 Xem chi tiết xử lý"):
                st.code(f"Văn bản gốc: {text}\nSau xử lý : {processed}")
            
            st.divider()

elif analyze_button and len(texts_to_analyze) == 0:
    st.warning("Vui lòng nhập ít nhất một câu để phân tích!")


