# 🎭 Phân Tích Cảm Xúc Tiếng Việt - Vietnamese Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án phân tích cảm xúc (Sentiment Analysis) cho văn bản tiếng Việt, áp dụng các thuật toán Machine Learning để phân loại đánh giá về giảng viên thành hai nhóm: **Tích cực (Positive)** và **Tiêu cực (Negative)**.

---

## 📋 Mục Lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Mô hình sử dụng](#-mô-hình-sử-dụng)
- [Kết quả](#-kết-quả)
- [Hướng dẫn chạy](#-hướng-dẫn-chạy)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới Thiệu

### Bài toán

Phân tích cảm xúc (Sentiment Analysis) là bài toán phân loại văn bản theo cảm xúc mà người viết thể hiện. Trong dự án này, chúng tôi tập trung vào việc phân tích các đánh giá của sinh viên về giảng viên.

### Mục tiêu

- Xây dựng hệ thống tự động phân loại cảm xúc cho văn bản tiếng Việt
- So sánh hiệu quả của các thuật toán Machine Learning: Logistic Regression, Linear SVM, Naive Bayes
- Đạt được độ chính xác cao (> 90%) trên tập test
- Xây dựng ứng dụng demo thân thiện với người dùng

### Ứng dụng thực tế

- Phân tích ý kiến sinh viên về giảng viên, khóa học
- Giám sát phản hồi trên mạng xã hội
- Đánh giá chất lượng dịch vụ từ reviews khách hàng
- Hỗ trợ ra quyết định dựa trên phân tích dư luận

---

## 📊 Dataset

### UIT-VSFC (Vietnamese Students' Feedback Corpus)

**Nguồn:** [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)

### Mô tả

Dataset bao gồm các đánh giá của sinh viên về giảng viên, được gán nhãn với 3 loại cảm xúc:
- `positive`: Đánh giá tích cực
- `negative`: Đánh giá tiêu cực  

### Thống kê

| Split | Positive | Negative | Tổng |
|-------|----------|----------|------|
| Train | 5,071    | 2,909    | 7,980 |
| Dev   | 714      | 405      | 1,119 |
| Test  | 1,425    | 791      | 2,216 |
| **TỔNG** | **7,210** | **4,105** | **11,315** |

**Phân bố:** ~64% Positive, ~36% Negative (imbalanced)

### Cấu trúc dữ liệu

```json
{
  "sentence": "Thầy giảng bài rất hay và dễ hiểu",
  "sentiment": "positive"
}
```

### Download Dataset

```bash
git clone https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
```

---

## 🔄 Pipeline

### 1. **Thu thập & Chuẩn bị dữ liệu**
   - Load dữ liệu từ file JSON
   - Encode nhãn: `negative=0`, `positive=1`

### 2. **Tiền xử lý (Preprocessing)**

Pipeline tiền xử lý bao gồm các bước:

```python
Text → Lowercase → Unicode Normalization → Remove URLs/Emoji → Remove Duplicate Chars → Remove Punctuation → Word Tokenization (underthesea) → Remove Stopwords → Clean Text
```

**Chi tiết:**
- **Chuẩn hóa Unicode:** Đồng nhất các ký tự tiếng Việt (NFC)
- **Xóa noise:** URLs, emoji, dấu câu, ký tự lặp ("haaay" → "hay")
- **Tách từ:** Sử dụng `underthesea` để word tokenization tiếng Việt
- **Loại stopwords:** Xóa các từ không mang nghĩa (từ danh sách 2,063 stopwords)

### 3. **Trích xuất đặc trưng (Feature Extraction)**

**TF-IDF Vectorizer:**
- `max_features=5000`: Giữ lại 5000 từ quan trọng nhất
- `ngram_range=(1, 2)`: Unigram + Bigram
- `min_df=2`: Bỏ các từ xuất hiện < 2 lần
- `sublinear_tf=True`: Áp dụng logarithmic scaling

### 4. **Huấn luyện mô hình (Training)**

So sánh 3 thuật toán:
- **Logistic Regression** (LR)
- **Linear Support Vector Machine** (SVM)
- **Multinomial Naive Bayes** (NB)

Với `class_weight='balanced'` để xử lý imbalanced data.

### 5. **Tối ưu hóa (Hyperparameter Tuning)**

- Sử dụng **GridSearchCV** với 5-fold cross-validation
- Tìm optimal threshold trên dev set để maximize F1-score
- Chọn mô hình tốt nhất dựa trên F1-score

### 6. **Đánh giá & Inference**

- Đánh giá trên tập test với các metrics: Accuracy, F1-Score, Precision, Recall
- Export model để sử dụng cho inference
- Demo qua Streamlit web app

---

## 🤖 Mô Hình Sử Dụng

### Tổng quan các mô hình

| Mô hình | Ưu điểm | Nhược điểm |
|---------|---------|------------|
| **Logistic Regression** | Đơn giản, nhanh, hiệu quả với text | Giả định tuyến tính |
| **Linear SVM** | Hiệu quả với high-dim data, robust | Tốn thời gian train với dataset lớn |
| **Naive Bayes** | Rất nhanh, ít data cũng hoạt động tốt | Giả định independence |

### Mô hình được chọn: **Linear SVM** ✅

**Lý do:**
1. **Hiệu suất cao nhất:** F1-Score = 0.9266 trên test set
2. **Robust:** Hoạt động tốt với imbalanced data
3. **Generalization tốt:** Không bị overfit, gap train-test nhỏ
4. **Hiệu quả với TF-IDF:** SVM phù hợp với feature space sparse và high-dimensional

**Hyperparameters tối ưu:**
```python
LinearSVC(
    C=1.0,                    # Regularization strength
    class_weight='balanced',  # Xử lý imbalanced data
    max_iter=2000,
    random_state=42
)
```

**Optimal Threshold:** 0.46

---

## 📈 Kết Quả

### Hiệu suất mô hình trên Test Set

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9220 | 0.9254 | 0.9220 | 0.9221 |
| **Linear SVM** | **0.9260** | **0.9287** | **0.9260** | **0.9266** ✅ |
| Naive Bayes | 0.9334 | 0.9347 | 0.9334 | 0.9335 |

### Confusion Matrix (Test Set - Linear SVM)

|               | Predicted Negative | Predicted Positive |
|---------------|-------------------:|-------------------:|
| **Actual Negative** | 715 | 76 |
| **Actual Positive** | 88 | 1,337 |

### Phân tích chi tiết (Linear SVM)

**Class-wise Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.89 | 0.90 | 0.90 | 791 |
| Positive | 0.95 | 0.94 | 0.94 | 1,425 |
| **Macro Avg** | **0.92** | **0.92** | **0.92** | **2,216** |

### Nhận xét

✅ **Ưu điểm:**
- Độ chính xác cao (>90%) trên tất cả các metrics
- Cân bằng tốt giữa Precision và Recall
- Generalization tốt (train-dev-test performance ổn định)
- Hiệu quả với cả 2 classes (Positive & Negative)

⚠️ **Hạn chế:**
- Vẫn còn confuse ~4-5% trường hợp (do ngôn ngữ mỉa mai, phức tạp)
- Performance trên Negative class thấp hơn Positive (do imbalanced data)

---

## 🚀 Hướng Dẫn Chạy

### 1. Cài đặt môi trường

#### a. Clone repository

```bash
git clone <repository-url>
cd big-ex
```

#### b. Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### c. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Chuẩn bị dữ liệu

dataset UIT-VSFC đã cài sẵn trong thư mục `../data/`:

```
../data/
├── UIT-VSFC-train.json
├── UIT-VSFC-dev.json
├── UIT-VSFC-test.json
└── vietnamese-stopwords.txt
```

---

### 3. Chạy Training

#### Sử dụng Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

Chạy tất cả các cells để:
- Khám phá dữ liệu (EDA)
- Thử nghiệm nhiều mô hình
- So sánh hiệu suất
- Export model tốt nhất

**Output:**
```
app/models/
├── sentiment_pipeline.pkl  # Model pipeline (TF-IDF + Classifier)
├── label_encoder.pkl       # Label encoder
└── model_metadata.pkl      # Model info & metrics
```

---

### 4. Chạy Demo/Inference

#### Demo Script (Command Line)

```bash
cd demo
python demo_inference.py
```

Features:
- Test với các câu mẫu có sẵn
- Interactive mode: nhập câu để phân tích real-time

#### Streamlit Web App 🌟

```bash
cd app
streamlit run streamlit_app.py
```

Giao diện web với:
- Nhập văn bản và nhận kết quả real-time
- Hiển thị xác suất (probability bars)
- Xem văn bản sau preprocessing
- Thông tin model metadata

#### Python API

```python
from app.predict import SentimentPredictor

# Khởi tạo predictor
predictor = SentimentPredictor(model_dir='app/models')

# Dự đoán một câu
result = predictor.predict_single("Thầy giảng bài rất hay")
print(result['sentiment'])  # 'positive'
print(result['prob_positive'])  # 0.95

# Dự đoán nhiều câu
texts = ["Câu 1", "Câu 2", "Câu 3"]
results = predictor.predict_batch(texts)
```

---

## 📁 Cấu Trúc Thư Mục

```
big-ex/
├── app/                          # Source code chính
│   ├── models/                   # Models đã train (generated)
│   │   ├── sentiment_pipeline.pkl
│   │   ├── label_encoder.pkl
│   │   └── model_metadata.pkl
│   ├── main.ipynb                # Script training
│   ├── utils.py                  # Utility functions
│   └── app.py                    # Streamlit web app
│
├── demo/                         # Demo scripts
│   └── demo_inference.py         # Python script demo
│
├── data/                         # Data
│   ├── UIT-VSFC-train.json
│   ├── UIT-VSFC-dev.json
│   ├── UIT-VSFC-test.json
│   └── vietnamese-stopwords.txt
│
├── reports/                      # Báo cáo
│   └── Report.docx
│
├── slides/                       # Slide thuyết trình
│   └── Report.pptx
│
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation
```

---

## 👥 Tác Giả

### Thông tin nhóm

| Họ và tên | Mã SV |
|-----------|-------|
| Hoàng Hải Đăng | 12423009 |
| Trần Khánh Toàn | 12423035 |

**Lớp:** 124231
**Giảng viên hướng dẫn:** Assoc. Prof. Dr. Van-Hau Nguyen

---

## 📚 Tài Liệu Tham Khảo

1. **Dataset:** [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) - Vietnamese Students' Feedback Corpus
2. **Vietnamese NLP:** [Underthesea](https://github.com/undertheseanlp/underthesea) - Vietnamese Natural Language Processing
3. **Scikit-learn:** [Example](https://scikit-learn.org/stable/auto_examples/text/index.html)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Kết Luận

Dự án đã thành công trong việc:
- ✅ Xây dựng pipeline hoàn chỉnh cho bài toán Sentiment Analysis tiếng Việt
- ✅ So sánh và chọn được mô hình tối ưu (Linear SVM, F1=92.66%)
- ✅ Xây dựng ứng dụng demo thân thiện với người dùng
- ✅ Code sạch, có cấu trúc, dễ tái sử dụng và mở rộng

**Hướng phát triển:**
- Thử nghiệm với Deep Learning (LSTM, BERT-Vietnamese)
- Mở rộng cho multi-class classification (more sentiments)
- Deploy model lên cloud (Heroku, AWS, GCP)
- Tích hợp API RESTful

---
**⭐ Nếu thấy dự án hữu ích, hãy cho chúng tôi một star trên GitHub!**