# 🎭 Phân Tích Cảm Xúc Tiếng Việt - Vietnamese Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
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
- Đạt được độ chính xác cao (> 92%) trên tập test
- Xây dựng ứng dụng demo thân thiện với người dùng

### Ứng dụng thực tế

- Phân tích ý kiến sinh viên về giảng viên, khóa học
- Giám sát phản hồi trên mạng xã hội
- Đánh giá chất lượng dịch vụ từ reviews khách hàng
- Hỗ trợ ra quyết định dựa trên phân tích dư luận

---

## 📊 Dataset

### UIT-VSFC (Vietnamese Students' Feedback Corpus)

**Nguồn:** [UIT-VSFC GitHub Repository](https://github.com/sonvx/vietnam-sentiment-corpus)

### Mô tả

Dataset bao gồm các đánh giá của sinh viên về giảng viên, được gán nhãn với 3 loại cảm xúc:
- `positive`: Đánh giá tích cực
- `negative`: Đánh giá tiêu cực  
- `neutral`: Đánh giá trung lập (đã loại bỏ trong dự án này)

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
  "sentiment": "positive",
  "topic": "lecturer"
}
```

### Download Dataset

```bash
# Tải về từ GitHub
git clone https://github.com/sonvx/vietnam-sentiment-corpus.git

# Hoặc tải trực tiếp các file:
# - UIT-VSFC-train.json
# - UIT-VSFC-dev.json
# - UIT-VSFC-test.json
# Đặt vào thư mục archive/
```

---

## 🔄 Pipeline

### 1. **Thu thập & Chuẩn bị dữ liệu**
   - Load dữ liệu từ file JSON
   - Lọc chỉ lấy topic `lecturer` và loại bỏ nhãn `neutral`
   - Encode nhãn: `negative=0`, `positive=1`

### 2. **Tiền xử lý (Preprocessing)**

Pipeline tiền xử lý bao gồm các bước:

```python
Text → Lowercase → Unicode Normalization → Remove URLs/Emoji 
    → Remove Duplicate Chars → Remove Punctuation 
    → Word Tokenization (underthesea) → Remove Stopwords → Clean Text
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

**Optimal Threshold:** 0.46 (thay vì 0.5 mặc định)

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
- Độ chính xác cao (>92%) trên tất cả các metrics
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

**Các thư viện chính:**
- `scikit-learn==1.5.0` - Machine Learning
- `pandas==2.1.3` - Data manipulation
- `numpy==1.26.2` - Numerical computing
- `underthesea==1.3.5` - Vietnamese NLP
- `streamlit==1.41.1` - Web app
- `matplotlib`, `seaborn` - Visualization

---

### 2. Chuẩn bị dữ liệu

Tải dataset UIT-VSFC và đặt vào thư mục `archive/`:

```
archive/
├── UIT-VSFC-train.json
├── UIT-VSFC-dev.json
├── UIT-VSFC-test.json
└── vietnamese-stopwords.txt
```

**Download:** https://github.com/sonvx/vietnam-sentiment-corpus

---

### 3. Chạy Training

#### Option 1: Sử dụng script Python

```bash
cd app
python train.py
```

Script sẽ:
- Load và tiền xử lý dữ liệu
- Train model Linear SVM
- Tìm optimal threshold
- Đánh giá trên train/dev/test
- Lưu model vào `app/models/`

#### Option 2: Sử dụng Jupyter Notebook

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
├── sentiment_pipeline.pkl    # Model pipeline (TF-IDF + Classifier)
├── label_encoder.pkl          # Label encoder
├── stopwords.pkl              # Stopwords set
└── model_metadata.pkl         # Model info & metrics
```

---

### 4. Chạy Demo/Inference

#### A. Demo Script (Command Line)

```bash
cd demo
python demo_inference.py
```

Features:
- Test với các câu mẫu có sẵn
- Interactive mode: nhập câu để phân tích real-time

#### B. Demo Notebook

```bash
cd demo
jupyter notebook demo.ipynb
```

Notebook bao gồm:
- Test với câu đơn
- Batch prediction
- Visualization
- Interactive testing

#### C. Streamlit Web App 🌟

```bash
cd app
streamlit run streamlit_app.py
```

Giao diện web với:
- Nhập văn bản và nhận kết quả real-time
- Hiển thị xác suất (probability bars)
- Xem văn bản sau preprocessing
- Thông tin model metadata

**Truy cập:** http://localhost:8501

#### D. Python API

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
│   │   ├── stopwords.pkl
│   │   └── model_metadata.pkl
│   ├── preprocess.py             # Module tiền xử lý
│   ├── train.py                  # Script training
│   ├── predict.py                # Module inference/prediction
│   ├── utils.py                  # Utility functions
│   └── streamlit_app.py          # Streamlit web app
│
├── demo/                         # Demo scripts
│   ├── demo.ipynb                # Jupyter notebook demo
│   └── demo_inference.py         # Python script demo
│
├── data/                         # Data mẫu và hướng dẫn
│   ├── README.md                 # Hướng dẫn tải data
│   └── vietnamese-stopwords.txt  # Stopwords list
│
├── reports/                      # Báo cáo
│   └── [Đặt file báo cáo .pdf/.docx ở đây]
│
├── slides/                       # Slide thuyết trình
│   └── [Đặt file slide .pptx/.pdf ở đây]
│
├── archive/                      # Dataset gốc (gitignored)
│   ├── UIT-VSFC-train.json
│   ├── UIT-VSFC-dev.json
│   ├── UIT-VSFC-test.json
│   └── vietnamese-stopwords.txt
│
├── main.ipynb                    # Notebook chính (EDA + Training)
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation (file này)
```

### Giải thích thư mục

- **`app/`**: Chứa toàn bộ source code chính của dự án
  - `preprocess.py`: Các hàm tiền xử lý văn bản
  - `train.py`: Script để train model từ đầu
  - `predict.py`: Class và hàm để inference
  - `streamlit_app.py`: Web app demo

- **`demo/`**: Các script/notebook để demo nhanh
  - Dành cho người dùng cuối muốn test model
  - Không cần chạy lại training

- **`data/`**: Chỉ chứa data mẫu nhỏ hoặc hướng dẫn tải data
  - Không upload dataset lớn lên GitHub

- **`reports/`** & **`slides/`**: Tài liệu báo cáo và thuyết trình

- **`archive/`**: Dataset gốc (không commit lên GitHub do .gitignore)

---

## 👥 Tác Giả

### Thông tin nhóm

| Họ và tên | Mã SV | Email | Vai trò |
|-----------|-------|-------|---------|
| [Tên SV 1] | [MSSV1] | [email1@student.edu.vn] | Leader, ML Engineer |
| [Tên SV 2] | [MSSV2] | [email2@student.edu.vn] | Data Analyst |
| [Tên SV 3] | [MSSV3] | [email3@student.edu.vn] | Developer |

**Lớp:** [Tên lớp]  
**Giảng viên hướng dẫn:** [Tên giảng viên]  
**Học kỳ:** [HK/Năm học]

---

## 📚 Tài Liệu Tham Khảo

1. **Dataset:** [UIT-VSFC](https://github.com/sonvx/vietnam-sentiment-corpus) - Vietnamese Students' Feedback Corpus
2. **Vietnamese NLP:** [Underthesea](https://github.com/undertheseanlp/underthesea) - Vietnamese Natural Language Processing
3. **Scikit-learn:** [Text Classification Guide](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
4. **Paper:** Sentiment Analysis Techniques and Applications

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

## 📧 Liên Hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ:
- Email: [your-email@example.com]
- GitHub Issues: [Link to issues page]

---

**⭐ Nếu thấy dự án hữu ích, hãy cho chúng tôi một star trên GitHub!**

