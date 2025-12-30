"""
Training Module
Module huấn luyện mô hình phân tích cảm xúc tiếng Việt
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from preprocess import load_stopwords, preprocess_dataframe
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str = '../archive'):
    """
    Load dữ liệu từ file JSON
    
    Parameters:
    -----------
    data_path : str
        Đường dẫn tới thư mục chứa data
    
    Returns:
    --------
    dict : Dictionary chứa train, dev, test DataFrames
    """
    print("📂 Đang tải dữ liệu...")
    
    df = {}
    for split in ['train', 'dev', 'test']:
        filepath = f'{data_path}/UIT-VSFC-{split}.json'
        df[split] = pd.read_json(filepath)
        
        # Lọc chỉ lấy topic 'lecturer' và bỏ 'neutral'
        df[split] = df[split][
            (df[split]['topic'] == 'lecturer') & 
            (df[split]['sentiment'] != 'neutral')
        ].drop('topic', axis=1)
        
        df[split].reset_index(drop=True, inplace=True)
        print(f"  ✅ {split.capitalize()}: {df[split].shape[0]} mẫu")
    
    return df


def encode_labels(df: dict):
    """
    Encode nhãn sentiment thành số
    
    Parameters:
    -----------
    df : dict
        Dictionary chứa train, dev, test DataFrames
    
    Returns:
    --------
    LabelEncoder : Encoder đã được fit
    """
    print("\n🔢 Đang encode nhãn...")
    
    label_encoder = LabelEncoder()
    df['train']['sentiment_encoded'] = label_encoder.fit_transform(df['train']['sentiment'])
    df['dev']['sentiment_encoded'] = label_encoder.transform(df['dev']['sentiment'])
    df['test']['sentiment_encoded'] = label_encoder.transform(df['test']['sentiment'])
    
    print(f"  Mapping: negative={label_encoder.transform(['negative'])[0]}, positive={label_encoder.transform(['positive'])[0]}")
    
    return label_encoder


def create_pipeline(model_type='svm'):
    """
    Tạo pipeline cho model
    
    Parameters:
    -----------
    model_type : str
        Loại model: 'svm', 'lr', 'nb'
    
    Returns:
    --------
    Pipeline : Sklearn pipeline
    """
    # Chọn model
    if model_type == 'svm':
        classifier = LinearSVC(
            C=1.0, 
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        )
    elif model_type == 'lr':
        classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'nb':
        classifier = MultinomialNB(alpha=1.0)
    else:
        raise ValueError(f"Model type '{model_type}' không hợp lệ")
    
    # Tạo pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ('clf', classifier)
    ])
    
    return pipeline


def find_optimal_threshold(model, X_dev, y_dev):
    """
    Tìm ngưỡng tối ưu để maximize F1-score trên dev set
    
    Parameters:
    -----------
    model : sklearn model
        Model đã được train
    X_dev : array-like
        Dev set features
    y_dev : array-like
        Dev set labels
    
    Returns:
    --------
    float : Ngưỡng tối ưu
    """
    # Lấy decision function hoặc probability
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_dev)
        # Convert to probability-like scores
        scores = 1 / (1 + np.exp(-scores))
    else:
        scores = model.predict_proba(X_dev)[:, 1]
    
    best_f1 = 0
    best_threshold = 0.5
    
    # Thử các ngưỡng từ 0.3 đến 0.7
    for threshold in np.arange(0.3, 0.8, 0.01):
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_dev, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def evaluate_model(model, X, y, threshold=0.5):
    """
    Đánh giá model
    
    Parameters:
    -----------
    model : sklearn model
        Model đã được train
    X : array-like
        Features
    y : array-like
        True labels
    threshold : float
        Ngưỡng để classify
    
    Returns:
    --------
    dict : Dictionary chứa các metrics
    """
    # Dự đoán
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        scores = 1 / (1 + np.exp(-scores))
        y_pred = (scores >= threshold).astype(int)
    else:
        probs = model.predict_proba(X)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    
    # Tính metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred
    }


def train_and_save_model(data_path='../archive', 
                         output_dir='models',
                         model_type='svm'):
    """
    Hàm chính để train và lưu model
    
    Parameters:
    -----------
    data_path : str
        Đường dẫn tới thư mục chứa data
    output_dir : str
        Thư mục để lưu models
    model_type : str
        Loại model: 'svm', 'lr', 'nb'
    """
    print("="*60)
    print("🚀 BẮT ĐẦU TRAINING MODEL")
    print("="*60)
    
    # 1. Load dữ liệu
    df = load_data(data_path)
    
    # 2. Encode labels
    label_encoder = encode_labels(df)
    
    # 3. Tiền xử lý
    print("\n🔧 Đang tiền xử lý văn bản...")
    stopwords = load_stopwords(f'{data_path}/vietnamese-stopwords.txt')
    
    for split in ['train', 'dev', 'test']:
        df[split] = preprocess_dataframe(df[split], 'sentence', stopwords)
        print(f"  ✅ {split.capitalize()}: Hoàn thành")
    
    # 4. Tạo model và train
    print(f"\n🤖 Đang train model ({model_type.upper()})...")
    pipeline = create_pipeline(model_type)
    
    X_train = df['train']['sentence_processed']
    y_train = df['train']['sentiment_encoded']
    X_dev = df['dev']['sentence_processed']
    y_dev = df['dev']['sentiment_encoded']
    X_test = df['test']['sentence_processed']
    y_test = df['test']['sentiment_encoded']
    
    pipeline.fit(X_train, y_train)
    print("  ✅ Training hoàn thành!")
    
    # 5. Tìm optimal threshold
    print("\n🎯 Đang tìm optimal threshold...")
    optimal_threshold = find_optimal_threshold(pipeline, X_dev, y_dev)
    print(f"  ✅ Optimal threshold: {optimal_threshold:.4f}")
    
    # 6. Đánh giá
    print("\n📊 ĐÁNH GIÁ MODEL")
    print("-" * 60)
    
    for split_name, X, y in [('Train', X_train, y_train),
                              ('Dev', X_dev, y_dev),
                              ('Test', X_test, y_test)]:
        results = evaluate_model(pipeline, X, y, optimal_threshold)
        print(f"{split_name:6s} | Accuracy: {results['accuracy']:.4f} | F1-Score: {results['f1_score']:.4f}")
    
    # 7. Lưu model
    print(f"\n💾 Đang lưu model vào {output_dir}/...")
    
    # Tạo thư mục nếu chưa có
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu các file
    joblib.dump(pipeline, f'{output_dir}/sentiment_pipeline.pkl')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    joblib.dump(stopwords, f'{output_dir}/stopwords.pkl')
    
    # Lưu metadata
    metadata = {
        'model_type': model_type,
        'model_name': 'Linear SVM' if model_type == 'svm' else 'Logistic Regression' if model_type == 'lr' else 'Naive Bayes',
        'optimal_threshold': optimal_threshold,
        'f1_score': evaluate_model(pipeline, X_test, y_test, optimal_threshold)['f1_score'],
        'accuracy': evaluate_model(pipeline, X_test, y_test, optimal_threshold)['accuracy']
    }
    joblib.dump(metadata, f'{output_dir}/model_metadata.pkl')
    
    print("  ✅ sentiment_pipeline.pkl")
    print("  ✅ label_encoder.pkl")
    print("  ✅ stopwords.pkl")
    print("  ✅ model_metadata.pkl")
    
    print("\n" + "="*60)
    print("✨ HOÀN THÀNH!")
    print("="*60)
    
    return pipeline, label_encoder, metadata


if __name__ == "__main__":
    # Train model
    pipeline, label_encoder, metadata = train_and_save_model(
        data_path='../archive',
        output_dir='models',
        model_type='svm'  # Có thể thay bằng 'lr' hoặc 'nb'
    )
    
    print("\n✅ Có thể chạy demo bằng: streamlit run streamlit_app.py")

