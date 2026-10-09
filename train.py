# -*- coding: utf-8 -*-
"""
train.py — Huấn luyện & so sánh 3 thuật toán phân loại nhóm nghề (Tầng 1), tự động
chọn mô hình có F1 5-fold CV cao nhất để triển khai và lưu vào model_best.joblib.

VỀ Ý NGHĨA CỦA CÁC CHỈ SỐ (đọc kỹ trước khi đưa vào báo cáo):
  * Nhãn `career_group` trong data/students.csv được SUY RA từ chính 12 điểm môn học
    bằng công thức (domain_knowledge.dominant_group) — là nhãn giả (pseudo-label).
    Vì vậy Accuracy/F1 ở bảng so sánh chỉ đo "mô hình xấp xỉ lại công thức gán nhãn
    tốt đến đâu", KHÔNG phải độ chính xác dự đoán nghề nghiệp thực tế.
  * Chỉ số độc lập duy nhất là `validate_on_feedback()`: mô hình (chỉ học từ nhãn suy
    ra) được kiểm tra trên nhóm nghề mà NGƯỜI DÙNG thật đã xác nhận "phù hợp".
  * Bảng so sánh chỉ dùng các dòng nhãn suy ra (label_source == "derived"). Khi lưu mô
    hình triển khai, dữ liệu huấn luyện gồm cả các dòng do phản hồi người dùng bổ sung.
"""
import json
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from domain_knowledge import (
    LABEL_DERIVED, LABEL_FEEDBACK, LABEL_SOURCE_COL, MIN_FEEDBACK_FOR_VALIDATION,
    SUBJECT_KEYS, SUBJECTS,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/students.csv"
FEEDBACK_PATH = "data/feedback.json"
CAREERS_PATH = "data/careers.json"
MODEL_PATH = "model_best.joblib"
TARGET_COL = "career_group"

FEATURE_COLS = [SUBJECT_KEYS[s] for s in SUBJECTS]  # 12 cột điểm môn học


def load_data():
    df = pd.read_csv(DATA_PATH)
    if LABEL_SOURCE_COL not in df.columns:
        df[LABEL_SOURCE_COL] = LABEL_DERIVED
    df[LABEL_SOURCE_COL] = df[LABEL_SOURCE_COL].fillna(LABEL_DERIVED)
    return df


def derived_only(df):
    """Các dòng có nhãn suy ra từ bảng điểm (loại các dòng do phản hồi người dùng bổ sung)."""
    return df[df[LABEL_SOURCE_COL] != LABEL_FEEDBACK].reset_index(drop=True)


def build_preprocessor():
    # Chỉ còn đặc trưng số (điểm 0-10) nên chỉ cần chuẩn hoá StandardScaler.
    return StandardScaler()


def build_models():
    return {
        "Logistic Regression (multinomial)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }


def validate_on_feedback(fitted_pipeline, feedback_path=FEEDBACK_PATH, careers_path=CAREERS_PATH):
    """
    Kiểm định ĐỘC LẬP (không vòng tròn): với mỗi phản hồi có bảng điểm và nghề được người dùng
    bấm "phù hợp", nhóm nghề của các nghề đó là nhóm "được xác nhận". Đo xem mô hình
    (phải là mô hình CHỈ học từ nhãn suy ra, chưa từng thấy các phản hồi này) có xếp nhóm
    được xác nhận ở Top-1 / Top-3 hay không.
    Trả về dict {n, top1, top3}; n = số phản hồi hợp lệ.
    """
    result = {"n": 0, "top1": 0, "top3": 0}
    if not (os.path.exists(feedback_path) and os.path.exists(careers_path)):
        return result
    with open(feedback_path, encoding="utf-8") as f:
        feedbacks = json.load(f)
    with open(careers_path, encoding="utf-8") as f:
        careers = json.load(f)

    for fb in feedbacks:
        scores, liked = fb.get("scores"), fb.get("liked_careers", [])
        if not scores or not liked or any(s not in scores for s in SUBJECTS):
            continue
        confirmed = {careers[c]["career_group"] for c in liked if c in careers}
        if not confirmed:
            continue
        x = pd.DataFrame([[float(scores[s]) for s in SUBJECTS]], columns=FEATURE_COLS)
        proba = fitted_pipeline.predict_proba(x)[0]
        ranked = [g for g, _ in sorted(zip(fitted_pipeline.classes_, proba), key=lambda t: -t[1])]
        result["n"] += 1
        result["top1"] += ranked[0] in confirmed
        result["top3"] += any(g in confirmed for g in ranked[:3])
    return result


def print_feedback_validation(res):
    print("\n--- KIỂM ĐỊNH ĐỘC LẬP: khớp với nhóm nghề do NGƯỜI DÙNG xác nhận ---")
    n = res["n"]
    if n < MIN_FEEDBACK_FOR_VALIDATION:
        print(f"Chưa đủ phản hồi để kết luận (có {n}, cần tối thiểu {MIN_FEEDBACK_FOR_VALIDATION}). "
              "Đây là lý do các chỉ số ở trên KHÔNG nên trình bày như độ chính xác dự đoán thực tế.")
    else:
        print(f"Số phản hồi hợp lệ: {n} | Top-1: {res['top1'] / n * 100:.1f}% | Top-3: {res['top3'] / n * 100:.1f}%")


def main():
    df_all = load_data().dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    df = derived_only(df_all)
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    print(f"Số dòng nhãn suy ra dùng để so sánh: {len(df)}  |  Số nhóm nghề: {y.nunique()}"
          f"  |  Dòng từ phản hồi người dùng (chỉ dùng khi huấn luyện bản triển khai): {len(df_all) - len(df)}")

    models = build_models()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Đánh giá cả 3 mô hình ---
    rows, cv_scores = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cv_pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        f1cv = cross_val_score(cv_pipe, X, y, cv=skf, scoring="f1_weighted").mean()
        cv_scores[name] = f1cv
        rows.append((
            name,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average="weighted", zero_division=0),
            recall_score(y_test, y_pred, average="weighted", zero_division=0),
            f1_score(y_test, y_pred, average="weighted", zero_division=0),
            f1cv,
        ))

    # Tự động chốt mô hình có F1 5-fold CV cao nhất
    production_model_name = max(cv_scores, key=cv_scores.get)

    print("\n" + "=" * 80)
    print("ĐỘ KHỚP VỚI NHÃN SUY RA TỪ BẢNG ĐIỂM (không phải độ chính xác dự đoán nghề thực tế)")
    print("=" * 80)
    print(f"{'Mô hình':<36}{'Accuracy':>10}{'Precision':>10}{'Recall':>10}{'F1(test)':>10}{'F1(5CV)':>10}")
    print("-" * 80)
    for name, acc, prec, rec, f1t, f1cv in rows:
        marker = "  <-- triển khai" if name == production_model_name else ""
        print(f"{name:<36}{acc:>10.3f}{prec:>10.3f}{rec:>10.3f}{f1t:>10.3f}{f1cv:>10.3f}{marker}")
    print("=" * 80)

    # --- Mô hình triển khai: đánh giá trên tập test, rồi kiểm định độc lập bằng phản hồi thật ---
    production_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", build_models()[production_model_name]),
    ])
    production_pipeline.fit(X_train, y_train)
    print(f"\n>>> Mô hình triển khai (F1 5-fold CV cao nhất): {production_model_name} "
          f"(F1 5-fold CV = {cv_scores[production_model_name]:.3f})")
    print(classification_report(y_test, production_pipeline.predict(X_test), zero_division=0))
    print_feedback_validation(validate_on_feedback(production_pipeline))

    # --- Fit lại trên toàn bộ dữ liệu (nhãn suy ra + phản hồi người dùng nếu có) rồi lưu ---
    production_pipeline.fit(df_all[FEATURE_COLS], df_all[TARGET_COL])
    joblib.dump(production_pipeline, MODEL_PATH)
    print(f"\nĐã lưu mô hình {production_model_name} (đã gồm cả tiền xử lý) vào: {MODEL_PATH}")


if __name__ == "__main__":
    main()
