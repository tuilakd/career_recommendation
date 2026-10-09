# -*- coding: utf-8 -*-
"""
retrain_service.py — Dịch vụ tự động huấn luyện lại mô hình dựa trên phản hồi tường minh (Explicit Feedback).

CƠ CHẾ:
  1. Đọc data/feedback.json, lọc các bản ghi có đánh giá "Phù hợp" (liked_careers)
     kèm theo bảng điểm 12 môn của sinh viên.
  2. Ánh xạ các nghề được đánh giá "Phù hợp" về nhóm nghề thực tế (ground-truth do người dùng xác nhận).
  3. Ghép các mẫu dữ liệu mới này vào tập dữ liệu huấn luyện (loại bỏ trùng lặp) và gắn
     label_source = "feedback" để phân biệt với nhãn suy ra từ bảng điểm (label_source = "derived").
  4. Huấn luyện lại ĐÚNG LOẠI mô hình đang triển khai (nhân bản pipeline trong model_best.joblib qua
     sklearn.base.clone; mặc định Logistic Regression nếu chưa có mô hình) và lưu đè vào model_best.joblib.
     Nhờ vậy thuật toán được chọn bởi train.py không bị âm thầm đổi khi huấn luyện lại.
  5. Kích hoạt cbf_recommender.reload() để cập nhật lại phản hồi của CBF (mốc chuẩn hoá z-score của CBF
     chỉ tính trên các dòng "derived" nên không bị dịch chuyển bởi dòng phản hồi).
"""
import json
import os
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cbf_recommender import cbf_recommender
from domain_knowledge import (
    LABEL_DERIVED, LABEL_FEEDBACK, LABEL_SOURCE_COL, SUBJECT_KEYS, SUBJECTS,
)

DATA_PATH = "data/students.csv"
FEEDBACK_PATH = "data/feedback.json"
CAREERS_PATH = "data/careers.json"
MODEL_PATH = "model_best.joblib"
FEATURE_COLS = [SUBJECT_KEYS[s] for s in SUBJECTS]
TARGET_COL = "career_group"


def _default_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def _current_pipeline() -> Pipeline:
    """Pipeline (chưa fit) cùng loại với mô hình đang triển khai; mặc định Logistic Regression."""
    if os.path.exists(MODEL_PATH):
        try:
            return clone(joblib.load(MODEL_PATH))
        except Exception:
            pass
    return _default_pipeline()


def retrain_from_explicit_feedback() -> dict:
    """
    Huấn luyện lại mô hình Tầng 1 (cùng thuật toán đang triển khai) và cập nhật hệ thống CBF từ phản hồi tường minh.
    """
    if not os.path.exists(FEEDBACK_PATH):
        return {
            "status": "warning",
            "samples_added": 0,
            "message": "Chưa có file phản hồi người dùng nào (data/feedback.json).",
        }

    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        feedbacks = json.load(f)

    with open(CAREERS_PATH, encoding="utf-8") as f:
        careers_meta = json.load(f)

    # 1. Trích xuất các mẫu học mới từ phản hồi tường minh
    new_rows = []
    for idx, fb in enumerate(feedbacks, 1):
        scores = fb.get("scores")
        liked = fb.get("liked_careers", [])

        if not scores or not liked or len(scores) < len(SUBJECTS):
            continue

        # Lấy nhóm nghề của các nghề được người dùng like
        confirmed_groups = set()
        for cid in liked:
            if cid in careers_meta:
                grp = careers_meta[cid].get("career_group")
                if grp:
                    confirmed_groups.add(grp)

        # Với mỗi nhóm nghề được người dùng xác nhận qua like
        for grp in confirmed_groups:
            row = {"name": f"Phản hồi #{idx} ({fb.get('rating', 5)}★)"}
            for s in SUBJECTS:
                row[SUBJECT_KEYS[s]] = float(scores.get(s, 7.0))
            row[TARGET_COL] = grp
            row[LABEL_SOURCE_COL] = LABEL_FEEDBACK
            new_rows.append(row)

    # 2. Đọc tập dữ liệu hiện tại
    df_base = pd.read_csv(DATA_PATH)
    if LABEL_SOURCE_COL not in df_base.columns:
        df_base[LABEL_SOURCE_COL] = LABEL_DERIVED
    df_base[LABEL_SOURCE_COL] = df_base[LABEL_SOURCE_COL].fillna(LABEL_DERIVED)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Hợp nhất và loại bỏ các dòng bị trùng lặp hoàn toàn về điểm và nhóm (giữ dòng gốc, đứng trước)
        df_merged = pd.concat([df_base, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
        samples_added = len(df_merged) - len(df_base)
    else:
        df_merged = df_base
        samples_added = 0

    X = df_merged[FEATURE_COLS]
    y = df_merged[TARGET_COL]

    # 3. Huấn luyện lại cùng loại mô hình đang triển khai
    pipeline = _current_pipeline()
    algo_name = type(pipeline.named_steps["model"]).__name__

    # Đánh giá 5-fold CV (độ khớp với nhãn suy ra + nhãn phản hồi, KHÔNG phải độ chính xác dự đoán thực tế)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(clone(pipeline), X, y, cv=skf, scoring="f1_weighted").mean()

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    # Cập nhật lại file data/students.csv (luôn ghi để bổ sung cột label_source nếu file cũ chưa có)
    df_merged.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    # 4. Tải lại Content-Based Filtering Recommender
    cbf_recommender.reload()

    return {
        "status": "success",
        "samples_added": samples_added,
        "total_samples": len(df_merged),
        "algorithm": algo_name,
        "f1_score": round(cv_score, 4),
        "message": f"Huấn luyện lại thành công ({algo_name})! Đã tích hợp {samples_added} mẫu từ phản hồi tường minh. "
                   f"F1 (5-fold CV, khớp với nhãn huấn luyện) = {cv_score*100:.1f}%.",
    }
