# -*- coding: utf-8 -*-
"""
app.py — Ứng dụng Flask cho hệ thống định hướng nghề nghiệp (phiên bản mới).

QUY TRÌNH RA KHUYẾN NGHỊ:

    SINH VIÊN -> Tiền xử lý (StandardScaler) -> LOGISTIC REGRESSION (multinomial)
              -> xác suất 7 nhóm nghề -> điều chỉnh theo sở thích / kỹ năng mềm
              -> Career Group đứng đầu
              -> CBF (Cosine Similarity + Core Subject Mastery + kỹ năng mềm + feedback)
              -> TOP 3 NGHỀ

ĐIỂM THAY ĐỔI so với bản cũ (theo yêu cầu):
  1) Dữ liệu huấn luyện: DATASET.xlsx mới (bảng điểm 12 môn học thay vì khảo
     sát Python/SQL/Java + lĩnh vực + dự án). Xem prepare_data.py, train.py.
  2) Cách nhập liệu: sinh viên có thể TẢI FILE BẢNG ĐIỂM (.csv/.xlsx) hoặc
     NHẬP TAY điểm từng môn, đồng thời CHỌN KỸ NĂNG tự tin và SỞ THÍCH/lĩnh
     vực quan tâm — cả hai đều được đưa vào mô hình (xem build_feature_vector).
  3) Giao diện (HTML/CSS) làm lại hoàn toàn.
  Thuật toán Tầng 1 được train.py tự chọn theo F1 5-fold CV cao nhất; hiện là
  LOGISTIC REGRESSION (MULTINOMIAL). Lưu ý: nhãn huấn luyện là nhãn suy ra từ bảng
  điểm (xem README.md, mục "Giới hạn").
"""
import io
import json
import os
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for

import feedback_store
from cbf_recommender import cbf_recommender
from retrain_service import retrain_from_explicit_feedback
from domain_knowledge import (
    CAREER_SUBJECT_WEIGHTS, INTEREST_BOOST, INTEREST_OPTIONS, MODEL_LABELS,
    SKILL_LEVEL_CHOICES, SKILL_LEVEL_LABELS, SKILL_LEVEL_MULTIPLIERS, SKILL_OPTIONS,
    SOFT_SKILL_GROUP_BOOST, TOP_N_CAREERS,
    SUBJECT_KEYS, SUBJECTS, group_scores,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "doi-chuoi-nay-truoc-khi-nop-bao-cao")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

MODEL = joblib.load("model_best.joblib")

def model_label(model) -> str:
    """Tên thuật toán Tầng 1 đang được triển khai (đọc từ mô hình thật, không hard-code trong giao diện)."""
    step = model.steps[-1][1] if hasattr(model, "steps") else model
    return MODEL_LABELS.get(type(step).__name__, type(step).__name__)


with open("data/careers.json", encoding="utf-8") as f:
    CAREERS = json.load(f)

FEATURE_COLS = [SUBJECT_KEYS[s] for s in SUBJECTS]  # thứ tự cột dùng xuyên suốt hệ thống


COURSE_CODE_MAP = {
    "th4306": "Công nghệ phần mềm",
    "th4320": "Trí tuệ nhân tạo",
    "th5206": "Mạng máy tính",
    "th5209": "Xử lý ảnh",
    "th4316": "Công nghệ Java",
    "th5208": "Phân tích thiết kế HTTT",
    "th5221": "Hệ quản trị CSDL",
    "th4315": "Ngôn ngữ C# và công nghệ .NET",
    "th5210": "An toàn và bảo mật HTTT",
    "th5231": "Kỹ thuật đồ hoạ máy tính",
    "th5211": "Hệ điều hành Linux",
    "th4309": "Công nghệ web",
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    ).lower()


def _score_column(columns, normalized_columns):
    for marker in ("tbchp", "tbhp", "tong ket", "diem tb"):
        for column, normalized in zip(columns, normalized_columns):
            if marker in normalized:
                return column
    return None


def _optional_score(raw_score, subject: str):
    if pd.isna(raw_score) or not str(raw_score).strip() or _strip_accents(raw_score).strip() == "chua hoc":
        return None
    try:
        return float(str(raw_score).replace(",", ".").strip())
    except ValueError as exc:
        raise ValueError(f"Điểm môn '{subject}' không phải giá trị số hợp lệ.") from exc


# ------------------------------------------------------------------
# Đọc điểm từ file bảng điểm sinh viên tải lên (.csv hoặc .xlsx).
# Hỗ trợ 2 định dạng:
# 1) Bảng điểm tín chỉ dạng dọc (mỗi dòng là 1 môn học có cột TBCHP/Ký hiệu)
# 2) Bảng điểm dạng ngang (mỗi môn là 1 hoặc nhiều cột điểm thành phần)
# ------------------------------------------------------------------
def parse_score_file(file_storage) -> dict:
    """Đọc các điểm môn nhận diện được trong file .xlsx / .csv, thuộc [0, 10]."""
    filename = (file_storage.filename or "").lower()
    if not filename.endswith((".csv", ".xlsx")):
        raise ValueError("Chỉ hỗ trợ file Excel (.xlsx) hoặc CSV (.csv). Vui lòng chọn lại file hoặc nhập tay.")
    raw = file_storage.read()
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    else:
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    scores = _extract_scores(df)
    bad = [s for s, v in scores.items() if not (0 <= v <= 10)]
    if bad:
        raise ValueError("Điểm phải nằm trong khoảng 0-10; kiểm tra lại các môn: " + ", ".join(bad))
    return scores


def _extract_scores(df) -> dict:

    col_names = [str(c).strip() for c in df.columns]
    norm_cols = [_strip_accents(c) for c in col_names]

    # 1. Định dạng bảng điểm dạng dọc (mỗi dòng là 1 môn học — như cổng đào tạo đại học)
    score_col = _score_column(df.columns, norm_cols)
    has_subj_col = any("ten hoc phan" in c or "mon hoc" in c or "ky hieu" in c or "ma mon" in c or "hoc phan" in c for c in norm_cols)

    if score_col is not None and has_subj_col:
        tbchp_col = score_col
        name_col = next((c for c, nc in zip(df.columns, norm_cols) if "ten hoc phan" in nc or "mon hoc" in nc or "hoc phan" in nc), None)
        code_col = next((c for c, nc in zip(df.columns, norm_cols) if "ky hieu" in nc or "ma mon" in nc), None)

        scores = {}
        for _, row in df.iterrows():
            c_name = _strip_accents(row[name_col]) if name_col and pd.notna(row[name_col]) else ""
            c_code = _strip_accents(row[code_col]) if code_col and pd.notna(row[code_col]) else ""

            matched_subj = None
            # Ưu tiên mã môn học THxxxx
            for code, sname in COURSE_CODE_MAP.items():
                if code in c_code or code in c_name:
                    matched_subj = sname
                    break
            if not matched_subj:
                for sname in SUBJECTS:
                    if _strip_accents(sname) in c_name:
                        matched_subj = sname
                        break
            if matched_subj and matched_subj not in scores:
                val = _optional_score(row[tbchp_col], matched_subj)
                if val is not None:
                    scores[matched_subj] = val

        if scores:
            return scores

    # 2. Định dạng bảng điểm dạng ngang (mỗi môn là 1 hoặc nhiều cột)
    col_map = {subj: [] for subj in SUBJECTS}
    for col in df.columns:
        key = str(col).strip()
        norm_key = _strip_accents(key)
        for subj in SUBJECTS:
            if key == subj or key == SUBJECT_KEYS[subj] or norm_key == _strip_accents(subj) or _strip_accents(subj) in norm_key:
                col_map[subj].append(col)
                break

    available = {subj: cols for subj, cols in col_map.items() if cols}
    if not available:
        raise ValueError("Không tìm thấy cột điểm môn học trong file. Vui lòng kiểm tra file hoặc nhập tay.")
    row = df.iloc[0]
    scores = {}
    for subj, cols in available.items():
        if len(cols) == 1:
            score = _optional_score(row[cols[0]], subj)
            if score is not None:
                scores[subj] = score
        else:
            # Always prefer TBCHP, even if a similar total-score column appears first.
            normalized_subject_cols = [_strip_accents(str(column)) for column in cols]
            tb_col = _score_column(cols, normalized_subject_cols)
            if tb_col is not None:
                score = _optional_score(row[tb_col], subj)
                if score is not None:
                    scores[subj] = score
            else:
                # Công thức chuẩn tín chỉ quy chuẩn (QT 40% + Thi 60%)
                v1 = _optional_score(row[cols[0]], subj)
                v2 = _optional_score(row[cols[1]], subj)
                if v1 is not None and v2 is not None:
                    scores[subj] = round(v1 * 0.4 + v2 * 0.6, 1)
    if not scores:
        raise ValueError("Không tìm thấy điểm hợp lệ. Vui lòng cung cấp điểm ít nhất một môn trong danh sách.")
    return scores


def build_feature_vector(scores: dict) -> np.ndarray:
    """Tạo vector mô hình; môn chưa có điểm được điền bằng trung bình huấn luyện."""
    preprocessor = MODEL.named_steps.get("preprocess")
    reference_scores = getattr(preprocessor, "mean_", cbf_recommender.mu)
    return np.array([
        float(scores[subject]) if subject in scores else float(reference_scores[index])
        for index, subject in enumerate(SUBJECTS)
    ])


def apply_interest_and_skill_boost(ranked_groups: list, interests_selected: list, skill_levels: dict = None) -> list:
    """Boost xác suất nhóm nghề theo sở thích và cấp độ kỹ năng mềm đã đánh giá."""
    boosted = []
    for group, probability in ranked_groups:
        factor = 1.0
        if interests_selected and group in interests_selected:
            factor += INTEREST_BOOST
        if skill_levels:
            for skill, level in skill_levels.items():
                multiplier = SKILL_LEVEL_MULTIPLIERS.get(level, 0.08)
                affinity = SOFT_SKILL_GROUP_BOOST.get(skill, {}).get(group, 0.0)
                factor += affinity * multiplier * 3.0
        boosted.append((group, probability * factor))
    total = sum(probability for _, probability in boosted) or 1.0
    ranked = [(group, probability / total) for group, probability in boosted]
    return sorted(ranked, key=lambda item: -item[1])


def recommend_next_subjects(group_name: str, scores: dict, limit: int = 3) -> list:
    """Return highest-priority not-yet-graded subjects from the careers in a group."""
    group_career_ids = [
        career_id for career_id, career in CAREERS.items()
        if career.get("career_group") == group_name
    ]
    subject_weights = {}
    for career_id in group_career_ids:
        for subject, weight in CAREER_SUBJECT_WEIGHTS.get(career_id, {}).items():
            if subject not in scores:
                subject_weights[subject] = subject_weights.get(subject, 0.0) + weight
    return [
        subject for subject, _ in sorted(subject_weights.items(), key=lambda item: -item[1])[:limit]
    ]


@app.route("/")
@app.route("/survey")
def home():
    return render_template(
        "index.html",
        subjects=[{"key": SUBJECT_KEYS[s], "label": s} for s in SUBJECTS],
        skill_options=SKILL_OPTIONS,
        skill_levels=SKILL_LEVEL_CHOICES,
        interest_options=INTEREST_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    try:
        name = form.get("name", "").strip() or "Bạn"
        input_mode = form.get("input_mode", "manual")

        if input_mode == "file":
            # Chỉ đọc file .xlsx / .csv do người dùng tải lên (không có ảnh/OCR, không có điểm mặc định).
            uploaded = request.files.get("score_file")
            if not uploaded or uploaded.filename == "":
                raise ValueError("Bạn chưa chọn file bảng điểm (.xlsx hoặc .csv) để tải lên.")
            scores = parse_score_file(uploaded)
        else:
            scores = {}
            for subj in SUBJECTS:
                key = SUBJECT_KEYS[subj]
                raw_val = form.get(f"score_{key}")
                if raw_val is None or raw_val.strip() == "":
                    continue
                val = float(raw_val)
                if not (0 <= val <= 10):
                    raise ValueError(f"Điểm môn '{subj}' phải trong khoảng 0-10.")
                scores[subj] = val
            if not scores:
                raise ValueError("Vui lòng nhập điểm ít nhất một môn đã học.")

        # Thu thập cấp độ của từng kỹ năng mềm từ form. Kỹ năng để "Chưa đánh giá" (giá trị rỗng)
        # bị BỎ QUA hoàn toàn — không cộng/trừ điểm — thay vì mặc định coi là "Trung bình".
        skill_levels = {}
        for s in SKILL_OPTIONS:
            k = s["key"]
            val = form.get(f"skill_level_{k}")
            if not val and k in form.getlist("skills"):  # tương thích form cũ dạng checkbox
                val = "strong"
            if val in SKILL_LEVEL_LABELS:
                skill_levels[k] = val

        interests_selected = form.getlist("interests")

        feature_vec = build_feature_vector(scores)
        student_features = pd.DataFrame([feature_vec], columns=FEATURE_COLS)
        probabilities = MODEL.predict_proba(student_features)[0]
        ranked_groups = sorted(zip(MODEL.classes_, probabilities), key=lambda item: -item[1])
        ranked_groups = apply_interest_and_skill_boost(ranked_groups, interests_selected, skill_levels)
        top_group, top_prob = ranked_groups[0]

        # Bước 4: Xếp hạng (Content-Based Filtering — CBF) và chọn TOP 3 NGHỀ.
        # Xếp hạng nghề trong nhóm hàng đầu theo điểm CBF (có tích hợp kỹ năng mềm theo cấp độ), lấy
        # TOP_N_CAREERS nghề đầu; nếu nhóm có ít hơn 3 nghề thì bổ sung từ nhóm xếp hạng kế tiếp.
        ranked_careers, group_remaining = cbf_recommender.recommend_top_careers(
            ranked_groups, scores, skill_levels, n=TOP_N_CAREERS)

        # Xếp hạng toàn bộ nghề cụ thể của các nhóm khác để trợ giúp người dùng mở rộng lựa chọn
        other_groups_ranked = cbf_recommender.rank_all_groups(ranked_groups[1:4], scores, skill_levels)

        # Bước 5: Sinh khuyến nghị & Bổ sung trạng thái quan tâm từ cộng đồng
        engagement = feedback_store.get_career_engagement()
        for career in ranked_careers + group_remaining:
            net_likes, views = engagement.get(career["career_id"], (0, 0))
            career["is_popular"] = (net_likes, views) > (0, 0)
            career["name"] = career.get("career_name")

        for grp in other_groups_ranked:
            for career in grp["careers"]:
                net_likes, views = engagement.get(career["career_id"], (0, 0))
                career["is_popular"] = (net_likes, views) > (0, 0)
                career["name"] = career.get("career_name")

        group_breakdown = sorted(group_scores(scores).items(), key=lambda x: -x[1])
        next_subjects = recommend_next_subjects(top_group, scores)

        skill_summary = [
            {
                "key": s["key"],
                "label": s["label"],
                "level": skill_levels[s["key"]],
                "level_label": SKILL_LEVEL_LABELS[skill_levels[s["key"]]],
            }
            for s in SKILL_OPTIONS
            if s["key"] in skill_levels
        ]

        # Lưu phiên làm việc cho Bước 6 & 7 (phản hồi tường minh và huấn luyện lại)
        session["last_result"] = {
            "name": name,
            "top_group": top_group,
            "top_prob": round(top_prob * 100, 1),
            "career_ids": [c["career_id"] for c in ranked_careers],
            "scores": scores,
            "skill_levels": skill_levels,
        }

        # Bước 6: Trợ giúp người dùng lựa chọn (giao diện hiển thị so sánh & phản hồi)
        return render_template(
            "result.html",
            name=name,
            top_group=top_group,
            top_prob=round(top_prob * 100, 1),
            graded_subject_count=len(scores),
            model_name=model_label(MODEL),
            careers=ranked_careers,
            group_remaining=group_remaining,
            other_groups=other_groups_ranked,
            group_breakdown=group_breakdown,
            skill_summary=skill_summary,
            skill_levels=skill_levels,
            interests_selected=interests_selected,
            scores=scores,
            next_subjects=next_subjects,
        )
    except Exception as e:
        return render_template("error.html", message=str(e))


@app.route("/feedback", methods=["POST"])
def feedback():
    last = session.get("last_result", {})
    rating = int(request.form.get("rating", 0))
    liked = request.form.getlist("like")
    disliked = request.form.getlist("dislike")
    comment = request.form.get("comment", "")

    feedback_store.add_feedback(
        top_group=last.get("top_group", "Không rõ"),
        top_prob=last.get("top_prob", 0),
        rating=rating,
        liked_careers=liked,
        disliked_careers=disliked,
        comment=comment,
        scores=last.get("scores", {}),
        student_name=last.get("name", "Sinh viên"),
    )
    # Tự động cập nhật hệ thống CBF khi có phản hồi tường minh mới
    cbf_recommender.reload()
    return render_template("feedback_thanks.html")


@app.route("/track", methods=["POST"])
def track():
    data = request.get_json(silent=True) or {}
    last = session.get("last_result", {})
    feedback_store.add_interaction(
        event_type=data.get("event_type", "unknown"),
        top_group=last.get("top_group", "Không rõ"),
        career_id=data.get("career_id"),
        duration_seconds=data.get("duration_seconds"),
    )
    return ("", 204)


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["is_admin"] = True
        else:
            return render_template("admin_login.html", error="Sai mật khẩu.")

    if not session.get("is_admin"):
        return render_template("admin_login.html", error=None)

    all_feedback = feedback_store.get_all_feedback()
    summary = feedback_store.summarize_interactions()

    ratings = [f["rating"] for f in all_feedback if f.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    views_named = [
        {"career_id": cid, "name": CAREERS.get(cid, {}).get("career_name", cid), "views": n}
        for cid, n in summary["views_by_career"]
    ]

    retrain_message = session.pop("retrain_message", None)
    ready_count = sum(
        1 for f in all_feedback
        if len(f.get("scores") or {}) == len(SUBJECTS) and f.get("liked_careers")
    )

    return render_template(
        "admin.html",
        feedback_list=all_feedback,
        avg_rating=avg_rating,
        total_feedback=len(all_feedback),
        views_named=views_named,
        avg_duration=summary["avg_duration_seconds"],
        total_time_events=summary["total_time_events"],
        retrain_message=retrain_message,
        ready_count=ready_count,
    )


@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    global MODEL
    result = retrain_from_explicit_feedback()
    MODEL = joblib.load("model_best.joblib")
    session["retrain_message"] = result.get("message")
    return redirect(url_for("admin"))


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin"))
        else:
            return render_template("admin_login.html", error="Sai mật khẩu.")
    return render_template("admin_login.html", error=None)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin"))


if __name__ == "__main__":
    app.run(debug=True)
