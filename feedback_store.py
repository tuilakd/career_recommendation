# -*- coding: utf-8 -*-
"""
feedback_store.py — Lưu và đọc lại phản hồi người dùng
"""
import json
import os
import threading
import uuid
from datetime import datetime

FEEDBACK_PATH = "data/feedback.json"
INTERACTIONS_PATH = "data/interactions.json"


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# Khoá chống 2 request ghi cùng lúc làm mất bản ghi (đọc-sửa-ghi phải liền mạch).
_LOCK = threading.Lock()


def _save(path, records):
    """Ghi ra file tạm rồi thay thế file thật, để nếu lỗi giữa chừng thì file cũ
    không bị hỏng dở."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------- Explicit Feedback (đánh giá tường minh) ----------------

def add_feedback(top_group: str, top_prob: float, rating: int,
                  liked_careers: list, disliked_careers: list, comment: str,
                  scores: dict = None, student_name: str = "") -> dict:
    """Lưu 1 lượt đánh giá tường minh: rating 1-5 sao + like/dislike theo từng
    nghề được gợi ý + bình luận tự do + bảng điểm scores của phiên đó."""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student_name": student_name.strip() if student_name else "Sinh viên",
        "top_group": top_group,
        "top_prob": top_prob,
        "rating": rating,
        "liked_careers": liked_careers,
        "disliked_careers": disliked_careers,
        "comment": comment.strip() if comment else "",
        "scores": scores or {},
    }
    with _LOCK:
        records = _load(FEEDBACK_PATH)
        records.append(record)
        _save(FEEDBACK_PATH, records)
    return record


def get_all_feedback() -> list:
    return sorted(_load(FEEDBACK_PATH), key=lambda r: r["timestamp"], reverse=True)


# ---------------- Implicit Feedback (đánh giá ngầm qua hành vi) ----------------

def add_interaction(event_type: str, top_group: str, career_id: str | None = None,
                     duration_seconds: float | None = None) -> dict:
    """Lưu 1 sự kiện hành vi ngầm: mở xem chi tiết 1 nghề (view_career) hoặc
    tổng thời gian xem trang kết quả (time_on_result)."""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,
        "top_group": top_group,
        "career_id": career_id,
        "duration_seconds": duration_seconds,
    }
    with _LOCK:
        records = _load(INTERACTIONS_PATH)
        records.append(record)
        _save(INTERACTIONS_PATH, records)
    return record


def get_all_interactions() -> list:
    return sorted(_load(INTERACTIONS_PATH), key=lambda r: r["timestamp"], reverse=True)


def summarize_interactions() -> dict:
    """Tổng hợp nhanh cho trang Admin: số lượt xem theo từng nghề, thời gian
    xem trang trung bình."""
    records = _load(INTERACTIONS_PATH)
    views_by_career = {}
    durations = []
    for r in records:
        if r["event_type"] == "view_career" and r.get("career_id"):
            views_by_career[r["career_id"]] = views_by_career.get(r["career_id"], 0) + 1
        elif r["event_type"] == "time_on_result" and r.get("duration_seconds") is not None:
            durations.append(r["duration_seconds"])
    avg_duration = round(sum(durations) / len(durations), 1) if durations else None
    top_viewed = sorted(views_by_career.items(), key=lambda x: -x[1])
    return {
        "total_events": len(records),
        "views_by_career": top_viewed,
        "avg_duration_seconds": avg_duration,
        "total_time_events": len(durations),
    }


def get_career_engagement() -> dict:
    """Trả về {career_id: (like_ròng, lượt_xem)}."""
    feedback = _load(FEEDBACK_PATH)
    interactions = _load(INTERACTIONS_PATH)

    net_likes: dict = {}
    views: dict = {}
    for f in feedback:
        for cid in f.get("liked_careers", []):
            net_likes[cid] = net_likes.get(cid, 0) + 1
        for cid in f.get("disliked_careers", []):
            net_likes[cid] = net_likes.get(cid, 0) - 1
    for r in interactions:
        if r.get("event_type") == "view_career" and r.get("career_id"):
            views[r["career_id"]] = views.get(r["career_id"], 0) + 1

    return {cid: (net_likes.get(cid, 0), views.get(cid, 0))
            for cid in set(net_likes) | set(views)}