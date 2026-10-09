# -*- coding: utf-8 -*-
"""
eval_system.py — Đánh giá hệ thống trên dữ liệu mới (DATASET.xlsx).

THAY ĐỔI SO VỚI BẢN CŨ: dữ liệu cũ có "future_career" (nghề cụ thể) tự khai
của từng sinh viên nên có thể đánh giá CBF ở mức "nghề cụ thể" (Tầng 2) so
với nhãn thật. Dữ liệu MỚI chỉ có bảng điểm, không có nghề cụ thể thật của
từng người — career_group đã là nhãn SUY RA (xem prepare_data.py), và hồ sơ
từng nghề cụ thể (CAREER_SUBJECT_WEIGHTS) cũng là tri thức chuyên môn gán
sẵn chứ không học từ dữ liệu. Do đó phần đánh giá dưới đây gồm:

  - Tầng 1 (Logistic Regression / Random Forest -> Career Group): độ khớp với
    nhãn SUY RA bằng 5-fold CV, so với baseline "luôn đoán nhóm phổ biến nhất".
    Đây KHÔNG phải độ chính xác dự đoán nghề thực tế (nhãn là hàm của chính điểm).
  - Kiểm định ĐỘC LẬP trên nhóm nghề do người dùng xác nhận (feedback), khi đủ mẫu.
  - Thống kê số lượng nhãn mỗi nhóm nghề (để nhận biết mất cân bằng dữ liệu).
  - Kiểm tra tính nhất quán giữa hồ sơ từng nghề (tri thức chuyên gia) và nhóm nghề.
  - Tổng hợp feedback thật của người dùng nếu đã có.

Tầng 2 (CBF chọn đúng nghề cụ thể trong nhóm) KHÔNG có ground-truth để so
sánh định lượng — đây là hạn chế cần nêu rõ trong báo cáo.

Chạy: python eval_system.py
"""
import sys
import warnings
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import feedback_store
from domain_knowledge import (
    CAREER_GROUPS, CAREER_SOFT_SKILLS, CAREER_SUBJECT_WEIGHTS, MODEL_LABELS,
    SUBJECTS, career_profile_vector, group_scores_z,
)
from train import (
    FEATURE_COLS, MODEL_PATH, TARGET_COL, build_models, build_preprocessor, derived_only, load_data,
    print_feedback_validation, validate_on_feedback,
)

warnings.filterwarnings("ignore")


def pct(x, n):
    return f"{x / n * 100:5.1f}%" if n else "   n/a"


def deployed_algorithm():
    """Tên thuật toán Tầng 1 đang nằm trong model_best.joblib (None nếu không đọc được)."""
    try:
        return MODEL_LABELS.get(type(joblib.load(MODEL_PATH).steps[-1][1]).__name__)
    except Exception:
        return None


def check_profile_consistency():
    """
    Kiểm tra nhất quán giữa HAI nguồn tri thức viết tay: trọng số môn -> nhóm nghề (SUBJECT_GROUP_WEIGHTS,
    dùng để gán nhãn) và hồ sơ môn học của từng nghề (vector 5-10 của CBF, sinh ra từ CAREER_SUBJECT_WEIGHTS).
    Với mỗi nghề, dựng một "sinh viên lý tưởng" có điểm từng môn đúng bằng vector nghề (môn cốt lõi = 10,
    môn không liên quan = 5), rồi xem công thức gán nhãn có xếp sinh viên đó vào đúng nhóm của nghề không.
    Trả về danh sách các nghề bị lệch.
    """
    import json
    with open("data/careers.json", encoding="utf-8") as f:
        careers = json.load(f)

    problems = []
    for cid, info in careers.items():
        if cid not in CAREER_SUBJECT_WEIGHTS:
            problems.append((cid, "thiếu hồ sơ môn học trong CAREER_SUBJECT_WEIGHTS"))
        if cid not in CAREER_SOFT_SKILLS:
            problems.append((cid, "thiếu kỹ năng mềm trong CAREER_SOFT_SKILLS"))
        if info.get("career_group") not in CAREER_GROUPS:
            problems.append((cid, f"nhóm nghề '{info.get('career_group')}' không có trong CAREER_GROUPS"))
    for cid in set(CAREER_SUBJECT_WEIGHTS) - set(careers):
        problems.append((cid, "có trong CAREER_SUBJECT_WEIGHTS nhưng không có trong careers.json"))

    mismatches = []
    for cid, info in careers.items():
        w = CAREER_SUBJECT_WEIGHTS.get(cid)
        if not w:
            continue
        scores = dict(zip(SUBJECTS, career_profile_vector(cid)))
        gz = group_scores_z(scores)
        ranked = sorted(gz, key=gz.get, reverse=True)
        if ranked[0] != info["career_group"]:
            mismatches.append((cid, info.get("career_name", cid), info["career_group"], ranked[0],
                               ranked.index(info["career_group"]) + 1))
    return problems, mismatches, len(careers)


def evaluate():
    df = derived_only(load_data()).dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    X, y = df[FEATURE_COLS], df[TARGET_COL]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n = 0
    lr_top1 = lr_top3 = rf_top1 = rf_top3 = base_top1 = base_top3 = 0

    for train_idx, test_idx in skf.split(X, y):
        lr_pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", build_models()["Logistic Regression (multinomial)"]),
        ])
        lr_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        lr_proba = lr_pipe.predict_proba(X.iloc[test_idx])

        rf_pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", build_models()["Random Forest"]),
        ])
        rf_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        rf_proba = rf_pipe.predict_proba(X.iloc[test_idx])

        group_freq = [g for g, _ in Counter(y.iloc[train_idx]).most_common()]

        for lr_p, rf_p, true_group in zip(lr_proba, rf_proba, y.iloc[test_idx]):
            lr_ranked = [g for g, _ in sorted(zip(lr_pipe.classes_, lr_p), key=lambda x: -x[1])]
            rf_ranked = [g for g, _ in sorted(zip(rf_pipe.classes_, rf_p), key=lambda x: -x[1])]
            n += 1
            lr_top1 += lr_ranked[0] == true_group
            lr_top3 += true_group in lr_ranked[:3]
            rf_top1 += rf_ranked[0] == true_group
            rf_top3 += true_group in rf_ranked[:3]
            base_top1 += group_freq[0] == true_group
            base_top3 += true_group in group_freq[:3]

    line = "=" * 66
    deployed = deployed_algorithm()
    role = lambda nm: "(triển khai)" if nm == deployed else "(đối chứng)"
    lr_name, rf_name = "Logistic Regression (multinomial)", "Random Forest"
    print(line); print("TẦNG 1 — NHÓM NGHỀ: ĐỘ KHỚP VỚI NHÃN SUY RA (5-fold CV)"); print(line)
    print("Lưu ý: nhãn được suy ra từ chính điểm 12 môn nên đây KHÔNG phải độ chính xác dự đoán nghề thực tế.")
    print(f"Số sinh viên (nhãn suy ra): {n}")
    print(f"{'':44}{'Top-1':>10}{'Top-3':>10}")
    print(f"{lr_name + ' ' + role(lr_name):<44}{pct(lr_top1, n):>10}{pct(lr_top3, n):>10}")
    print(f"{rf_name + ' ' + role(rf_name):<44}{pct(rf_top1, n):>10}{pct(rf_top3, n):>10}")
    print(f"{'Baseline: luôn đoán nhóm phổ biến':<44}{pct(base_top1, n):>10}{pct(base_top3, n):>10}")

    print(); print(line); print("SỐ MẪU MỖI NHÓM NGHỀ (nhãn được suy ra từ bảng điểm)"); print(line)
    counts = Counter(y)
    for group, k in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {group:<28}{k:>4}  ({k / len(y) * 100:.1f}%)")

    print(); print(line); print("TẦNG 2 — XẾP HẠNG NGHỀ BẰNG CONTENT-BASED FILTERING (CBF)"); print(line)
    import time
    import numpy as np
    from cbf_recommender import cbf_recommender

    print(f"Hồ sơ đặc trưng nghề (Item Profiles): {len(cbf_recommender.career_ids)} nghề nghiệp với trọng số 12 môn học")
    print("Thuật toán: Content-Based Filtering (Cosine Similarity & Core Subject Mastery) kết hợp kỹ năng mềm.")
    print("Quy tắc: Xếp hạng nghề cụ thể trong nhóm ngành từ cao xuống thấp, khuyến nghị TOP 3.")
    print("Lưu ý: hồ sơ từng nghề là TRI THỨC CHUYÊN GIA viết tay, không học từ dữ liệu; các chỉ số dưới đây")
    print("       chỉ mô tả hành vi của thuật toán, không phải độ chính xác so với nghề thực tế.")

    top1_gap = []
    full_gap = []
    distinct_top1 = 0
    top1_by_group = {}
    consistent_count = 0
    times = []

    for idx, row in df.iterrows():
        scores = {s: row[FEATURE_COLS[SUBJECTS.index(s)]] for s in SUBJECTS}
        grp = row[TARGET_COL]

        t0 = time.perf_counter()
        ranked = cbf_recommender.rank_careers_in_group(grp, scores)
        times.append(time.perf_counter() - t0)

        if len(ranked) >= 2:
            top1_gap.append(ranked[0]["cbf_score_pct"] - ranked[1]["cbf_score_pct"])
            full_gap.append(ranked[0]["cbf_score_pct"] - ranked[-1]["cbf_score_pct"])
            if ranked[0]["cbf_score_pct"] > ranked[1]["cbf_score_pct"]:
                distinct_top1 += 1

        top1_name = ranked[0]["career_name"]
        top1_by_group.setdefault(grp, Counter())[top1_name] += 1

        if ranked[0]["strengths"]:
            consistent_count += 1

    print("\n--- CÁC CHỈ SỐ ĐỊNH LƯỢNG CỦA TẦNG 2 (CBF) ---")
    print(f"{'Chỉ số đánh giá Lọc dựa trên nội dung (CBF)':<44}{'Kết quả':>20}")
    print("-" * 66)
    print(f"{'Độ phân tách điểm số TB (#1 so với #2)':<44}{'+' + str(round(np.mean(top1_gap), 2)) + '%':>20}")
    print(f"{'Khoảng cách điểm số TB (#1 so với cuối nhóm)':<44}{'+' + str(round(np.mean(full_gap), 2)) + '%':>20}")
    print(f"{'Tỉ lệ phân định vị trí #1 dứt khoát':<44}{pct(distinct_top1, len(df)):>20}")
    all_top1_careers = set(c for grp_c in top1_by_group.values() for c in grp_c)
    print(f"{'Độ phủ danh mục nghề nghiệp (Catalog Coverage)':<44}{f'{len(all_top1_careers)}/21 nghề ({pct(len(all_top1_careers), 21).strip()})':>20}")
    print(f"{'Độ tương thích thế mạnh môn học':<44}{pct(consistent_count, len(df)):>20}")
    print(f"{'Thời gian xếp hạng CBF trung bình':<44}{f'{np.mean(times)*1000:.2f} ms/sv':>20}")
    print("-" * 66)

    print("\n--- PHÂN BỐ NGHỀ ĐƯỢC XẾP HẠNG #1 TRONG TỪNG NHÓM ---")
    for grp, c_counts in sorted(top1_by_group.items()):
        total_grp = sum(c_counts.values())
        print(f"Nhóm: {grp} ({total_grp} sinh viên)")
        for cname, cnt in c_counts.most_common():
            print(f"  • {cname:<36}: {cnt:>3} ({cnt / total_grp * 100:.1f}%)")

    print(); print(line); print("FEEDBACK THẬT CỦA NGƯỜI DÙNG (data/feedback.json)"); print(line)
    fb = feedback_store.get_all_feedback()
    if not fb:
        print("Chưa có đánh giá nào. Khi có, mục này sẽ hiện điểm hài lòng trung bình và tỉ lệ nghề được like/dislike.")
    else:
        ratings = [f["rating"] for f in fb if f.get("rating")]
        likes = sum(len(f.get("liked_careers", [])) for f in fb)
        dislikes = sum(len(f.get("disliked_careers", [])) for f in fb)
        print(f"Số lượt đánh giá        : {len(fb)}")
        print(f"Điểm hài lòng trung bình: {sum(ratings) / len(ratings):.2f}/5" if ratings else "Chưa có điểm sao")
        print(f"Nghề được like / dislike : {likes} / {dislikes}"
              + (f"  (tỉ lệ like {likes / (likes + dislikes) * 100:.1f}%)" if likes + dislikes else ""))

    # --- Kiểm định độc lập: mô hình chỉ học từ nhãn suy ra, thử trên nhóm nghề người dùng xác nhận ---
    val_pipe = Pipeline([("preprocess", build_preprocessor()), ("model", build_models()["Logistic Regression (multinomial)"])])
    val_pipe.fit(df[FEATURE_COLS], df[TARGET_COL])
    print_feedback_validation(validate_on_feedback(val_pipe))

    print(); print(line); print("NHẤT QUÁN GIỮA HỒ SƠ NGHỀ (TRI THỨC CHUYÊN GIA) VÀ NHÓM NGHỀ"); print(line)
    problems, mismatches, total = check_profile_consistency()
    for cid, msg in problems:
        print(f"  [LỖI CẤU HÌNH] {cid}: {msg}")
    if not problems:
        print("  Cấu hình đầy đủ: mọi nghề trong careers.json đều có hồ sơ môn học và kỹ năng mềm.")
    print(f"  Nghề có hồ sơ khớp đúng nhóm: {total - len(mismatches)}/{total}")
    for cid, cname, grp, got, rank in mismatches:
        print(f"  [LỆCH] {cid} {cname}: thuộc nhóm '{grp}' nhưng hồ sơ môn học lại nghiêng về '{got}' "
              f"(nhóm đúng đứng hạng {rank}). Nên rà soát trọng số của nghề này.")


if __name__ == "__main__":
    evaluate()
