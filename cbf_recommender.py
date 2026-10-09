# -*- coding: utf-8 -*-
"""
cbf_recommender.py — Module gợi ý nghề nghiệp sử dụng Content-Based Filtering (CBF - Lọc dựa trên nội dung).

QUY TRÌNH HOẠT ĐỘNG:
  1. Xây dựng hồ sơ đặc trưng nghề nghiệp (Item Profiles) — theo sơ đồ CBF:
     Mỗi nghề là 1 vector 12 chiều (cùng số chiều với vector sinh viên): MÔN CỐT LÕI = 10, MÔN KHÔNG
     LIÊN QUAN = 5, môn liên quan một phần nằm ở giữa (domain_knowledge.career_profile_vector).
     Kỹ năng mềm then chốt lấy từ CAREER_SOFT_SKILLS.
  2. Xây dựng hồ sơ sinh viên (User Profile):
     Vector điểm 12 môn học (vector A). Mức độ thành thạo các kỹ năng mềm được số hoá theo cấp độ
     (Yếu / Trung bình / Tốt / Xuất sắc).
  3. Đo lường độ tương đồng và tính điểm phù hợp (Content-Based Matching):
     - Cosine Similarity giữa vector sinh viên A và vector nghề B: cos = A·B / (||A||·||B||). Vì điểm
       sinh viên đều cao (7-9) nên Cosine trên điểm thô gần như bằng nhau giữa các nghề (khoảng cách
       #1-#2 chỉ ~0.003); do đó A được chuẩn hoá z-score theo từng môn và B được tâm hoá (trừ trung bình
       các môn) trước khi tính Cosine — giá trị 10/5 của nghề vẫn giữ nguyên như sơ đồ.
     - Điểm thuần thục môn học trọng tâm (Core Subject Mastery Score) = điểm TB có trọng số các môn của nghề x 10.
     - Điểm cộng/trừ từ mức độ đáp ứng các kỹ năng mềm cốt lõi của nghề.
     - Điểm ưu tiên nhóm ngành (Career Group Alignment).
     - Hiệu chỉnh theo phản hồi tích cực từ cộng đồng (Explicit Feedback).
  4. Xếp hạng: Sắp xếp toàn bộ nghề cụ thể trong nhóm ngành từ cao xuống thấp theo % phù hợp CBF,
     rồi lấy TOP_N_CAREERS (=3) nghề đầu làm khuyến nghị (recommend_top_careers).
  5. Trợ giúp lựa chọn: Phân tích môn học thế mạnh, môn cần cải thiện và đối sánh kỹ năng mềm.

LƯU Ý VỀ BẢN CHẤT CỦA ITEM PROFILES: môn cốt lõi của từng nghề (suy ra từ CAREER_SUBJECT_WEIGHTS) và CAREER_SOFT_SKILLS là TRI THỨC
CHUYÊN GIA do người xây dựng hệ thống định nghĩa (dựa trên mô tả nghề), KHÔNG được học từ dữ liệu
(dữ liệu chỉ có bảng điểm, không có nghề thực tế của từng sinh viên). Vì vậy Tầng 2 là hệ khuyến
nghị dựa trên tri thức, chưa thể đánh giá định lượng bằng ground-truth. Tính nhất quán giữa các hồ
sơ nghề và nhóm nghề được kiểm tra bởi eval_system.check_profile_consistency().
"""
import json
import os
import numpy as np
import pandas as pd

from domain_knowledge import (
    CAREER_SOFT_SKILLS, CAREER_SUBJECT_WEIGHTS, career_profile_vector,
    KEY_TO_SUBJECT, LABEL_FEEDBACK, LABEL_SOURCE_COL, SKILL_LEVEL_CHOICES, SKILL_LEVEL_LABELS,
    SOFT_SKILL_GROUP_BOOST, SOFT_SKILL_LABELS, SUBJECT_KEYS, SUBJECTS, TOP_N_CAREERS,
)

# Điểm cộng CBF cho từng cấp độ kỹ năng mềm
SKILL_LEVEL_CBF_POINTS = {c["key"]: c.get("cbf_bonus", c.get("cf_bonus", 2.0)) for c in SKILL_LEVEL_CHOICES}

FEATURE_COLS = [SUBJECT_KEYS[s] for s in SUBJECTS]


class ContentBasedFilteringRecommender:
    def __init__(self, students_csv_path="data/students.csv",
                 careers_json_path="data/careers.json",
                 feedback_json_path="data/feedback.json"):
        self.students_csv_path = students_csv_path
        self.careers_json_path = careers_json_path
        self.feedback_json_path = feedback_json_path
        self.reload()

    def reload(self):
        """Tải lại danh mục nghề nghiệp, tham số phân phối chuẩn của sinh viên và phản hồi tường minh."""
        # 1. Đọc danh mục 21 nghề
        with open(self.careers_json_path, encoding="utf-8") as f:
            self.careers = json.load(f)
        self.career_ids = list(self.careers.keys())
        self.cid_to_idx = {cid: i for i, cid in enumerate(self.career_ids)}

        # 2. Đọc tập sinh viên tham chiếu để lấy trung bình (mean) và độ lệch chuẩn (std)
        if os.path.exists(self.students_csv_path):
            self.df_students = pd.read_csv(self.students_csv_path)
            # Mốc chuẩn hoá z-score chỉ tính trên tập sinh viên gốc: các dòng do phản hồi người
            # dùng bổ sung (label_source == "feedback") không được làm dịch mốc tham chiếu.
            if LABEL_SOURCE_COL in self.df_students.columns:
                self.df_students = self.df_students[self.df_students[LABEL_SOURCE_COL] != LABEL_FEEDBACK]
            self.mu = self.df_students[FEATURE_COLS].mean().to_numpy().copy()
            self.sd = self.df_students[FEATURE_COLS].std().to_numpy().copy()
            self.sd[self.sd == 0] = 1.0
        else:
            self.mu = np.full(len(SUBJECTS), 7.0)
            self.sd = np.full(len(SUBJECTS), 1.0)

        # 3. Tạo Item Profiles (vector B): môn cốt lõi = 10, môn không liên quan = 5 (xem domain_knowledge)
        n_c = len(self.career_ids)
        self.career_profile_vecs = np.zeros((n_c, len(SUBJECTS)))      # vector B thô (thang 5-10)
        for cid in self.career_ids:
            idx = self.cid_to_idx[cid]
            self.career_profile_vecs[idx] = career_profile_vector(cid)
        # B tâm hoá (trừ trung bình các môn của chính nghề đó) để Cosine phản ánh "hình dạng" hồ sơ
        self.career_profile_centered = self.career_profile_vecs - self.career_profile_vecs.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(self.career_profile_centered, axis=1)
        self.career_profile_norms = np.where(norms > 0, norms, 1.0)

        # 4. Tích hợp dữ liệu phản hồi tường minh (Explicit Feedback) để tinh chỉnh thích ứng
        self.career_net_likes = {cid: 0 for cid in self.career_ids}
        if os.path.exists(self.feedback_json_path):
            try:
                with open(self.feedback_json_path, encoding="utf-8") as f:
                    feedbacks = json.load(f)
                for fb in feedbacks:
                    liked = fb.get("liked_careers", [])
                    disliked = fb.get("disliked_careers", [])
                    for cid in liked:
                        if cid in self.career_net_likes:
                            self.career_net_likes[cid] += 1
                    for cid in disliked:
                        if cid in self.career_net_likes:
                            self.career_net_likes[cid] -= 1
            except Exception:
                pass

    def _normalize_skill_input(self, skill_input) -> dict:
        """Chuẩn hoá đầu vào kỹ năng mềm thành dict {skill_key: level_key}"""
        if isinstance(skill_input, dict):
            return skill_input
        if isinstance(skill_input, (list, set)):
            return {sk: "strong" for sk in skill_input}
        return {}

    def predict_career_cbf_scores(self, student_scores: dict, skills_selected=None, target_group: str = None) -> dict:
        """
        Dự đoán điểm độ phù hợp theo Content-Based Filtering (CBF) cho toàn bộ 21 nghề.
        Kết hợp:
          - Cosine Similarity giữa vector năng lực sinh viên (z-score) và vector nghề (môn cốt lõi = 10, không liên quan = 5, tâm hoá).
          - Core Subject Mastery: điểm trung bình có trọng số các môn của nghề x 10.
          - Soft Skill Alignment: điểm cộng/trừ từ mức độ thành thạo kỹ năng mềm thực tế.
          - Career Group Alignment: ưu tiên nhẹ nếu nghề thuộc nhóm ngành mục tiêu.
          - Explicit Feedback Tuning: hiệu chỉnh dựa trên phản hồi cộng đồng.
        Trả về dict {career_id: cbf_score_pct}.
        """
        available_indices = [i for i, subject in enumerate(SUBJECTS) if subject in student_scores]
        if not available_indices:
            raise ValueError("Cần điểm ít nhất một môn để tính mức độ phù hợp nghề.")
        raw_vec = np.array([float(student_scores[SUBJECTS[i]]) for i in available_indices])
        z_query = (raw_vec - self.mu[available_indices]) / self.sd[available_indices]
        norm_query = np.linalg.norm(z_query) or 1.0
        profiles = self.career_profile_vecs[:, available_indices]
        centered_profiles = profiles - profiles.mean(axis=1, keepdims=True)
        profile_norms = np.linalg.norm(centered_profiles, axis=1)
        profile_norms = np.where(profile_norms > 0, profile_norms, 1.0)
        cosine_sims = (centered_profiles @ z_query) / (profile_norms * norm_query)

        skill_levels = self._normalize_skill_input(skills_selected)

        cbf_pct_scores = {}
        for j, cid in enumerate(self.career_ids):
            # Quy đổi Cosine [-1, 1] sang thang [0, 100]
            cos_sim = cosine_sims[j]
            cos_pct = max(0.0, min(100.0, (cos_sim + 1.0) / 2.0 * 100.0))

            # 2. Tính Core Subject Mastery: điểm trung bình có trọng số các môn của nghề x 10
            weights = CAREER_SUBJECT_WEIGHTS.get(cid, {})
            available_weights = {
                subject: weight for subject, weight in weights.items()
                if subject in student_scores
            }
            sum_w = sum(available_weights.values())
            if sum_w > 0:
                mastery_score = sum(
                    weight * float(student_scores[subject])
                    for subject, weight in available_weights.items()
                ) / sum_w * 10.0
            else:
                mastery_score = 50.0

            # Điểm CBF nền tảng: phối hợp tương đồng hướng (Cosine) và độ tinh thông môn học (Mastery)
            base_score = 0.55 * cos_pct + 0.45 * mastery_score

            # 3. Ưu tiên nếu nghề thuộc đúng nhóm ngành đang xem xét
            group_bonus = 0.0
            c_group = self.careers[cid].get("career_group", "")
            if target_group and c_group == target_group:
                group_bonus = 5.0

            # 4. Hiệu chỉnh từ Kỹ năng mềm (Soft Skills theo cấp độ Yếu / Vừa / Tốt / Xuất sắc)
            soft_bonus = 0.0
            if skill_levels:
                req_skills = CAREER_SOFT_SKILLS.get(cid, [])
                for sk_key, level in skill_levels.items():
                    cbf_pts = SKILL_LEVEL_CBF_POINTS.get(level, 2.0)
                    if sk_key in req_skills:
                        soft_bonus += cbf_pts
                    # Bổ trợ theo nhóm nghề tương ứng
                    grp_aff = SOFT_SKILL_GROUP_BOOST.get(sk_key, {}).get(c_group, 0.0)
                    soft_bonus += grp_aff * cbf_pts * 0.4

            # 5. Hiệu chỉnh từ phản hồi cộng đồng (Explicit Feedback like/dislike)
            net_like = self.career_net_likes.get(cid, 0)
            feedback_bonus = min(3.0, max(-3.0, net_like * 0.5))

            total_pct = base_score + group_bonus + soft_bonus + feedback_bonus
            total_pct = round(min(98.5, max(45.0, total_pct)), 1)
            cbf_pct_scores[cid] = total_pct

        return cbf_pct_scores

    # Giữ alias cho tương thích ngược
    predict_career_cf_scores = predict_career_cbf_scores

    def analyze_career_subject_fit(self, career_id: str, student_scores: dict, skills_selected=None) -> dict:
        """
        Trợ giúp người dùng lựa chọn: Phân tích môn học thế mạnh, môn cần cải thiện
        và các kỹ năng mềm khớp với yêu cầu thực tế của nghề.
        """
        weights = CAREER_SUBJECT_WEIGHTS.get(career_id, {})
        strengths = []
        improvements = []

        for subj, w in weights.items():
            if subj not in student_scores:
                continue
            score = float(student_scores[subj])
            mean_score = self.mu[SUBJECTS.index(subj)]
            if score >= 8.0:
                strengths.append({"subject": subj, "score": score, "weight": w, "note": "Thế mạnh vượt trội"})
            elif score >= 7.0 and score >= mean_score:
                strengths.append({"subject": subj, "score": score, "weight": w, "note": "Đạt yêu cầu tốt"})
            elif w >= 2.0 and score < 7.0:
                improvements.append({"subject": subj, "score": score, "weight": w, "note": "Môn trọng tâm cần củng cố thêm"})
            elif score < 6.0:
                improvements.append({"subject": subj, "score": score, "weight": w, "note": "Điểm cần cải thiện"})

        strengths.sort(key=lambda x: -x["weight"])
        improvements.sort(key=lambda x: -x["weight"])

        # Phân tích kỹ năng mềm theo mức độ
        skill_levels = self._normalize_skill_input(skills_selected)
        soft_skill_matches = []
        soft_skill_improvements = []
        if skill_levels:
            req_skills = CAREER_SOFT_SKILLS.get(career_id, [])
            for sk in req_skills:
                if sk not in skill_levels:
                    continue  # kỹ năng chưa được đánh giá -> không suy đoán mức độ thay người dùng
                level = skill_levels[sk]
                level_label = SKILL_LEVEL_LABELS.get(level, level)
                skill_label = SOFT_SKILL_LABELS.get(sk, sk)
                pts = SKILL_LEVEL_CBF_POINTS.get(level, 2.0)
                if level in ["strong", "excellent"]:
                    soft_skill_matches.append({
                        "key": sk,
                        "label": skill_label,
                        "level_label": level_label,
                        "bonus": pts,
                    })
                elif level == "weak":
                    soft_skill_improvements.append({
                        "key": sk,
                        "label": skill_label,
                        "level_label": level_label,
                    })

        return {
            "strengths": strengths,
            "improvements": improvements,
            "soft_skill_matches": soft_skill_matches,
            "soft_skill_improvements": soft_skill_improvements,
        }

    def rank_careers_in_group(self, group_name: str, student_scores: dict, skills_selected=None) -> list:
        """
        XẾP HẠNG: Lọc ra TOÀN BỘ các nghề cụ thể thuộc nhóm ngành `group_name`
        và SẮP XẾP TẤT CẢ CÁC NGHỀ ĐÓ TỪ CAO XUỐNG THẤP theo điểm Content-Based Filtering (CBF).
        """
        cbf_scores = self.predict_career_cbf_scores(student_scores, skills_selected, target_group=group_name)

        # Lấy tất cả nghề trong nhóm ngành được chỉ định
        group_careers = []
        for cid, info in self.careers.items():
            if info.get("career_group") == group_name:
                fit_analysis = self.analyze_career_subject_fit(cid, student_scores, skills_selected)
                career_data = {
                    "career_id": cid,
                    "career_name": info.get("career_name"),
                    "career_group": info.get("career_group"),
                    "description": info.get("mo_ta"),
                    "salary": info.get("muc_luong"),
                    "trend": info.get("xu_huong_thi_truong"),
                    "cbf_score_pct": cbf_scores[cid],
                    "cf_score_pct": cbf_scores[cid],  # Alias giữ tương thích ngược
                    "strengths": fit_analysis["strengths"],
                    "improvements": fit_analysis["improvements"],
                    "soft_skill_matches": fit_analysis["soft_skill_matches"],
                    "soft_skill_improvements": fit_analysis["soft_skill_improvements"],
                }
                group_careers.append(career_data)

        # SẮP XẾP TỪ CAO XUỐNG THẤP THEO ĐIỂM CBF
        group_careers.sort(key=lambda c: c["cbf_score_pct"], reverse=True)

        # Gắn huy hiệu thứ hạng #1, #2, #3, ... trong nhóm
        for rank, c in enumerate(group_careers, 1):
            c["rank"] = rank

        return group_careers

    def recommend_top_careers(self, ranked_groups_with_prob: list, student_scores: dict,
                              skills_selected=None, n: int = TOP_N_CAREERS):
        """
        KHUYẾN NGHỊ TOP N NGHỀ (mặc định N = 3).
        - Lấy n nghề có điểm CBF cao nhất trong nhóm nghề đứng đầu (kết quả Tầng 1).
        - Nếu nhóm đó có ít hơn n nghề (vd. "Phát triển Web" chỉ có 2), bổ sung từ nhóm nghề
          xếp hạng kế tiếp theo xác suất Tầng 1, mỗi nhóm lấy theo thứ tự điểm CBF; các nghề
          bổ sung được gắn cờ `is_group_fill = True` để giao diện ghi rõ nguồn.
        Trả về (top_careers, remaining): `remaining` là các nghề còn lại của nhóm đứng đầu.
        """
        top_group = ranked_groups_with_prob[0][0]
        in_group = self.rank_careers_in_group(top_group, student_scores, skills_selected)

        top = [dict(c, is_group_fill=False) for c in in_group[:n]]
        remaining = in_group[n:]

        for group_name, _ in ranked_groups_with_prob[1:]:
            if len(top) >= n:
                break
            for c in self.rank_careers_in_group(group_name, student_scores, skills_selected):
                if len(top) >= n:
                    break
                top.append(dict(c, is_group_fill=True))

        for i, c in enumerate(top, 1):
            c["rank"] = i
        return top, remaining

    def rank_all_groups(self, ranked_groups_with_prob: list, student_scores: dict, skills_selected: list = None) -> list:
        """
        Duyệt qua các nhóm nghề (sắp xếp theo xác suất mô hình phân loại Tầng 1),
        trong mỗi nhóm sắp xếp TOÀN BỘ nghề từ cao xuống thấp theo CBF.
        """
        all_groups_ranked = []
        for group_name, prob in ranked_groups_with_prob:
            careers_in_grp = self.rank_careers_in_group(group_name, student_scores, skills_selected)
            all_groups_ranked.append({
                "group_name": group_name,
                "group_prob": round(prob * 100, 1),
                "careers": careers_in_grp,
            })
        return all_groups_ranked


# Khởi tạo singleton instance cho ứng dụng
cbf_recommender = ContentBasedFilteringRecommender()
cf_recommender = cbf_recommender  # alias giữ tương thích
