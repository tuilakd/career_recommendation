# -*- coding: utf-8 -*-
"""
Kiểm thử hồi quy cho 6 điểm không nhất quán đã được sửa. Chạy từ thư mục gốc dự án:

    python -m unittest discover -s tests -v
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

import app as app_module  # noqa: E402
from cbf_recommender import cbf_recommender  # noqa: E402
from domain_knowledge import (  # noqa: E402
    CAREER_GROUPS, SUBJECT_KEYS, SUBJECTS, TOP_N_CAREERS,
)

SCORES = {s: 7.5 for s in SUBJECTS}
SCORES.update({"Trí tuệ nhân tạo": 9.5, "Hệ quản trị CSDL": 9.0, "Xử lý ảnh": 8.5})


def form_scores(scores=SCORES):
    return {f"score_{SUBJECT_KEYS[s]}": str(v) for s, v in scores.items()}


class TestSystem(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    # ---- Điểm 1: mô tả thuật toán khớp với code ----
    def test_deployed_model_matches_docs(self):
        step = app_module.MODEL.steps[-1][1]
        label = app_module.model_label(app_module.MODEL)
        self.assertEqual(type(step).__name__, "LogisticRegression")
        self.assertEqual(label, "Logistic Regression (multinomial)")
        readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        self.assertNotIn("Random Forest (triển khai)", readme)
        self.assertNotIn("lưu Random Forest", readme)

    def test_train_marks_only_best_model(self):
        tmp = tempfile.mkdtemp()
        try:
            for f in ["train.py", "domain_knowledge.py"]:
                shutil.copy(os.path.join(ROOT, f), tmp)
            shutil.copytree(os.path.join(ROOT, "data"), os.path.join(tmp, "data"))
            out = subprocess.run([sys.executable, "train.py"], cwd=tmp, capture_output=True, text=True).stdout
            marked = [l for l in out.splitlines() if "<-- triển khai" in l]
            self.assertEqual(len(marked), 1)
            self.assertIn("Logistic Regression", marked[0])
            self.assertIn("KIỂM ĐỊNH ĐỘC LẬP", out)
        finally:
            shutil.rmtree(tmp)

    def test_retrain_keeps_algorithm_and_labels_source(self):
        tmp = tempfile.mkdtemp()
        try:
            for f in os.listdir(ROOT):
                p = os.path.join(ROOT, f)
                if f.endswith(".py") or f.endswith(".joblib"):
                    shutil.copy(p, tmp)
            shutil.copytree(os.path.join(ROOT, "data"), os.path.join(tmp, "data"))
            fake = [
                {"id": "t1", "student_name": "Test", "rating": 4, "liked_careers": ["AI01"],
                 "disliked_careers": [], "scores": {s: 6.1 + i * 0.3 for i, s in enumerate(SUBJECTS)}},
                {"id": "t2", "student_name": "Partial", "rating": 4, "liked_careers": ["AI01"],
                 "disliked_careers": [], "scores": {s: 8.0 for s in SUBJECTS[:3]}},
            ]
            with open(os.path.join(tmp, "data", "feedback.json"), "w", encoding="utf-8") as fh:
                json.dump(fake, fh, ensure_ascii=False)
            code = ("import warnings; warnings.filterwarnings('ignore');"
                    "from retrain_service import retrain_from_explicit_feedback as r; import json;"
                    "print('RES=' + json.dumps(r(), ensure_ascii=False))")
            out = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True)
            self.assertIn("RES=", out.stdout, out.stderr[-500:])
            self.assertIn("LogisticRegression", out.stdout)
            result_line = next(line for line in out.stdout.splitlines() if line.startswith("RES="))
            self.assertEqual(json.loads(result_line[4:])["samples_added"], 1)
            df = pd.read_csv(os.path.join(tmp, "data", "students.csv"))
            self.assertIn("label_source", df.columns)
            self.assertEqual(set(df["label_source"]), {"derived", "feedback"})
        finally:
            shutil.rmtree(tmp)

    # ---- Điểm 2: nhãn suy ra được đánh dấu rõ, không trộn với nhãn người dùng ----
    def test_students_csv_labels_are_marked_derived(self):
        df = pd.read_csv(os.path.join(ROOT, "data", "students.csv"))
        self.assertIn("label_source", df.columns)
        label_sources = set(df["label_source"].dropna())
        self.assertIn("derived", label_sources)
        self.assertLessEqual(label_sources, {"derived", "feedback"})

    # ---- Điểm 3: hồ sơ nghề (tri thức chuyên gia) đầy đủ & báo cáo được điểm lệch ----
    def test_profile_consistency_has_no_config_errors(self):
        from eval_system import check_profile_consistency
        problems, mismatches, total = check_profile_consistency()
        self.assertEqual(problems, [])
        self.assertEqual(total, 21)
        self.assertEqual(mismatches, [])  # 21/21 nghề nhất quán (MB01 đã được sửa hồ sơ)

    # ---- Điểm 4: kỹ năng chưa đánh giá không được cộng điểm ----
    def test_unassessed_skills_add_nothing(self):
        base = cbf_recommender.predict_career_cbf_scores(SCORES, None, "Dữ liệu & AI")
        empty = cbf_recommender.predict_career_cbf_scores(SCORES, {}, "Dữ liệu & AI")
        medium = cbf_recommender.predict_career_cbf_scores(
            SCORES, {"problem_solving": "medium", "english": "medium"}, "Dữ liệu & AI")
        self.assertEqual(base, empty)
        self.assertNotEqual(base, medium)  # "Trung bình" chọn có chủ đích vẫn có tác dụng

    def test_form_defaults_to_not_assessed(self):
        html = self.client.get("/").get_data(as_text=True)
        selects = re.findall(r'<select name="skill_level_[^>]*>(.*?)</select>', html, re.S)
        self.assertEqual(len(selects), 6)
        for sel in selects:
            m = re.search(r'<option value="" selected>Chưa đánh giá</option>', sel)
            self.assertIsNotNone(m)
            self.assertEqual(sel.count("selected"), 1)

    def test_blank_skills_do_not_change_result(self):
        data = form_scores()
        r_none = self.client.post("/predict", data=data).get_data(as_text=True)
        data_blank = dict(data, **{f"skill_level_{k}": "" for k in
                                   ["english", "teamwork", "self_learning", "communication",
                                    "problem_solving", "time_management"]})
        r_blank = self.client.post("/predict", data=data_blank).get_data(as_text=True)
        self.assertNotIn("Kỹ năng mềm đã đánh giá", r_blank)
        pct = lambda h: re.findall(r"([\d.]+)% Phù hợp", h)
        self.assertEqual(pct(r_none), pct(r_blank))
        r_medium = self.client.post("/predict", data=dict(data, skill_level_teamwork="medium")).get_data(as_text=True)
        self.assertIn("Kỹ năng mềm đã đánh giá", r_medium)

    def test_partial_manual_scores_return_provisional_recommendations(self):
        data = form_scores({subject: 8.0 for subject in SUBJECTS[:3]})
        html = self.client.post("/predict", data=data).get_data(as_text=True)
        self.assertIn("Gợi ý tham khảo ban đầu từ 3/12 môn", html)
        self.assertIn("% Phù hợp", html)

    def test_partial_grade_file_returns_provisional_recommendations(self):
        scores = {subject: 7.5 for subject in SUBJECTS[:3]}
        html = self._upload("diem.csv", self._csv_bytes(scores))
        self.assertIn("Gợi ý tham khảo ban đầu từ 3/12 môn", html)
        self.assertIn("% Phù hợp", html)

    def test_empty_scores_are_rejected(self):
        html = self.client.post("/predict", data={"input_mode": "manual"}).get_data(as_text=True)
        self.assertIn("Vui lòng nhập điểm ít nhất một môn", html)

    def test_missing_model_features_use_training_means(self):
        scores = {SUBJECTS[0]: 9.0}
        vector = app_module.build_feature_vector(scores)
        means = app_module.MODEL.named_steps["preprocess"].mean_
        self.assertEqual(vector[0], 9.0)
        self.assertEqual(list(vector[1:]), list(means[1:]))

    def test_partial_cbf_uses_only_available_subjects(self):
        partial_scores = {SUBJECTS[0]: 8.0, SUBJECTS[1]: 9.0}
        cbf_scores = cbf_recommender.predict_career_cbf_scores(partial_scores)
        self.assertEqual(set(cbf_scores), set(cbf_recommender.career_ids))
        analysis = cbf_recommender.analyze_career_subject_fit("AI01", partial_scores, {})
        analyzed_subjects = {
            item["subject"] for key in ("strengths", "improvements")
            for item in analysis[key]
        }
        self.assertLessEqual(analyzed_subjects, set(partial_scores))

    def test_career_fit_analysis_with_complete_scores(self):
        analysis = cbf_recommender.analyze_career_subject_fit("AI01", SCORES, {})
        analyzed_subjects = {
            item["subject"] for key in ("strengths", "improvements")
            for item in analysis[key]
        }
        self.assertLessEqual(analyzed_subjects, set(SUBJECTS))

    # ---- Tải file: chỉ .xlsx / .csv, không còn ảnh/OCR/điểm cứng ----
    def _upload(self, name, content, extra=None):
        data = dict(extra or {}, input_mode="file")
        data["score_file"] = (io.BytesIO(content), name)
        return self.client.post("/predict", data=data, content_type="multipart/form-data").get_data(as_text=True)

    def _csv_bytes(self, scores=SCORES):
        return pd.DataFrame([scores]).to_csv(index=False).encode("utf-8")

    def _xlsx_bytes(self, scores=SCORES):
        buf = io.BytesIO()
        pd.DataFrame([scores]).to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()

    def test_image_upload_is_not_supported(self):
        html = self._upload("bang_diem.png", b"\x89PNG\r\n\x1a\n fake")
        self.assertIn("Chỉ hỗ trợ file Excel", html)
        self.assertNotIn("% Phù hợp", html)

    def test_csv_and_xlsx_upload_work_and_match_manual_input(self):
        pct = lambda h: re.findall(r"([\d.]+)% Phù hợp", h)
        manual = pct(self.client.post("/predict", data=form_scores()).get_data(as_text=True))
        self.assertTrue(manual)
        self.assertEqual(pct(self._upload("diem.csv", self._csv_bytes())), manual)
        self.assertEqual(pct(self._upload("diem.xlsx", self._xlsx_bytes())), manual)

    def test_vertical_transcript_uses_tbchp_column(self):
        rows = [
            ("TH4306", "Công nghệ phần mềm", "9,7"),
            ("TH4320", "Trí tuệ nhân tạo", "9,4"),
            ("TH5206", "Mạng máy tính", "8"),
            ("TH5209", "Xử lý ảnh", "9,2"),
            ("TH4316", "Công nghệ Java", "7,9"),
            ("TH5208", "Phân tích thiết kế HTTT", "8,1"),
            ("TH5221", "Hệ quản trị CSDL", "7,9"),
            ("TH4315", "Ngôn ngữ C# và công nghệ .NET", "8,3"),
            ("TH5210", "An toàn và bảo mật HTTT", "9,7"),
            ("TH5231", "Kỹ thuật đồ hoạ máy tính", "10"),
            ("TH5211", "Hệ điều hành Linux", "9"),
            ("TH4309", "Công nghệ Web", "8,9"),
        ]
        df = pd.DataFrame([
            {
                "Ký hiệu": code,
                "Tên học phần": name,
                "Điểm TB": "6,1",
                "Điểm thi": "5,2",
                "TBCHP": tbchp,
            }
            for code, name, tbchp in rows
        ])
        parsed = app_module._extract_scores(df)
        self.assertEqual(len(parsed), 12)
        self.assertEqual(parsed["Công nghệ phần mềm"], 9.7)
        self.assertEqual(parsed["Trí tuệ nhân tạo"], 9.4)
        self.assertEqual(parsed["Mạng máy tính"], 8.0)

    def test_partial_vertical_transcript_ignores_ungraded_courses(self):
        df = pd.DataFrame([
            {"Ký hiệu": "TH4306", "Tên học phần": "Công nghệ phần mềm", "TBCHP": "8,5"},
            {"Ký hiệu": "TH4320", "Tên học phần": "Trí tuệ nhân tạo", "TBCHP": "Chưa học"},
            {"Ký hiệu": "TH5206", "Tên học phần": "Mạng máy tính", "TBCHP": None},
        ])
        self.assertEqual(
            app_module._extract_scores(df),
            {"Công nghệ phần mềm": 8.5},
        )

    def test_uploaded_file_is_not_overridden_by_zero_form_fields(self):
        # lỗi cũ: ô điểm ẩn = 0 khiến máy chủ bỏ qua file đã tải lên
        zeros = {f"score_{SUBJECT_KEYS[s]}": "0" for s in SUBJECTS}
        pct = lambda h: re.findall(r"([\d.]+)% Phù hợp", h)
        manual = pct(self.client.post("/predict", data=form_scores()).get_data(as_text=True))
        self.assertEqual(pct(self._upload("diem.csv", self._csv_bytes(), zeros)), manual)

    def test_file_with_out_of_range_score_is_rejected(self):
        bad = dict(SCORES, **{"Công nghệ web": 95})
        self.assertIn("khoảng 0-10", self._upload("diem.csv", self._csv_bytes(bad)))

    def test_no_image_or_hardcoded_scores_left(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('accept=".csv,.xlsx"', html)
        for bad in ["tesseract", "TRANSCRIPT_TBCHP_SCORES", "btn-scan-ocr", "image-preview", ".png", ".jpg"]:
            self.assertNotIn(bad.lower(), html.lower(), bad)
        src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
        self.assertNotIn("9.7", src)
        self.assertNotIn("is_image", src)

    def test_demo_feedback_record_removed(self):
        fb = json.load(open(os.path.join(ROOT, "data", "feedback.json"), encoding="utf-8"))
        for r in fb:
            sc = r.get("scores", {})
            self.assertFalse(sc.get("Công nghệ phần mềm") == 9.7 and sc.get("Trí tuệ nhân tạo") == 9.4
                             and sc.get("Kỹ thuật đồ hoạ máy tính") == 10.0, "bản ghi demo trùng bộ điểm cứng cũ")

    # ---- CBF theo sơ đồ: vector nghề thang 5-10 ----
    def test_career_profile_vectors_follow_diagram(self):
        from domain_knowledge import CAREER_SUBJECT_WEIGHTS, career_profile_vector
        for cid, w in CAREER_SUBJECT_WEIGHTS.items():
            v = career_profile_vector(cid)
            self.assertEqual(len(v), len(SUBJECTS))
            self.assertTrue(all(5.0 <= x <= 10.0 for x in v))
            for i, s in enumerate(SUBJECTS):
                if w.get(s, 0) == 0:
                    self.assertEqual(v[i], 5.0)      # môn không liên quan = 5
                if w.get(s, 0) == 3:
                    self.assertEqual(v[i], 10.0)     # môn cốt lõi = 10
        mb = career_profile_vector("MB01")
        self.assertEqual(mb[SUBJECTS.index("Công nghệ web")], 5.0)   # MB01 không còn nghiêng về Web

    def test_profiles_are_distinct_within_group(self):
        from domain_knowledge import career_profile_vector
        import json as _j
        careers = _j.load(open(os.path.join(ROOT, "data", "careers.json"), encoding="utf-8"))
        vecs = [tuple(career_profile_vector(c)) for c in careers]
        self.assertEqual(len(set(vecs)), len(vecs), "không có 2 nghề trùng vector (sẽ bằng điểm)")

    # ---- Điểm 6: luôn Top 3 nghề ----
    def test_recommend_top_careers_returns_exactly_three(self):
        for top_group in CAREER_GROUPS:
            others = [(g, 0.1) for g in CAREER_GROUPS if g != top_group]
            ranked = [(top_group, 0.5)] + others
            top, remaining = cbf_recommender.recommend_top_careers(ranked, SCORES, {})
            ids = [c["career_id"] for c in top]
            self.assertEqual(len(top), TOP_N_CAREERS, top_group)
            self.assertEqual(len(set(ids)), len(ids), "không được trùng nghề")
            self.assertEqual([c["rank"] for c in top], [1, 2, 3])
            self.assertFalse(set(ids) & {c["career_id"] for c in remaining})
            n_in_group = sum(1 for c in top if not c["is_group_fill"])
            self.assertEqual(n_in_group, min(3, len(cbf_recommender.rank_careers_in_group(top_group, SCORES))))

    def test_result_page_shows_top3_without_algorithm_terms(self):
        html = self.client.post("/predict", data=form_scores()).get_data(as_text=True)
        self.assertRegex(html, r"3 nghề phù hợp nhất")
        self.assertEqual(len(re.findall(r'<details class="career-card" data-career-id=', html)), 3)
        for technical_term in ["Logistic Regression", "Content-Based Filtering", "CBF", "thuật toán", "xác suất nhóm"]:
            self.assertNotIn(technical_term, html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
