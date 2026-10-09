# -*- coding: utf-8 -*-
"""
domain_knowledge.py — "Nguồn sự thật" duy nhất (single source of truth) cho
toàn bộ hệ thống: tên 12 môn học, 7 nhóm nghề, trọng số môn học -> nhóm nghề
(dùng để TẠO NHÃN huấn luyện vì DATASET.xlsx mới không có cột nghề tương lai
của sinh viên, chỉ có bảng điểm), trọng số môn học -> từng nghề cụ thể (dùng
cho bước CBF phân biệt các nghề trong cùng 1 nhóm) và danh sách kỹ năng/lĩnh
vực quan tâm hiển thị trên form.

Mọi module khác (prepare_data.py, train.py, app.py, eval_system.py) BẮT BUỘC
import từ đây thay vì tự định nghĩa lại, để tránh lặp lại lỗi cũ của hệ thống
(2 nơi định nghĩa nhãn nhóm nghề bị lệch nhau -> 1 nhóm luôn trả về rỗng).
"""

# ---------------------------------------------------------------------------
# 1) 12 môn học có điểm trong DATASET.xlsx (sheet "DỮ LIỆU ĐIỂM SINH VIÊN")
# ---------------------------------------------------------------------------
SUBJECTS = [
    "An toàn và bảo mật HTTT",
    "Công nghệ phần mềm",
    "Hệ điều hành Linux",
    "Hệ quản trị CSDL",
    "Kỹ thuật đồ hoạ máy tính",
    "Mạng máy tính",
    "Phân tích thiết kế HTTT",
    "Xử lý ảnh",
    "Công nghệ Java",
    "Ngôn ngữ C# và công nghệ .NET",
    "Công nghệ web",
    "Trí tuệ nhân tạo",
]

# Nhãn ngắn gọn dùng làm tên cột file CSV / khoá JSON (không dấu, không khoảng
# trắng) — tránh lỗi khi ghi/đọc CSV với các ký tự đặc biệt.
SUBJECT_KEYS = {
    "An toàn và bảo mật HTTT": "sec",
    "Công nghệ phần mềm": "se",
    "Hệ điều hành Linux": "linux",
    "Hệ quản trị CSDL": "db",
    "Kỹ thuật đồ hoạ máy tính": "graphics",
    "Mạng máy tính": "network",
    "Phân tích thiết kế HTTT": "analysis",
    "Xử lý ảnh": "image",
    "Công nghệ Java": "java",
    "Ngôn ngữ C# và công nghệ .NET": "csharp",
    "Công nghệ web": "web",
    "Trí tuệ nhân tạo": "ai",
}
KEY_TO_SUBJECT = {v: k for k, v in SUBJECT_KEYS.items()}

# ---------------------------------------------------------------------------
# 2) 7 nhóm nghề (career_group) xuất hiện trong sheet "ĐỊNH HƯỚNG NGHỀ NGHIỆP"
# ---------------------------------------------------------------------------
CAREER_GROUPS = [
    "Phát triển phần mềm",
    "Phát triển Web",
    "Đồ họa & Game",
    "Dữ liệu & AI",
    "Hạ tầng & Bảo mật",
    "Kiểm thử & Quản lý dự án",
    "An toàn thông tin",
]

# ---------------------------------------------------------------------------
# 3) Trọng số MÔN HỌC -> NHÓM NGHỀ
#
# DATASET.xlsx mới CHỈ có bảng điểm học phần, KHÔNG có cột "nghề mong muốn"
# như dữ liệu cũ. Vì vậy bước "tiền xử lý dữ liệu" (prepare_data.py) phải suy
# ra nhãn career_group cho từng sinh viên bằng cách quy đổi bảng điểm sang
# "điểm phù hợp" với từng nhóm nghề theo ma trận trọng số dưới đây (kiến thức
# chuyên môn: môn nào liên quan tới nhóm nghề nào), sau đó gán nhãn = nhóm có
# điểm phù hợp cao nhất (argmax). Đây là nhãn thay thế hợp lý duy nhất có thể
# suy ra được từ dữ liệu hiện có, đóng vai trò nhãn huấn luyện cho 3 thuật
# toán phân loại (Logistic Regression / Random Forest / Gradient Boosting) — đúng
# quy trình cũ, chỉ khác nguồn nhãn.
#
# Trọng số trong khoảng (0, 1], nhiều môn có thể đóng góp cho nhiều nhóm với
# tỉ trọng khác nhau (vd. "Hệ quản trị CSDL" liên quan cả Phát triển phần mềm
# lẫn Dữ liệu & AI).
#
# LƯU Ý QUAN TRỌNG (giới hạn của nhãn giả / pseudo-label): vì nhãn là một hàm
# XÁC ĐỊNH của chính 12 điểm dùng làm đặc trưng, các chỉ số Accuracy/F1 của bộ
# phân loại chỉ cho biết mô hình xấp xỉ lại công thức này tốt đến đâu — KHÔNG
# phải khả năng dự đoán nghề nghiệp thực tế của sinh viên. Chỉ số độc lập duy
# nhất là độ khớp với nhóm nghề do NGƯỜI DÙNG xác nhận (feedback), xem
# train.validate_on_feedback().
# ---------------------------------------------------------------------------
SUBJECT_GROUP_WEIGHTS = {
    "An toàn và bảo mật HTTT": {"An toàn thông tin": 1.0},
    "Công nghệ phần mềm": {"Phát triển phần mềm": 0.6, "Kiểm thử & Quản lý dự án": 0.4},
    "Hệ điều hành Linux": {"Hạ tầng & Bảo mật": 0.8, "An toàn thông tin": 0.2},
    "Hệ quản trị CSDL": {"Phát triển phần mềm": 0.5, "Dữ liệu & AI": 0.5},
    "Kỹ thuật đồ hoạ máy tính": {"Đồ họa & Game": 1.0},
    "Mạng máy tính": {"Hạ tầng & Bảo mật": 0.7, "An toàn thông tin": 0.3},
    "Phân tích thiết kế HTTT": {"Kiểm thử & Quản lý dự án": 1.0},
    "Xử lý ảnh": {"Đồ họa & Game": 0.5, "Dữ liệu & AI": 0.5},
    "Công nghệ Java": {"Phát triển phần mềm": 0.6, "Phát triển Web": 0.4},
    "Ngôn ngữ C# và công nghệ .NET": {"Phát triển phần mềm": 0.7, "Phát triển Web": 0.3},
    "Công nghệ web": {"Phát triển Web": 1.0},
    "Trí tuệ nhân tạo": {"Dữ liệu & AI": 1.0},
}


def group_scores(subject_score: dict) -> dict:
    """Quy đổi {tên môn: điểm 0-10} -> {nhóm nghề: điểm phù hợp 0-10} bằng
    trung bình có trọng số theo SUBJECT_GROUP_WEIGHTS. Đây là điểm THÔ (thang
    0-10), dùng để HIỂN THỊ cho sinh viên ("điểm phù hợp theo nhóm ngành")
    chứ KHÔNG dùng để gán nhãn huấn luyện (xem dominant_group bên dưới) vì
    các môn có độ khó khác nhau (vd. môn Đồ hoạ máy tính có điểm trung bình
    toàn trường cao hơn hẳn các môn khác), nên so sánh điểm thô giữa các
    nhóm sẽ thiên vị nhóm có môn "dễ được điểm cao"."""
    totals = {g: 0.0 for g in CAREER_GROUPS}
    weights = {g: 0.0 for g in CAREER_GROUPS}
    for subject, score in subject_score.items():
        for group, w in SUBJECT_GROUP_WEIGHTS.get(subject, {}).items():
            totals[group] += float(score) * w
            weights[group] += w
    return {g: round(totals[g] / weights[g], 2) if weights[g] else 0.0 for g in CAREER_GROUPS}


# Trung bình & độ lệch chuẩn của từng môn học, tính một lần trên toàn bộ 354
# sinh viên hợp lệ trong DATASET.xlsx (data/students.csv). Dùng làm mốc chuẩn
# hoá (z-score) CỐ ĐỊNH để mọi lần gán nhãn / suy luận sau này đều so sánh
# một sinh viên với CÙNG một mặt bằng tham chiếu, tránh nhãn bị lệch nhóm vì
# môn dễ/khó khác nhau (xem ghi chú ở group_scores()).
SUBJECT_STATS = {
    "An toàn và bảo mật HTTT": (7.603, 1.176),
    "Công nghệ phần mềm": (7.405, 1.226),
    "Hệ điều hành Linux": (8.038, 1.601),
    "Hệ quản trị CSDL": (7.604, 1.206),
    "Kỹ thuật đồ hoạ máy tính": (8.332, 1.478),
    "Mạng máy tính": (7.258, 1.062),
    "Phân tích thiết kế HTTT": (7.439, 1.045),
    "Xử lý ảnh": (7.303, 1.417),
    "Công nghệ Java": (7.567, 1.172),
    "Ngôn ngữ C# và công nghệ .NET": (7.620, 1.379),
    "Công nghệ web": (7.658, 1.432),
    "Trí tuệ nhân tạo": (7.671, 1.292),
}


def group_scores_z(subject_score: dict) -> dict:
    """Giống group_scores() nhưng chuẩn hoá từng môn về z-score (so với
    SUBJECT_STATS) trước khi gộp theo trọng số — công bằng giữa các môn có
    độ khó khác nhau. Dùng để GÁN NHÃN career_group khi huấn luyện
    (prepare_data.py) và để XẾP HẠNG nhóm nghề một cách nhất quán."""
    totals = {g: 0.0 for g in CAREER_GROUPS}
    weights = {g: 0.0 for g in CAREER_GROUPS}
    for subject, score in subject_score.items():
        mean, std = SUBJECT_STATS.get(subject, (float(score), 1.0))
        z = (float(score) - mean) / (std or 1.0)
        for group, w in SUBJECT_GROUP_WEIGHTS.get(subject, {}).items():
            totals[group] += z * w
            weights[group] += w
    return {g: round(totals[g] / weights[g], 3) if weights[g] else 0.0 for g in CAREER_GROUPS}


def dominant_group(subject_score: dict) -> str:
    """Nhóm nghề phù hợp nhất — dùng làm nhãn career_group, dựa trên điểm đã
    chuẩn hoá (group_scores_z) để không thiên vị nhóm có môn dễ đạt điểm cao."""
    scores = group_scores_z(subject_score)
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# 4) Trọng số MÔN HỌC -> TỪNG NGHỀ CỤ THỂ (21 nghề)
#
# careers.json chỉ gắn mỗi nghề với 1 NHÓM nghề (career_group), nhưng để bước
# CBF (Cosine Similarity) phân biệt được các nghề CÙNG một nhóm (vd. nhóm
# "Phát triển phần mềm" có 3 nghề: Backend, Mobile, Kỹ sư phần mềm), hệ thống
# cần một "hồ sơ điển hình" (typical profile) riêng cho từng nghề. Vì dữ liệu
# sinh viên trong DATASET.xlsx không gắn với nghề cụ thể (chỉ có điểm), hồ sơ
# này được xây dựng từ kiến thức chuyên môn (đọc mô tả nghề) thay vì học trực
# tiếp từ dữ liệu — tương tự cách careers.json cũ mô tả "ky_nang" bằng văn
# bản. Thang trọng số 0-3 (0 = không liên quan, 3 = cốt lõi).
# ---------------------------------------------------------------------------
CAREER_SUBJECT_WEIGHTS = {
    "BE01": {"Công nghệ Java": 2.5, "Ngôn ngữ C# và công nghệ .NET": 2.5, "Hệ quản trị CSDL": 3,
             "Công nghệ phần mềm": 2, "Mạng máy tính": 1},
    "FE01": {"Công nghệ web": 3, "Kỹ thuật đồ hoạ máy tính": 1, "Công nghệ phần mềm": 1},
    "FS01": {"Công nghệ web": 3, "Công nghệ Java": 1.5, "Ngôn ngữ C# và công nghệ .NET": 1.5,
             "Hệ quản trị CSDL": 2, "Công nghệ phần mềm": 1.5},
    "UD01": {"Kỹ thuật đồ hoạ máy tính": 3, "Xử lý ảnh": 2, "Công nghệ web": 1.5},
    # MB01 (di động): Java (Android) + C#/.NET (đa nền tảng) + Công nghệ phần mềm là môn cốt lõi. Bản trước có
    # "Công nghệ web" = 1.5 khiến hồ sơ nghiêng về nhóm Phát triển Web hơn nhóm được gán (Phát triển phần mềm).
    "MB01": {"Công nghệ Java": 2.5, "Ngôn ngữ C# và công nghệ .NET": 2, "Công nghệ phần mềm": 2, "Hệ quản trị CSDL": 1},
    "DA01": {"Hệ quản trị CSDL": 3, "Trí tuệ nhân tạo": 2, "Xử lý ảnh": 1, "Phân tích thiết kế HTTT": 1.5},
    "AI01": {"Trí tuệ nhân tạo": 3, "Hệ quản trị CSDL": 1.5, "Xử lý ảnh": 1.5, "Công nghệ phần mềm": 1},
    "DO01": {"Hệ điều hành Linux": 3, "Mạng máy tính": 2.5, "An toàn và bảo mật HTTT": 1.5},
    "QA01": {"Công nghệ phần mềm": 2.5, "Phân tích thiết kế HTTT": 2},
    "BA01": {"Phân tích thiết kế HTTT": 3, "Công nghệ phần mềm": 1},
    "SA01": {"Hệ điều hành Linux": 3, "Mạng máy tính": 2, "An toàn và bảo mật HTTT": 1},
    "DG01": {"Kỹ thuật đồ hoạ máy tính": 3, "Công nghệ Java": 1.5, "Ngôn ngữ C# và công nghệ .NET": 1.5,
             "Xử lý ảnh": 1.5},
    "SE01": {"Công nghệ phần mềm": 3, "Công nghệ Java": 2, "Ngôn ngữ C# và công nghệ .NET": 2,
             "Hệ quản trị CSDL": 1.5},
    "SY01": {"Phân tích thiết kế HTTT": 3, "Hệ quản trị CSDL": 1.5, "Công nghệ phần mềm": 1},
    "IT01": {"Phân tích thiết kế HTTT": 2.5, "Mạng máy tính": 1, "Công nghệ phần mềm": 1.5},
    "CV01": {"Xử lý ảnh": 3, "Trí tuệ nhân tạo": 3, "Kỹ thuật đồ hoạ máy tính": 1},
    "NW01": {"Mạng máy tính": 3, "Hệ điều hành Linux": 2, "An toàn và bảo mật HTTT": 1},
    "SC01": {"An toàn và bảo mật HTTT": 3, "Mạng máy tính": 2, "Hệ điều hành Linux": 1},
    "PT01": {"An toàn và bảo mật HTTT": 3, "Mạng máy tính": 2, "Hệ điều hành Linux": 2},
    "SG01": {"An toàn và bảo mật HTTT": 3, "Mạng máy tính": 1.5, "Hệ điều hành Linux": 1.5},
    "GR01": {"Kỹ thuật đồ hoạ máy tính": 3, "Xử lý ảnh": 2},
}

# ---------------------------------------------------------------------------
# 4b) VECTOR ĐẶC TRƯNG CỦA NGHỀ cho Content-Based Filtering (theo sơ đồ CBF)
#
# Mỗi nghề là 1 vector 12 chiều (cùng số chiều với vector điểm sinh viên): MÔN CỐT LÕI = 10,
# MÔN KHÔNG LIÊN QUAN = 5. Ví dụ Kỹ sư dữ liệu: (CSDL, Trí tuệ nhân tạo cao = 10, các môn khác = 5).
# Trọng số chuyên gia 0-3 ở trên được quy đổi TUYẾN TÍNH sang thang 5-10:
#       giá trị = 5 + 5 * (trọng số / 3)      (trọng số 3 -> 10, trọng số 0 -> 5)
# Không dùng nhị phân thuần tuý (chỉ 10 hoặc 5) vì khi đó nhiều nghề cùng nhóm có vector giống hệt nhau
# (vd. DA01 và AI01 đều là "CSDL + AI"; DO01, SA01, NW01 đều là "Linux + Mạng") -> điểm bằng nhau, không xếp hạng được.
# ---------------------------------------------------------------------------
PROFILE_MAX_SCORE = 10.0   # môn cốt lõi (trọng số = WEIGHT_SCALE_MAX)
PROFILE_BASE_SCORE = 5.0   # môn không liên quan (trọng số = 0)
WEIGHT_SCALE_MAX = 3.0


def career_profile_vector(career_id: str) -> list:
    """Vector 12 chiều (đúng thứ tự SUBJECTS) của nghề: môn cốt lõi = 10, môn không liên quan = 5."""
    weights = CAREER_SUBJECT_WEIGHTS.get(career_id, {})
    span = PROFILE_MAX_SCORE - PROFILE_BASE_SCORE
    return [PROFILE_BASE_SCORE + span * min(weights.get(s, 0.0), WEIGHT_SCALE_MAX) / WEIGHT_SCALE_MAX
            for s in SUBJECTS]


# ---------------------------------------------------------------------------
# 5) Danh sách KỸ NĂNG MỀM cho form nhập liệu (checkbox)
# Sinh viên chọn các kỹ năng mềm mà mình có thế mạnh (làm việc nhóm, tiếng Anh,
# tự học, giao tiếp,...). Hệ thống sẽ dùng các kỹ năng này để cộng điểm ưu tiên
# cho các nhóm nghề và nghề nghiệp đòi hỏi kỹ năng mềm tương ứng.
# ---------------------------------------------------------------------------
SOFT_SKILL_OPTIONS = [
    {"key": "english", "label": "Đọc hiểu tiếng Anh"},
    {"key": "teamwork", "label": "Làm việc nhóm"},
    {"key": "self_learning", "label": "Tự học"},
    {"key": "communication", "label": "Giao tiếp"},
    {"key": "problem_solving", "label": "Tư duy logic"},
    {"key": "time_management", "label": "Quản lý thời gian"},
]

SKILL_LEVEL_CHOICES = [
    {"key": "weak", "label": "Yếu", "multiplier": -0.05, "cbf_bonus": -2.5, "cf_bonus": -2.5},
    {"key": "medium", "label": "Trung bình", "multiplier": 0.08, "cbf_bonus": 2.5, "cf_bonus": 2.5},
    {"key": "strong", "label": "Tốt", "multiplier": 0.18, "cbf_bonus": 5.5, "cf_bonus": 5.5},
    {"key": "excellent", "label": "Xuất sắc", "multiplier": 0.30, "cbf_bonus": 9.0, "cf_bonus": 9.0},
]

SKILL_LEVEL_CBF_POINTS = {c["key"]: c["cbf_bonus"] for c in SKILL_LEVEL_CHOICES}
SKILL_LEVEL_CF_POINTS = SKILL_LEVEL_CBF_POINTS  # alias tương thích ngược
SKILL_LEVEL_LABELS = {c["key"]: c["label"] for c in SKILL_LEVEL_CHOICES}
SKILL_LEVEL_MULTIPLIERS = {c["key"]: c["multiplier"] for c in SKILL_LEVEL_CHOICES}

DEFAULT_SKILL_LEVEL = "medium"

SKILL_OPTIONS = SOFT_SKILL_OPTIONS  # alias giữ tương thích ngược

SOFT_SKILL_GROUP_BOOST = {
    "teamwork": {"Kiểm thử & Quản lý dự án": 0.15, "Phát triển phần mềm": 0.10, "Phát triển Web": 0.10},
    "english": {"Dữ liệu & AI": 0.15, "Phát triển phần mềm": 0.10, "An toàn thông tin": 0.10},
    "self_learning": {"Dữ liệu & AI": 0.15, "Hạ tầng & Bảo mật": 0.15, "An toàn thông tin": 0.15, "Phát triển Web": 0.10},
    "communication": {"Kiểm thử & Quản lý dự án": 0.25, "Phát triển Web": 0.10, "Đồ họa & Game": 0.10},
    "problem_solving": {"Dữ liệu & AI": 0.15, "An toàn thông tin": 0.15, "Phát triển phần mềm": 0.15},
    "time_management": {"Kiểm thử & Quản lý dự án": 0.15, "Phát triển phần mềm": 0.10},
}

# Lĩnh vực quan tâm (sở thích) — trùng với 7 nhóm nghề để có thể dùng làm hệ
# số ưu tiên (boost) khi xếp hạng nhóm nghề dự đoán.
INTEREST_OPTIONS = CAREER_GROUPS
INTEREST_BOOST = 0.15  # tăng thêm 15% xác suất (tương đối) cho nhóm được chọn làm sở thích

# ---------------------------------------------------------------------------
# 6) Danh sách KỸ NĂNG MỀM then chốt của từng nghề cụ thể (21 nghề)
# Dùng để cộng điểm ưu tiên trực tiếp vào điểm Content-Based Filtering (CBF)
# và giải thích lý do đề xuất tại giao diện kết quả.
# ---------------------------------------------------------------------------
CAREER_SOFT_SKILLS = {
    "BE01": ["problem_solving", "teamwork", "english"],
    "FE01": ["teamwork", "communication", "self_learning"],
    "FS01": ["self_learning", "problem_solving", "time_management"],
    "UD01": ["communication", "teamwork", "self_learning"],
    "MB01": ["problem_solving", "self_learning", "teamwork"],
    "DA01": ["problem_solving", "communication", "english"],
    "AI01": ["problem_solving", "english", "self_learning"],
    "DO01": ["problem_solving", "self_learning", "time_management"],
    "QA01": ["time_management", "teamwork", "communication"],
    "BA01": ["communication", "teamwork", "time_management"],
    "SA01": ["problem_solving", "self_learning", "time_management"],
    "DG01": ["teamwork", "problem_solving", "self_learning"],
    "SE01": ["problem_solving", "teamwork", "time_management"],
    "SY01": ["communication", "problem_solving", "teamwork"],
    "IT01": ["communication", "english", "teamwork"],
    "CV01": ["english", "problem_solving", "self_learning"],
    "NW01": ["problem_solving", "time_management", "self_learning"],
    "SC01": ["problem_solving", "english", "self_learning"],
    "PT01": ["problem_solving", "self_learning", "english"],
    "SG01": ["time_management", "english", "problem_solving"],
    "GR01": ["self_learning", "teamwork", "time_management"],
}

SOFT_SKILL_LABELS = {s["key"]: s["label"] for s in SOFT_SKILL_OPTIONS}

# Số nghề cụ thể được khuyến nghị cuối cùng ("TOP 3 NGHỀ"). Nếu nhóm nghề đứng đầu
# có ít hơn TOP_N_CAREERS nghề (vd. "Phát triển Web" chỉ có 2), phần còn thiếu được
# bổ sung từ nhóm nghề xếp hạng kế tiếp (xem cbf_recommender.recommend_top_careers).
TOP_N_CAREERS = 3
SKILL_BONUS = 0.5

# Nguồn gốc nhãn career_group trong data/students.csv:
#   "derived"  = nhãn SUY RA từ bảng điểm bằng công thức dominant_group() (nhãn giả /
#                pseudo-label — không phải lựa chọn nghề thực tế của sinh viên);
#   "feedback" = nhãn do NGƯỜI DÙNG xác nhận qua nút "phù hợp" (retrain_service.py).
LABEL_SOURCE_COL = "label_source"
LABEL_DERIVED = "derived"
LABEL_FEEDBACK = "feedback"

# Số phản hồi tối thiểu để báo cáo độ chính xác so với nhãn do người dùng xác nhận.
MIN_FEEDBACK_FOR_VALIDATION = 20

# Tên hiển thị của các thuật toán Tầng 1 (khoá = tên lớp scikit-learn của bước "model" trong pipeline).
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression (multinomial)",
    "RandomForestClassifier": "Random Forest",
    "GradientBoostingClassifier": "Gradient Boosting",
}
