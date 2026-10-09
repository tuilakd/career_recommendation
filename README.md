# Báo cáo đồ án môn học: Xây dựng hệ thống gợi ý lựa chọn nghề nghiệp cho sinh viên CNTT

## 1. Giới thiệu bài toán

Sinh viên Công nghệ thông tin có nhiều lựa chọn nghề nghiệp, mỗi nghề có yêu cầu khác nhau về
kiến thức chuyên môn và kỹ năng mềm. Đồ án xây dựng một ứng dụng web hỗ trợ sinh viên tham khảo
các nhóm nghề và nghề cụ thể dựa trên kết quả học tập, lĩnh vực quan tâm và mức độ tự đánh giá
kỹ năng mềm.

Đầu vào của hệ thống là điểm các học phần đã có, cùng lựa chọn sở thích và kỹ năng mềm của người
dùng. Điểm có thể được nhập trực tiếp hoặc đọc từ tệp bảng điểm CSV/XLSX; môn chưa học có thể để
trống. Mỗi điểm hợp lệ nằm trong khoảng 0–10. Khi thiếu môn, hệ thống dùng điểm trung bình huấn
luyện làm giá trị trung tính cho các đặc trưng còn thiếu ở tầng phân loại và chỉ tính CBF trên
các môn có điểm. Kết quả được đánh dấu là gợi ý sơ bộ cùng độ phủ số môn (N/12); các phản hồi có
bảng điểm chưa đủ 12 môn không được dùng để huấn luyện lại.
Tệp tải lên chỉ tự nhận diện 12 học phần đã khai báo; với chương trình dùng tên môn khác, người
dùng có thể nhập điểm thủ công vào môn tương đương gần nhất.

Đầu ra gồm nhóm nghề được xếp hạng cao nhất, tối đa ba nghề phù hợp trong nhóm, mức độ phù hợp
CBF, thông tin đối chiếu học phần và kỹ năng, cùng gợi ý học phần nên ưu tiên. Các kết quả có
tính chất tham khảo, không thay thế tư vấn nghề nghiệp của chuyên gia.

## 2. Kiến trúc hệ thống

Hệ thống được tổ chức thành hai tầng khuyến nghị:

```text
Điểm 12 học phần ──► StandardScaler ──► Mô hình phân loại ──► Xác suất 7 nhóm nghề
                                                ▲
                          Sở thích và kỹ năng mềm ── điều chỉnh thứ hạng nhóm
                                                │
                                                ▼
Hồ sơ điểm + hồ sơ nghề + kỹ năng + phản hồi ──► CBF ──► Top 3 nghề
```

### Tầng 1: Phân loại nhóm nghề

Mô hình phân loại nhận vector 12 học phần sau khi chuẩn hóa bằng `StandardScaler`, sau đó ước
lượng xác suất thuộc 7 nhóm nghề. Với môn thiếu điểm, giá trị trung bình của môn đó trong dữ liệu
huấn luyện được dùng để vector vẫn đúng định dạng mà không diễn giải môn chưa học là điểm 0.
Kết quả được điều chỉnh theo sở thích và kỹ năng mềm đã khai báo để xác định thứ hạng nhóm nghề.

`train.py` so sánh Logistic Regression đa lớp, Random Forest và Gradient Boosting; thuật toán
có F1-weighted 5-fold CV cao nhất được lưu để triển khai. Mô hình hiện tại là Logistic Regression
đa lớp, lưu trong `model_best.joblib`.

### Tầng 2: Xếp hạng nghề bằng Content-Based Filtering

`cbf_recommender.py` so sánh hồ sơ học tập với hồ sơ của từng nghề trên các học phần đã có điểm
(hoặc toàn bộ 12 môn nếu hồ sơ đầy đủ). Vector
hồ sơ nghề được xây dựng từ trọng số tri thức chuyên gia: môn không liên quan có giá trị 5,
môn cốt lõi có giá trị 10 và các mức trung gian được quy đổi tuyến tính về khoảng 5–10.

Điểm nền CBF được tính như sau:

```text
Điểm nền = 0,55 × Cosine% + 0,45 × Mastery
```

Trong đó Cosine% biểu diễn độ tương đồng giữa hồ sơ sinh viên và nghề; điểm sinh viên được chuẩn
hóa z-score theo môn, còn vector nghề được tâm hóa trước khi tính. Mastery là điểm trung bình có
trọng số của các môn trọng tâm. Điểm nghề còn được hiệu chỉnh theo mức độ phù hợp nhóm nghề,
kỹ năng mềm và phản hồi cộng đồng; điểm cuối được giới hạn trong khoảng 45–98,5.

Hệ thống chọn tối đa ba nghề đứng đầu trong nhóm được ưu tiên. Nếu nhóm đó có ít hơn ba nghề,
các vị trí thiếu được bổ sung từ nhóm xếp hạng kế tiếp. Trang kết quả cũng trình bày thông tin
nghề, điểm mạnh/cần cải thiện theo môn học và kỹ năng mềm.

## 3. Quy trình dữ liệu và huấn luyện

### 3.1. Chuẩn bị dữ liệu

Tệp `DATASET.xlsx` chứa bảng điểm và danh mục nghề. Chạy `prepare_data.py` để xử lý dữ liệu và
tạo:

- `data/students.csv`: hồ sơ điểm cùng nhãn nhóm nghề.
- `data/careers.json`: danh mục nghề và thông tin nhóm nghề.

Tên 12 học phần, nhóm nghề, trọng số và tham số dùng chung được khai báo tập trung trong
`domain_knowledge.py`.

### 3.2. Tạo nhãn nhóm nghề

Dữ liệu nguồn chỉ có bảng điểm, không có lựa chọn nghề thực tế đã xác nhận của từng sinh viên.
Vì vậy, nhãn `career_group` được suy ra từ điểm: chuẩn hóa theo thống kê tham chiếu của từng môn,
tính điểm phù hợp theo ma trận trọng số môn học–nhóm nghề và chọn nhóm có điểm cao nhất. Nhãn
suy ra được đánh dấu `label_source = "derived"`. Các nhãn có nguồn từ phản hồi người dùng được
đánh dấu riêng `label_source = "feedback"`.

### 3.3. Huấn luyện và chọn mô hình

`train.py` thực hiện các bước:

1. Đọc dữ liệu đã chuẩn bị và tách riêng các mẫu có nhãn suy ra để so sánh thuật toán.
2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80/20.
3. Huấn luyện và đánh giá ba thuật toán Logistic Regression, Random Forest và Gradient Boosting.
4. Tính F1-weighted bằng kiểm định chéo phân tầng 5-fold.
5. Chọn thuật toán có F1 5-fold CV cao nhất, huấn luyện pipeline triển khai trên toàn bộ dữ liệu
   khả dụng và lưu vào `model_best.joblib`.

Các trọng số môn học–nhóm nghề và hồ sơ nghề trong `domain_knowledge.py` là tri thức chuyên gia
được định nghĩa trước, không phải tham số được mô hình học từ bảng điểm.

## 4. Cài đặt và chạy chương trình

Ứng dụng sử dụng Python, Flask, pandas, scikit-learn, joblib và openpyxl. Có thể chạy bằng Python
cài đặt trên máy mà không cần kích hoạt môi trường ảo. Mở PowerShell tại thư mục dự án và cài
các thư viện:

```powershell
py -m pip install flask pandas scikit-learn joblib openpyxl
```

Chuẩn bị dữ liệu, huấn luyện mô hình, đánh giá, kiểm thử và khởi chạy ứng dụng:

```powershell
py prepare_data.py
py train.py
py eval_system.py
py -m unittest discover -s tests -v
py app.py
```

Sau khi chạy ứng dụng, truy cập `http://127.0.0.1:5000`. Trang quản trị ở `/admin_login`; mật
khẩu mặc định là `admin123`. Khi triển khai, cần thay mật khẩu bằng biến môi trường
`ADMIN_PASSWORD`.

Nếu xuất hiện cảnh báo `InconsistentVersionWarning` khi nạp `model_best.joblib`, hãy chạy lại
`py train.py` bằng phiên bản scikit-learn hiện tại.

## 5. Kết quả thực nghiệm

Các số liệu dưới đây được ghi nhận khi chạy `train.py` với scikit-learn 1.8.0. Tập kiểm tra dùng
một lần chia 80/20; F1-weighted 5-fold CV là tiêu chí được dùng để lựa chọn mô hình triển khai.

| Thuật toán | Accuracy | Precision | Recall | F1 (test) | F1 (5-fold CV) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (triển khai) | 0,761 | 0,764 | 0,761 | 0,755 | **0,856** |
| Random Forest | 0,704 | 0,688 | 0,704 | 0,668 | 0,721 |
| Gradient Boosting | 0,676 | 0,694 | 0,676 | 0,676 | 0,684 |

Kết quả xếp hạng nhóm nghề trong kiểm định 5-fold CV:

| Phương pháp | Top-1 | Top-3 |
|---|---:|---:|
| Logistic Regression (triển khai) | 85,6% | 99,2% |
| Random Forest | 73,7% | 94,1% |
| Baseline: luôn chọn nhóm phổ biến nhất | 22,3% | 57,1% |

Đối với tầng CBF, các kết quả được ghi nhận gồm: độ phân định vị trí thứ nhất 97,7%, khoảng
cách trung bình giữa vị trí thứ nhất và thứ hai 2,46%, độ phủ danh mục 21/21 nghề và mức tương
thích với thế mạnh học phần 98,3%.

**Diễn giải kết quả:** Nhãn nhóm nghề trong dữ liệu được suy ra từ chính điểm học phần bằng
ma trận trọng số, do đó các chỉ số phân loại phản ánh mức độ mô hình khớp với quy tắc tạo nhãn,
không phải độ chính xác dự đoán nghề thực tế. Các chỉ số CBF mô tả hoạt động của thuật toán;
hiện chưa có nhãn nghề thực tế để đánh giá trực tiếp chất lượng xếp hạng.

## 6. Cấu trúc chương trình

| Tệp/thư mục | Chức năng |
|---|---|
| `DATASET.xlsx` | Dữ liệu nguồn gồm bảng điểm và danh mục nghề. |
| `domain_knowledge.py` | Khai báo học phần, nhóm nghề, trọng số, hồ sơ nghề và tham số dùng chung. |
| `prepare_data.py` | Xử lý dữ liệu nguồn, tạo `data/students.csv` và `data/careers.json`. |
| `train.py` | Huấn luyện, so sánh thuật toán và lưu mô hình triển khai. |
| `cbf_recommender.py` | Tính điểm tương đồng và xếp hạng nghề bằng CBF. |
| `eval_system.py` | Đánh giá hệ thống và kiểm tra tính nhất quán hồ sơ nghề. |
| `retrain_service.py` | Huấn luyện lại mô hình từ phản hồi người dùng hợp lệ. |
| `app.py` | Ứng dụng Flask, xử lý nhập liệu, dự đoán và chức năng quản trị. |
| `feedback_store.py` | Lưu trữ và tổng hợp phản hồi, lượt tương tác. |
| `templates/`, `static/` | Các tệp giao diện HTML và CSS. |
| `tests/test_system.py` | Kiểm thử hồi quy bằng `unittest`. |
| `data/students.csv` | Dữ liệu sinh viên đã xử lý và nguồn nhãn. |
| `data/careers.json` | Danh mục nghề được ứng dụng sử dụng. |
| `model_best.joblib` | Pipeline tiền xử lý và mô hình phân loại đã huấn luyện. |
"# career_recommendation" 
