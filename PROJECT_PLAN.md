# VISUAL QUESTION ANSWERING (VQA)

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Mục Tiêu Chính

Xây dựng và triển khai một hệ thống Trả lời câu hỏi dựa trên hình ảnh (VQA) tập trung vào 3 loại câu hỏi cụ thể:

- **Nhận diện động vật**: Xác định loài vật trong ảnh (Ví dụ: "What animal is in the image?" → "dog", "cat").
- **Nhận diện màu sắc**: Xác định màu sắc của đối tượng (Ví dụ: "What color is the cat?" → "black", "white").
- **Đếm số lượng**: Đếm số lượng đối tượng (Ví dụ: "How many dogs are there?" → "1", "2", "3").

### 1.2. Phạm Vi & Dữ Liệu

- **Nguồn dữ liệu**: MS COCO Dataset.
- **Quy mô**: Khoảng 3.000 - 5.000 hình ảnh chứa 10 loại động vật phổ biến.
- **Phương pháp**: So sánh hiệu quả giữa các mô hình kiến trúc từ đơn giản đến phức tạp.

### 1.3. Thông Số Kỹ Thuật

- **Framework**: PyTorch.
- **Xử lý hình ảnh (Backbone)**: ResNet50 (Pretrained trên ImageNet).
- **Xử lý ngôn ngữ**: Tokenization cơ bản hoặc FastText embeddings (tùy chọn).
- **Chỉ số đánh giá (Metrics)**: Độ chính xác (Accuracy) tổng thể và theo từng loại câu hỏi.

---

## 2. QUY TRÌNH THỰC HIỆN (ROADMAP)

Dự án được chia thành 4 giai đoạn chính:

### Giai Đoạn 1: Chuẩn Bị Dữ Liệu (Data Preparation)

**Trạng thái hiện tại:**
- [Đã hoàn thành] Tải tệp chú thích (annotations) từ COCO.
- [Đã hoàn thành] Lọc danh sách ảnh có chứa động vật.
- [Đã hoàn thành] Tải hình ảnh về máy cục bộ.

**Công việc cần làm ngay:**
- [Đang thực hiện] Tự động sinh câu hỏi và câu trả lời từ dữ liệu gốc.
- [Chờ thực hiện] Chia tập dữ liệu theo tỷ lệ: 70% Train / 15% Validation / 15% Test.

### Giai Đoạn 2: Tiền Xử Lý (Preprocessing)

- **Xử lý hình ảnh**: Resize về kích thước 224x224, chuẩn hóa (normalize) và tăng cường dữ liệu (augmentation).
- **Xử lý văn bản**: Tách từ (tokenize), xây dựng từ điển (vocabulary), đệm câu (padding).
- **Mã hóa**: Chuyển đổi câu trả lời text sang dạng nhãn số (labels).
- **Đóng gói**: Tạo DataLoader tương thích với PyTorch.

### Giai Đoạn 3: Huấn Luyện Mô Hình (Training)

- Huấn luyện từng biến thể mô hình.
- Số lượng epochs dự kiến: 50.
- Lưu trữ checkpoints định kỳ.
- Ghi lại nhật ký huấn luyện (Loss, Accuracy) để theo dõi.

### Giai Đoạn 4: Đánh Giá & Phân Tích (Evaluation)

- Kiểm tra độ chính xác trên tập Test.
- Lập bảng so sánh kết quả giữa các mô hình.
- Vẽ biểu đồ đường cong huấn luyện (Training curves).
- Trực quan hóa sự tập trung của mô hình (Visualize Attention Maps).
- Tổng hợp báo cáo kết luận.

---

## 3. HƯỚNG DẪN THỰC THI (COMMAND GUIDE)

> **Lưu ý**: Phần này chưa thực hiện.
