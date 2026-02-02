# Tài liệu: Vai trò của YOLO trong giai đoạn Pre-data (VQA Project)

Trong dự án xây dựng dữ liệu Visual Question Answering (VQA) cho động vật, **YOLO (You Only Look Once)** đóng vai trò là "bộ não sơ cấp" giúp tự động hóa các công việc nặng nhọc trước khi đưa dữ liệu vào huấn luyện mô hình VQA chính.

---

## 1. Chức năng chính của YOLO

### A. Lọc dữ liệu (Data Filtering) - Tại folder `code/filter/`
Khi có hàng nghìn ảnh trong `data/raw/`, không phải ảnh nào cũng dùng được. YOLO giúp:
*   **Loại bỏ ảnh rác:** Tự động xóa hoặc di chuyển các ảnh không chứa đối tượng mục tiêu (không thấy chó, mèo, chim...).
*   **Kiểm soát chất lượng:** Chỉ giữ lại những ảnh có độ tự tin (Confidence Score) cao (ví dụ > 0.6) để đảm bảo mô hình VQA không học từ dữ liệu sai.
*   **Phân loại sơ bộ:** Chia ảnh vào các thư mục con dựa trên loài vật để dễ quản lý.

### B. Trích xuất Metadata (Feature Extraction)
YOLO cung cấp các thông tin "vàng" từ mỗi bức ảnh mà mô hình ngôn ngữ không thể tự lấy được một cách chính xác tuyệt đối:
*   **Label (Nhãn):** Tên loài động vật.
*   **Bounding Box (Tọa độ):** Vị trí chính xác ($x, y, w, h$) của con vật trong khung hình.
*   **Số lượng:** Đếm chính xác có bao nhiêu đối tượng trong một ảnh đơn lẻ hoặc ảnh phức tạp.

### C. Sinh câu hỏi tự động (Automated Question Generation)
Dựa trên Metadata từ YOLO, chúng ta sử dụng các thuật toán logic để tạo ra hàng vạn cặp QA mà không cần tốn phí API:
*   **Câu hỏi hiện diện:** Nếu YOLO detect "Dog" -> Tạo câu hỏi: "Có con chó không?" -> Trả lời: "Có".
*   **Câu hỏi số lượng:** Nếu YOLO có 3 boxes "Cat" -> Tạo câu hỏi: "Có bao nhiêu con mèo?" -> Trả lời: "3".
*   **Câu hỏi vị trí:** Dựa vào tọa độ $x$ của Bounding Box -> Trả lời: "Bên trái", "Bên phải" hoặc "Ở giữa".

---

## 2. Luồng công việc (Workflow)

1.  **Input:** Ảnh gốc từ `data/raw/`.
2.  **YOLO Inference:** Chạy mô hình YOLO trên tập `raw`.
3.  **Filter:** Code tại `code/filter/` đọc kết quả YOLO, di chuyển ảnh "đạt chuẩn" sang `data/dataset/`.
4.  **Annotations:** Code tại `code/scripts/generate_vqa_annotations.py` đọc file kết quả (`.txt`) của YOLO để xuất ra file `annotations.json` cuối cùng.

---

## 3. Lợi ích so với dùng API (như Gemini/GPT-4)

| Đặc điểm | Sử dụng YOLO (Local) | Sử dụng LLM API |
| :--- | :--- | :--- |
| **Chi phí** | **Miễn phí hoàn toàn** | Tốn phí theo lượt gọi (Token) |
| **Tốc độ** | Rất nhanh (hàng chục ảnh/giây) | Chậm (phụ thuộc đường truyền) |
| **Độ chính xác đếm** | Rất cao (dựa trên Box) | Đôi khi bị ảo giác (Hallucination) |
| **Giới hạn** | Hàng triệu ảnh cũng được | Dễ bị Rate Limit (giới hạn lượt gọi) |

---
*Tài liệu này được soạn thảo để định hướng xây dựng pipeline dữ liệu VQA tự động.*
