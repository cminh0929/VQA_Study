import json
import os

class VQADatasetLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.annotation_path = os.path.join(data_dir, "annotations.json")
        self.image_dir = os.path.join(data_dir, "images")
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.annotation_path):
            print(f"Cảnh báo: Không tìm thấy file {self.annotation_path}")
            return []
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_sample(self, index):
        """Lấy một mẫu dữ liệu: (Đường dẫn ảnh, Câu hỏi, Câu trả lời)"""
        item = self.data[index]
        image_path = os.path.join(self.image_dir, item['image_id'])
        return image_path, item['question'], item['answer']

    def __len__(self):
        return len(self.data)

# Ví dụ cách dùng (khi bạn đã đưa folder data vào dự án chính):
# loader = VQADatasetLoader()
# img, q, a = loader.get_sample(0)
# print(f"Q: {q} -> A: {a}")
