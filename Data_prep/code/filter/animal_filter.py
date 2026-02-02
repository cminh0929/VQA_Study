import os
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
SOURCE_DIR = r'C:\Users\cminh\Downloads\train2017\train2017'  # Thư mục chứa ảnh gốc
BASE_DATA_DIR = r'c:\Users\cminh\Desktop\Code\Deeplearning\VQA_Workspace\Data_prep\data'
ANIMAL_DIR = os.path.join(BASE_DATA_DIR, 'images')  # Chứa ảnh có animal
RAW_DIR = os.path.join(BASE_DATA_DIR, 'raw')        # Chứa ảnh không có animal

# Cấu hình YOLO
CONF_THRESHOLD = 0.5  # Độ tự tin tối thiểu (50%)

# Định nghĩa 10 mã ID của các con vật trong mô hình YOLOv8 chuẩn (COCO)
# 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 20: elephant, 21: bear, 22: zebra, 23: giraffe
ANIMAL_IDS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ANIMAL_NAMES = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"
}

def filter_animals():
    # 1. Tạo các thư mục nếu chưa có
    for folder in [ANIMAL_DIR, RAW_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Đã tạo thư mục: {folder}")

    # 2. Kiểm tra thư mục nguồn
    if not os.path.exists(SOURCE_DIR):
        print(f"Lỗi: Thư mục nguồn {SOURCE_DIR} không tồn tại!")
        return

    # 3. Tải mô hình YOLOv8 (tự động tải nếu chưa có)
    print("Đang tải mô hình YOLOv8...")
    model = YOLO("yolov8n.pt")

    # 4. Lấy danh sách tất cả ảnh trong thư mục nguồn
    image_files = [f for f in os.listdir(SOURCE_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục nguồn.")

    # 5. Phân loại và copy ảnh
    animal_count = 0
    raw_count = 0
    
    print("Bắt đầu phân loại ảnh...")
    for filename in tqdm(image_files, desc="Processing"):
        src_path = os.path.join(SOURCE_DIR, filename)
        
        # Chạy YOLO trên ảnh
        results = model(src_path, verbose=False)[0]
        
        found_animal = False
        detected_names = []

        # Kiểm tra các vật thể detect được
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if class_id in ANIMAL_IDS and conf >= CONF_THRESHOLD:
                found_animal = True
                detected_names.append(ANIMAL_NAMES[class_id])

        # Copy ảnh vào thư mục tương ứng
        if found_animal:
            dest_path = os.path.join(ANIMAL_DIR, filename)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
            animal_count += 1
        else:
            dest_path = os.path.join(RAW_DIR, filename)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
            raw_count += 1

    print(f"\n--- HOÀN THÀNH ---")
    print(f"Ảnh có động vật (đã đưa vào 'data/images'): {animal_count}")
    print(f"Ảnh không có động vật (đã đưa vào 'data/raw'): {raw_count}")
    print(f"Tổng cộng: {animal_count + raw_count} ảnh")

if __name__ == "__main__":
    filter_animals()
