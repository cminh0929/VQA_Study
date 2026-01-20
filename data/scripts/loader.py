"""
COCO Data Loader
Module để tải và xử lý COCO dataset
"""

import os
import json
import requests
from tqdm import tqdm
import zipfile
from pathlib import Path


class COCODownloader:
    """Download và quản lý COCO dataset"""
    
    # Animal categories trong COCO
    ANIMAL_CATEGORIES = {
        16: "bird",
        17: "cat", 
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe"
    }
    
    def __init__(self, data_dir="data/coco"):
        """
        Args:
            data_dir: Thư mục lưu COCO data
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
    
    def download_annotations(self):
        """Download COCO annotations"""
        print("\n" + "="*60)
        print("DOWNLOADING COCO ANNOTATIONS")
        print("="*60)
        
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(self.data_dir, "annotations_trainval2017.zip")
        
        if os.path.exists(os.path.join(self.annotations_dir, "instances_train2017.json")):
            print("✓ Annotations already exist")
            return
        
        # Download
        print(f"\nDownloading annotations (~250MB)...")
        self._download_file(url, zip_path)
        
        # Extract
        print(f"\nExtracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        print("✓ Annotations downloaded")
    
    def _download_file(self, url, destination):
        """Download file với progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def load_annotations(self, split="train"):
        """
        Load COCO annotations
        
        Args:
            split: "train" hoặc "val"
            
        Returns:
            dict: COCO annotations
        """
        anno_file = os.path.join(self.annotations_dir, f"instances_{split}2017.json")
        
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"Annotation file not found: {anno_file}")
        
        print(f"Loading {split} annotations...")
        with open(anno_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def get_image_url(self, file_name):
        """Get URL của ảnh COCO"""
        if "train" in file_name:
            return f"http://images.cocodataset.org/train2017/{file_name}"
        else:
            return f"http://images.cocodataset.org/val2017/{file_name}"
    
    def download_image(self, file_name):
        """
        Download một ảnh
        
        Args:
            file_name: Tên file ảnh
            
        Returns:
            bool: Success or not
        """
        url = self.get_image_url(file_name)
        dest_path = os.path.join(self.images_dir, file_name)
        
        # Skip if exists
        if os.path.exists(dest_path):
            return True
        
        # Download
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
            return False
