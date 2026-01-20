"""
COCO Data Filter
Module để lọc ảnh động vật từ COCO
"""

from tqdm import tqdm


class COCOAnimalFilter:
    """Lọc ảnh có động vật từ COCO dataset"""
    
    def __init__(self, animal_categories):
        """
        Args:
            animal_categories: Dict mapping category_id -> animal_name
        """
        self.animal_categories = animal_categories
    
    def filter_images(self, coco_data, max_images=5000, min_area=5000):
        """
        Lọc ảnh có động vật
        
        Args:
            coco_data: COCO annotations
            max_images: Số ảnh tối đa
            min_area: Diện tích tối thiểu của bbox
            
        Returns:
            list: Filtered image info
        """
        print(f"\nFiltering animal images...")
        print(f"  Max images: {max_images}")
        print(f"  Min bbox area: {min_area}")
        
        # Build mappings
        image_annotations = self._build_image_annotations(coco_data)
        images_dict = {img['id']: img for img in coco_data['images']}
        
        # Filter
        filtered_images = []
        
        for image_id, annotations in tqdm(image_annotations.items(), desc="Filtering"):
            # Check animals
            animal_anns = [
                ann for ann in annotations 
                if ann['category_id'] in self.animal_categories 
                and ann['area'] >= min_area
            ]
            
            if not animal_anns:
                continue
            
            # Get image info
            image_info = images_dict.get(image_id)
            if not image_info:
                continue
            
            # Process
            filtered_info = self._process_image(
                image_info, 
                animal_anns
            )
            filtered_images.append(filtered_info)
            
            if len(filtered_images) >= max_images:
                break
        
        print(f"✓ Filtered {len(filtered_images)} images")
        return filtered_images
    
    def _build_image_annotations(self, coco_data):
        """Build image_id -> annotations mapping"""
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        return image_annotations
    
    def _process_image(self, image_info, animal_anns):
        """Process một ảnh đã lọc"""
        # Get main animal (largest)
        main_animal = max(animal_anns, key=lambda x: x['area'])
        
        # Count by category
        animal_counts = {}
        for ann in animal_anns:
            cat_id = ann['category_id']
            animal_counts[cat_id] = animal_counts.get(cat_id, 0) + 1
        
        return {
            'image_id': image_info['id'],
            'file_name': image_info['file_name'],
            'width': image_info['width'],
            'height': image_info['height'],
            'main_animal_id': main_animal['category_id'],
            'main_animal_name': self.animal_categories[main_animal['category_id']],
            'main_animal_bbox': main_animal['bbox'],
            'animal_counts': animal_counts,
            'total_animals': len(animal_anns),
            'all_bboxes': [ann['bbox'] for ann in animal_anns]
        }
    
    def get_statistics(self, filtered_images):
        """
        Tính thống kê
        
        Args:
            filtered_images: Danh sách ảnh đã lọc
            
        Returns:
            dict: Statistics
        """
        stats = {
            'total': len(filtered_images),
            'by_animal': {},
            'by_count': {}
        }
        
        for img in filtered_images:
            # By animal
            animal = img['main_animal_name']
            stats['by_animal'][animal] = stats['by_animal'].get(animal, 0) + 1
            
            # By count
            count = img['total_animals']
            stats['by_count'][count] = stats['by_count'].get(count, 0) + 1
        
        return stats
    
    def print_statistics(self, stats):
        """In thống kê"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal images: {stats['total']}")
        
        print(f"\nBy animal type:")
        for animal, count in sorted(stats['by_animal'].items(), key=lambda x: -x[1]):
            pct = count / stats['total'] * 100
            print(f"  {animal:12s}: {count:4d} ({pct:.1f}%)")
        
        print(f"\nBy number of animals:")
        for count in sorted(stats['by_count'].keys()):
            num_images = stats['by_count'][count]
            pct = num_images / stats['total'] * 100
            print(f"  {count} animal(s): {num_images:4d} ({pct:.1f}%)")
