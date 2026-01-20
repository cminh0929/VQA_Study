"""
Prepare COCO Dataset
Script chính để chuẩn bị dữ liệu COCO
"""

import os
import json
import argparse
from coco_data import COCODownloader, COCOAnimalFilter


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare COCO Dataset')
    parser.add_argument('--max_images', type=int, default=5000,
                        help='Maximum number of images to filter')
    parser.add_argument('--min_area', type=int, default=5000,
                        help='Minimum bbox area')
    parser.add_argument('--output', type=str, default='data/coco/filtered_images.json',
                        help='Output file for filtered images')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("PREPARING COCO DATASET")
    print("="*60)
    
    # Initialize
    downloader = COCODownloader()
    filter_obj = COCOAnimalFilter(COCODownloader.ANIMAL_CATEGORIES)
    
    # Step 1: Download annotations
    downloader.download_annotations()
    
    # Step 2: Load annotations
    print("\n" + "="*60)
    print("LOADING ANNOTATIONS")
    print("="*60)
    
    train_data = downloader.load_annotations("train")
    val_data = downloader.load_annotations("val")
    
    print(f"\nTrain: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    
    # Step 3: Filter
    print("\n" + "="*60)
    print("FILTERING ANIMAL IMAGES")
    print("="*60)
    
    # Filter train (80% of max)
    train_max = int(args.max_images * 0.8)
    train_filtered = filter_obj.filter_images(
        train_data,
        max_images=train_max,
        min_area=args.min_area
    )
    
    # Filter val (20% of max)
    val_max = args.max_images - len(train_filtered)
    val_filtered = filter_obj.filter_images(
        val_data,
        max_images=val_max,
        min_area=args.min_area
    )
    
    # Combine
    all_filtered = train_filtered + val_filtered
    
    # Step 4: Statistics
    stats = filter_obj.get_statistics(all_filtered)
    filter_obj.print_statistics(stats)
    
    # Step 5: Save
    print("\n" + "="*60)
    print("SAVING FILTERED DATA")
    print("="*60)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_filtered, f, indent=2)
    
    print(f"\n✓ Saved to: {args.output}")
    
    # Summary
    print("\n" + "="*60)
    print("PREPARATION COMPLETED!")
    print("="*60)
    print(f"\n✓ Total images: {len(all_filtered)}")
    print(f"✓ Output: {args.output}")
    print(f"\n📝 Next step:")
    print(f"  python download_images.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
