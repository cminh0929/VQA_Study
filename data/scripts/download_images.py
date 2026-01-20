"""
Download COCO Images
Script để download ảnh từ danh sách đã lọc
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from coco_data import COCODownloader


def parse_args():
    parser = argparse.ArgumentParser(description='Download COCO Images')
    parser.add_argument('--input', type=str, default='data/coco/filtered_images.json',
                        help='Input filtered images JSON')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    return parser.parse_args()


def download_images_parallel(downloader, filtered_images, max_workers=8):
    """Download ảnh song song"""
    print(f"\nDownloading {len(filtered_images)} images...")
    print(f"Workers: {max_workers}")
    print(f"Destination: {downloader.images_dir}")
    
    success_count = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {
            executor.submit(downloader.download_image, img['file_name']): img 
            for img in filtered_images
        }
        
        # Progress
        with tqdm(total=len(filtered_images), desc="Downloading") as pbar:
            for future in as_completed(futures):
                success = future.result()
                
                if success:
                    success_count += 1
                else:
                    img = futures[future]
                    failed.append(img['file_name'])
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'failed': len(failed)
                })
    
    print(f"\n✓ Downloaded: {success_count}/{len(filtered_images)}")
    
    if failed:
        print(f"\n⚠ Failed: {len(failed)}")
        if len(failed) <= 10:
            for f in failed:
                print(f"  - {f}")


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("DOWNLOADING COCO IMAGES")
    print("="*60)
    
    # Load filtered images
    if not os.path.exists(args.input):
        print(f"\n❌ Error: {args.input} not found")
        print("Please run: python prepare_coco.py")
        return
    
    with open(args.input, 'r') as f:
        filtered_images = json.load(f)
    
    print(f"\nLoaded {len(filtered_images)} filtered images")
    
    # Download
    downloader = COCODownloader()
    download_images_parallel(
        downloader,
        filtered_images,
        max_workers=args.workers
    )
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETED!")
    print("="*60)
    print(f"\n✓ Images saved to: {downloader.images_dir}")
    print(f"\n📝 Next step:")
    print(f"  python generate_qa.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
