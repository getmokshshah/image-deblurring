#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path
import random
import zipfile
import urllib.request

import cv2
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image


class DatasetDownloader:
    def __init__(self, data_dir='./data', num_train=800, num_val=100):
        self.data_dir = Path(data_dir)
        self.num_train = num_train
        self.num_val = num_val
        
        # Create directories
        self.div2k_dir = self.data_dir / 'DIV2K'
        self.processed_dir = self.data_dir / 'processed'
        
        self.train_div2k = self.div2k_dir / 'train'
        self.val_div2k = self.div2k_dir / 'val'
        
        self.train_processed = self.processed_dir / 'train'
        self.val_processed = self.processed_dir / 'val'
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.train_div2k,
            self.val_div2k,
            self.train_processed / 'blurred',
            self.train_processed / 'sharp',
            self.val_processed / 'blurred',
            self.val_processed / 'sharp'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directory structure in {self.data_dir}")
    
    def download_div2k(self):
        """Download DIV2K dataset using Hugging Face datasets"""
        print("Downloading DIV2K dataset from Hugging Face...")
        
        try:
            # Load DIV2K dataset - try different configurations
            print("Loading DIV2K training set...")
            
            # First try to load with no specific config to see what's available
            try:
                train_dataset = load_dataset(
                    "eugenesiow/Div2k", 
                    split="train", 
                    cache_dir=str(self.data_dir / '.cache'),
                    trust_remote_code=True
                )
            except:
                # If that fails, try with the bicubic_x4 config but handle it differently
                train_dataset = load_dataset(
                    "eugenesiow/Div2k", 
                    "bicubic_x4",
                    split="train", 
                    cache_dir=str(self.data_dir / '.cache'),
                    trust_remote_code=True
                )
            
            print("Loading DIV2K validation set...")
            try:
                val_dataset = load_dataset(
                    "eugenesiow/Div2k",
                    split="validation",
                    cache_dir=str(self.data_dir / '.cache'),
                    trust_remote_code=True
                )
            except:
                val_dataset = load_dataset(
                    "eugenesiow/Div2k",
                    "bicubic_x4",
                    split="validation",
                    cache_dir=str(self.data_dir / '.cache'),
                    trust_remote_code=True
                )
            
            # Print dataset info to understand structure
            print(f"Train dataset features: {train_dataset.features}")
            print(f"First item keys: {train_dataset[0].keys() if len(train_dataset) > 0 else 'Empty dataset'}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def download_div2k_direct(self):
        """Alternative method to download DIV2K images directly"""
        print("Downloading DIV2K dataset directly...")
        
        # DIV2K dataset URLs
        base_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
        train_hr_url = base_url + "DIV2K_train_HR.zip"
        valid_hr_url = base_url + "DIV2K_valid_HR.zip"
        
        try:
            # Download training images
            if not (self.train_div2k / "0001.png").exists():
                print("Downloading DIV2K training images...")
                train_zip = self.data_dir / "DIV2K_train_HR.zip"
                
                if not train_zip.exists():
                    print(f"Downloading from {train_hr_url}")
                    urllib.request.urlretrieve(train_hr_url, train_zip)
                
                print("Extracting training images...")
                with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Move images to correct directory
                src_dir = self.data_dir / "DIV2K_train_HR"
                if src_dir.exists():
                    for img in src_dir.glob("*.png"):
                        shutil.move(str(img), str(self.train_div2k / img.name))
                    src_dir.rmdir()
            
            # Download validation images  
            if not (self.val_div2k / "0801.png").exists():
                print("Downloading DIV2K validation images...")
                val_zip = self.data_dir / "DIV2K_valid_HR.zip"
                
                if not val_zip.exists():
                    print(f"Downloading from {valid_hr_url}")
                    urllib.request.urlretrieve(valid_hr_url, val_zip)
                
                print("Extracting validation images...")
                with zipfile.ZipFile(val_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Move images to correct directory
                src_dir = self.data_dir / "DIV2K_valid_HR"
                if src_dir.exists():
                    for img in src_dir.glob("*.png"):
                        shutil.move(str(img), str(self.val_div2k / img.name))
                    src_dir.rmdir()
            
            return True
            
        except Exception as e:
            print(f"Error downloading DIV2K directly: {e}")
            return False
    def save_images_from_dataset(self, dataset, save_dir, max_images, dataset_name):
        """Save images from HuggingFace dataset to disk"""
        print(f"Saving {dataset_name} images to {save_dir}")
        
        count = 0
        for i, item in enumerate(tqdm(dataset, desc=f"Saving {dataset_name}")):
            if count >= max_images:
                break
            
            try:
                # Debug: print the structure of the item
                if i == 0:
                    print(f"Dataset item keys: {item.keys()}")
                    print(f"Item structure: {item}")
                
                # Try to get the image from different possible keys
                image = None
                
                # Check for 'hr' key (high resolution)
                if 'hr' in item:
                    hr_data = item['hr']
                    if hasattr(hr_data, 'mode'):  # It's a PIL Image
                        image = hr_data
                    else:
                        print(f"HR data type: {type(hr_data)}, content: {str(hr_data)[:200]}")
                
                # Check for 'image' key
                if image is None and 'image' in item:
                    img_data = item['image']
                    if hasattr(img_data, 'mode'):  # It's a PIL Image
                        image = img_data
                
                # If we still don't have an image, try to load from lr (low resolution) and use it
                if image is None and 'lr' in item:
                    lr_data = item['lr']
                    if hasattr(lr_data, 'mode'):  # It's a PIL Image
                        image = lr_data
                        print(f"Warning: Using LR image for item {i} as HR is not available")
                
                if image is None:
                    print(f"Could not find valid image in item {i}")
                    continue
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save image
                save_path = save_dir / f"{count:04d}.png"
                image.save(save_path, 'PNG')
                count += 1
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Saved {count} {dataset_name} images")
        return count
    
    def create_blur_kernel(self, blur_type='gaussian', kernel_size=15):
        """Create blur kernel"""
        if blur_type == 'gaussian':
            sigma = random.uniform(1.0, 4.0)
            kernel = cv2.getGaussianKernel(kernel_size, sigma)
            kernel = np.outer(kernel, kernel)
        
        elif blur_type == 'motion':
            # Motion blur kernel
            angle = random.uniform(0, 180)
            length = random.randint(5, 25)
            
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            
            # Calculate line endpoints
            angle_rad = np.radians(angle)
            x1 = int(center - length/2 * np.cos(angle_rad))
            y1 = int(center - length/2 * np.sin(angle_rad))
            x2 = int(center + length/2 * np.cos(angle_rad))
            y2 = int(center + length/2 * np.sin(angle_rad))
            
            # Draw line on kernel
            cv2.line(kernel, (x1, y1), (x2, y2), 1, 1)
            kernel = kernel / np.sum(kernel)
        
        else:
            # Box blur
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        return kernel
    
    def apply_blur(self, image, blur_type=None):
        """Apply blur to image"""
        if blur_type is None:
            blur_type = random.choice(['gaussian', 'motion', 'gaussian'])  # Favor gaussian
        
        if blur_type == 'gaussian':
            kernel_size = random.choice([15, 21, 25, 31])
            sigma = random.uniform(1.0, 4.0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif blur_type == 'motion':
            kernel = self.create_blur_kernel('motion')
            return cv2.filter2D(image, -1, kernel)
        
        else:  # box blur
            kernel_size = random.choice([9, 15, 21])
            return cv2.blur(image, (kernel_size, kernel_size))
    
    def generate_blur_pairs(self, source_dir, target_blur_dir, target_sharp_dir, 
                           augment_factor=1):
        """Generate blurred/sharp image pairs from source images"""
        
        source_images = list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg'))
        
        if not source_images:
            print(f"No images found in {source_dir}")
            return 0
        
        print(f"Generating blur pairs from {len(source_images)} source images")
        print(f"Augmentation factor: {augment_factor}x")
        
        pair_count = 0
        
        for img_path in tqdm(source_images, desc="Generating blur pairs"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if too large (to manage memory and training time)
            h, w = image.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Generate multiple blur versions if augment_factor > 1
            for aug_idx in range(augment_factor):
                # Apply blur
                blurred_image = self.apply_blur(image.copy())
                
                # Save images
                base_name = f"{pair_count:04d}"
                
                # Save sharp image (RGB format)
                sharp_path = target_sharp_dir / f"{base_name}.png"
                sharp_pil = Image.fromarray(image)
                sharp_pil.save(sharp_path)
                
                # Save blurred image (RGB format)
                blur_path = target_blur_dir / f"{base_name}.png"
                blur_pil = Image.fromarray(blurred_image)
                blur_pil.save(blur_path)
                
                pair_count += 1
        
        print(f"Generated {pair_count} blur/sharp pairs")
        return pair_count
    
    def prepare_blur_dataset(self):
        """Complete pipeline to prepare blur dataset"""
        print("=== Image Deblurring Dataset Setup ===")
        print(f"Target directory: {self.data_dir}")
        
        # Create directories
        self.create_directories()
        
        # Try HuggingFace first, fallback to direct download
        train_dataset, val_dataset = self.download_div2k()
        
        use_direct_download = False
        if train_dataset is None or val_dataset is None:
            print("\nHuggingFace download failed, trying direct download...")
            use_direct_download = True
        else:
            # Try to check if the dataset is usable
            try:
                test_item = train_dataset[0]
                if 'hr' in test_item and isinstance(test_item['hr'], str):
                    print("\nHuggingFace dataset returns file paths, switching to direct download...")
                    use_direct_download = True
            except:
                use_direct_download = True
        
        if use_direct_download:
            success = self.download_div2k_direct()
            if not success:
                print("Failed to download DIV2K dataset!")
                return False
            
            # Now we need to limit the number of images used
            train_images = sorted(list(self.train_div2k.glob("*.png")))[:self.num_train]
            val_images = sorted(list(self.val_div2k.glob("*.png")))[:self.num_val]
            
            print(f"\nUsing {len(train_images)} training images and {len(val_images)} validation images")
        else:
            # Save images to disk from HuggingFace dataset
            print("\n=== Saving Images ===")
            train_count = self.save_images_from_dataset(
                train_dataset, self.train_div2k, self.num_train, "training"
            )
            val_count = self.save_images_from_dataset(
                val_dataset, self.val_div2k, self.num_val, "validation"
            )
        
        # Generate blur pairs
        print("\n=== Generating Synthetic Blur ===")
        
        # Training pairs with augmentation
        train_pairs = self.generate_blur_pairs(
            self.train_div2k,
            self.train_processed / 'blurred',
            self.train_processed / 'sharp',
            augment_factor=2  # Create 2 blur variants per image
        )
        
        # Validation pairs (no augmentation)
        val_pairs = self.generate_blur_pairs(
            self.val_div2k,
            self.val_processed / 'blurred',
            self.val_processed / 'sharp',
            augment_factor=1
        )
        
        # Summary
        print("\n=== Dataset Setup Complete ===")
        print(f"Training pairs: {train_pairs}")
        print(f"Validation pairs: {val_pairs}")
        print(f"Total storage: ~{(train_pairs + val_pairs) * 0.002:.1f} GB")
        print(f"\nDataset ready at: {self.processed_dir}")
        print("\nYou can now train with:")
        print(f"  python train.py --train_path {self.train_processed} --val_path {self.val_processed}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Download and prepare deblurring dataset')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store dataset')
    parser.add_argument('--num_train', type=int, default=600,
                        help='Number of training images to use')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Number of validation images to use')
    parser.add_argument('--prepare-blur', action='store_true',
                        help='Generate synthetic blur pairs')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = DatasetDownloader(
        data_dir=args.data_dir,
        num_train=args.num_train,
        num_val=args.num_val
    )
    
    if args.prepare_blur:
        success = downloader.prepare_blur_dataset()
        if success:
            print("\n✅ Dataset preparation completed successfully!")
        else:
            print("\n❌ Dataset preparation failed!")
            sys.exit(1)
    else:
        print("Use --prepare-blur to download and setup the complete dataset")
        print("Example: python download_dataset.py --prepare-blur")


if __name__ == '__main__':
    main()