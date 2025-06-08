#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import time

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import get_model
from utils.metrics import denormalize_tensor


class ImageDeblurrer:
    def __init__(self, model_path, model_type='unet', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        
        # Setup transforms
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print("Model loaded successfully!")
    
    def _load_model(self, model_path, model_type):
        print(f"Loading model from {model_path}")
        
        model = get_model(model_type=model_type)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_psnr' in checkpoint:
                print(f"Model PSNR: {checkpoint['best_psnr']:.2f}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    def _postprocess_output(self, output_tensor, original_size):
        output = output_tensor.squeeze(0)
        output = denormalize_tensor(output)
        output = torch.clamp(output, 0, 1)
        
        # Convert to numpy
        output_np = output.cpu().numpy().transpose(1, 2, 0)
        
        # Resize to original size
        if output_np.shape[:2] != original_size:
            output_np = cv2.resize(output_np, (original_size[1], original_size[0]))
        
        return (output_np * 255).astype(np.uint8)
    
    def deblur_image(self, input_path, output_path=None, save_comparison=False):
        print(f"Processing {input_path}")
        
        # Process image
        input_tensor, original_size = self._preprocess_image(input_path)
        
        start_time = time.time()
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        deblurred_image = self._postprocess_output(output_tensor, original_size)
        print(f"Inference time: {inference_time:.3f}s")
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save deblurred image
            deblurred_bgr = cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), deblurred_bgr)
            print(f"Saved to {output_path}")
            
            # Save comparison if requested
            if save_comparison:
                self._save_comparison(input_path, deblurred_image, output_path)
        
        return deblurred_image
    
    def _save_comparison(self, input_path, deblurred_image, output_path):
        # Load original
        original = cv2.imread(str(input_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if original.shape[:2] != deblurred_image.shape[:2]:
            deblurred_resized = cv2.resize(deblurred_image, (original.shape[1], original.shape[0]))
        else:
            deblurred_resized = deblurred_image
        
        # Create comparison
        comparison = np.hstack([original, deblurred_resized])
        
        # Save with matplotlib for better quality
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(comparison)
        ax.axis('off')
        
        # Add labels
        h, w = original.shape[:2]
        ax.text(w//2, 30, 'Blurred', ha='center', color='white', fontsize=16, weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(w + w//2, 30, 'Deblurred', ha='center', color='white', fontsize=16, weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        comparison_path = output_path.parent / f"{output_path.stem}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison saved to {comparison_path}")
    
    def deblur_batch(self, input_dir, output_dir, save_comparisons=False):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        for image_file in tqdm(image_files):
            output_path = output_dir / image_file.name
            try:
                self.deblur_image(image_file, output_path, save_comparisons)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"Batch processing completed! Results in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Image Deblurring Inference')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_path', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'deblurnet'])
    parser.add_argument('--batch_mode', action='store_true', help='Process directory of images')
    parser.add_argument('--save_comparison', action='store_true', help='Save before/after comparison')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Inference Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Initialize deblurrer
    deblurrer = ImageDeblurrer(args.model_path, args.model_type, args.device)
    
    try:
        if args.batch_mode:
            deblurrer.deblur_batch(args.input_path, args.output_path, args.save_comparison)
        else:
            deblurrer.deblur_image(args.input_path, args.output_path, args.save_comparison)
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()