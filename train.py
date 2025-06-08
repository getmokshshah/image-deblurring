#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import get_model
from utils.dataset import get_dataloaders
from utils.metrics import CombinedLoss, calculate_psnr, calculate_ssim, denormalize_tensor


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = get_model(model_type=args.model_type).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize datasets
        self.train_loader, self.val_loader = get_dataloaders(
            args.train_path, args.val_path, args.batch_size, 
            args.image_size, args.num_workers
        )
        
        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        # Tracking
        self.writer = SummaryWriter(log_dir=self.save_dir / 'logs')
        self.best_psnr = 0.0
        
        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        psnr_values = []
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        for batch in pbar:
            blurred = batch['blurred'].to(self.device)
            sharp = batch['sharp'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(blurred)
            loss = self.criterion(output, sharp)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate PSNR for monitoring
            with torch.no_grad():
                pred_denorm = denormalize_tensor(output[0])
                sharp_denorm = denormalize_tensor(sharp[0])
                pred_denorm = torch.clamp(pred_denorm, 0, 1)
                sharp_denorm = torch.clamp(sharp_denorm, 0, 1)
                psnr_val = calculate_psnr(pred_denorm, sharp_denorm)
                psnr_values.append(psnr_val)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader), np.mean(psnr_values)
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        psnr_values = []
        ssim_values = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Val Epoch {epoch}'):
                blurred = batch['blurred'].to(self.device)
                sharp = batch['sharp'].to(self.device)
                
                output = self.model(blurred)
                loss = self.criterion(output, sharp)
                total_loss += loss.item()
                
                # Calculate metrics
                for i in range(output.size(0)):
                    pred_denorm = denormalize_tensor(output[i])
                    sharp_denorm = denormalize_tensor(sharp[i])
                    pred_denorm = torch.clamp(pred_denorm, 0, 1)
                    sharp_denorm = torch.clamp(sharp_denorm, 0, 1)
                    
                    psnr_val = calculate_psnr(pred_denorm, sharp_denorm)
                    ssim_val = calculate_ssim(pred_denorm, sharp_denorm)
                    
                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)
        
        return (total_loss / len(self.val_loader), 
                np.mean(psnr_values), 
                np.mean(ssim_values))
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            print(f"New best model saved! PSNR: {self.best_psnr:.2f}")
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.args.epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss, train_psnr = self.train_epoch(epoch)
            val_loss, val_psnr, val_ssim = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best model
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log metrics
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/PSNR', train_psnr, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/PSNR', val_psnr, epoch)
            self.writer.add_scalar('Val/SSIM', val_ssim, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{self.args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, PSNR={train_psnr:.2f}")
            print(f"  Val:   Loss={val_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")
            print(f"  LR: {lr:.6f}, Best PSNR: {self.best_psnr:.2f}")
        
        print(f"\nTraining completed! Best PSNR: {self.best_psnr:.2f}")
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Image Deblurring Model')
    
    # Required arguments
    parser.add_argument('--train_path', type=str, required=True, help='Training data path')
    parser.add_argument('--val_path', type=str, required=True, help='Validation data path')
    
    # Model and training
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'deblurnet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./models')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Training Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()