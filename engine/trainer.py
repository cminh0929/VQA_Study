"""
Trainer Utilities cho VQA Model
Training loop, validation, metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer class cho VQA model
    Quản lý training loop, validation, checkpointing
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 scheduler: Optional[_LRScheduler] = None,
                 checkpoint_dir: str = "checkpoints",
                 log_interval: int = 10,
                 save_best_only: bool = True):
        """
        Args:
            model: VQA model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            optimizer: Optimizer
            criterion: Loss function
            device: Device (cuda/cpu)
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory để lưu checkpoints
            log_interval: Log mỗi N batches
            save_best_only: Chỉ lưu model tốt nhất
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        
        # Tạo checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train một epoch
        
        Returns:
            Dict chứa metrics: loss, accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, questions, answers, _) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            questions = questions.to(self.device)
            answers = answers.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, questions)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, answers)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == answers).sum().item()
            total += answers.size(0)
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        # Epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dict chứa metrics: loss, accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for images, questions, answers, _ in pbar:
                # Move to device
                images = images.to(self.device)
                questions = questions.to(self.device)
                answers = answers.to(self.device)
                
                # Forward pass
                outputs = self.model(images, questions)
                logits = outputs['logits']
                
                # Compute loss
                loss = self.criterion(logits, answers)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == answers).sum().item()
                total += answers.size(0)
        
        # Validation metrics
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Train model cho num_epochs
        
        Args:
            num_epochs: Số epochs
            early_stopping_patience: Số epochs không cải thiện trước khi dừng
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                print(f"  ✓ New best validation accuracy: {self.best_val_acc:.2f}%")
                
                if self.save_best_only:
                    self.save_checkpoint(is_best=True)
            else:
                patience_counter += 1
                
                if not self.save_best_only:
                    self.save_checkpoint(is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
            
            print("=" * 60)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Lưu checkpoint
        
        Args:
            is_best: Có phải best model không
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, filepath)
            print(f"  Saved best model to {filepath}")
        else:
            filepath = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.current_epoch + 1}.pth')
            torch.save(checkpoint, filepath)
            print(f"  Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load checkpoint
        
        Args:
            filepath: Đường dẫn checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Tính accuracy
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        
    Returns:
        Accuracy (%)
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == "__main__":
    print("Trainer utilities module")
    print("Use this in train.py script")
