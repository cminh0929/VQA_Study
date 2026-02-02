"""
Trainer for VQA Models
Handles training loop, validation, and checkpointing
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import Dict

from utils import MetricsTracker


class VQATrainer:
    """Trainer for VQA models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        answer_vocab,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        teacher_forcing_ratio: float = 0.5,
        gradient_clip: float = 5.0
    ):
        """
        Args:
            model: VQA model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            answer_vocab: Answer vocabulary for decoding
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            teacher_forcing_ratio: Teacher forcing probability
            gradient_clip: Gradient clipping value
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.answer_vocab = answer_vocab
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.gradient_clip = gradient_clip
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_bleu1': [],
            'val_f1': []
        }
        
        self.best_val_accuracy = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch in pbar:
            # Move to device
            images = batch['images'].to(self.device)
            questions = batch['questions'].to(self.device)
            answers = batch['answers'].to(self.device)
            q_lengths = batch['question_lengths']
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(
                images, questions, q_lengths, answers,
                teacher_forcing_ratio=self.teacher_forcing_ratio
            )
            
            # Compute loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = answers.view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        tracker = MetricsTracker()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                images = batch['images'].to(self.device)
                questions = batch['questions'].to(self.device)
                answers = batch['answers'].to(self.device)
                q_lengths = batch['question_lengths']
                
                # Forward pass
                outputs, _ = self.model(images, questions, q_lengths, answers, teacher_forcing_ratio=0.0)
                
                # Compute loss
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = answers.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                # Get predictions
                predictions = outputs.argmax(dim=-1)  # (B, max_len)
                
                # Decode predictions and ground truths
                pred_texts = [self.answer_vocab.decode(p.cpu().tolist(), skip_special_tokens=True) 
                             for p in predictions]
                gt_texts = batch['answer_texts']
                categories = batch['question_types']
                
                # Update tracker
                tracker.update(pred_texts, gt_texts, categories, loss.item())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        metrics = tracker.compute()
        return metrics
    
    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_bleu1'].append(val_metrics['bleu1'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val BLEU-1: {val_metrics['bleu1']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            
            # Per-category metrics
            if 'per_category' in val_metrics:
                print(f"\n  Per-category accuracy:")
                for cat, cat_metrics in val_metrics['per_category'].items():
                    print(f"    {cat}: {cat_metrics['accuracy']:.4f} ({cat_metrics['count']} samples)")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  ✓ Saved best model (accuracy: {self.best_val_accuracy:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', val_metrics)
            
            print()
        
        print(f"{'='*80}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"{'='*80}\n")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, filename: str, metrics: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_history(self):
        """Save training history to JSON"""
        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {path}")


# Example usage
if __name__ == "__main__":
    print("Trainer module - use in train.py")
