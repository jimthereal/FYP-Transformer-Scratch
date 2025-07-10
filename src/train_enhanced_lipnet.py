# src/train_enhanced_lipnet.py
# Enhanced LipNet Training Script for FYP
# Integrates with your existing architecture and 458 processed samples

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import pickle
import os
import sys
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lipnet_base import LipNet
from src.language_corrector import LipReadingOptimizedCorrector

class EnhancedGRIDDataset(Dataset):
    """
    Enhanced dataset loader for your processed GRID samples
    Handles the 458 processed .pkl files
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Character mappings (matching your system)
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # Load all available samples
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self):
        """Load all processed sample files"""
        samples = []
        sample_dir = os.path.join(self.data_dir, 'processed', 'train')
        
        if not os.path.exists(sample_dir):
            raise ValueError(f"Sample directory not found: {sample_dir}")
        
        # Find all .pkl files
        pkl_files = [f for f in os.listdir(sample_dir) if f.startswith('sample_') and f.endswith('.pkl')]
        pkl_files.sort()  # Ensure consistent ordering
        
        print(f"Found {len(pkl_files)} sample files")
        
        for pkl_file in tqdm(pkl_files, desc="Loading samples"):
            try:
                sample_path = os.path.join(sample_dir, pkl_file)
                with open(sample_path, 'rb') as f:
                    sample_data = pickle.load(f)
                    
                # Validate sample structure
                if self._validate_sample(sample_data):
                    samples.append(sample_data)
                else:
                    print(f"Invalid sample: {pkl_file}")
                    
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue
        
        return samples
    
    def _validate_sample(self, sample):
        """Validate sample data structure"""
        required_keys = ['video', 'sentence']
        
        for key in required_keys:
            if key not in sample:
                return False
        
        # Check tensor shape [1, 75, 112, 112]
        video_tensor = sample['video']
        if not isinstance(video_tensor, torch.Tensor):
            return False
            
        if video_tensor.shape != torch.Size([1, 75, 112, 112]):
            return False
            
        return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get video tensor and sentence
        video_tensor = sample['video']  # [1, 75, 112, 112]
        sentence = sample['sentence']
        
        # Convert to proper format for training
        # Remove batch dimension: [75, 112, 112]
        video_tensor = video_tensor.squeeze(0)  
        
        # Convert grayscale to RGB format expected by model
        if len(video_tensor.shape) == 3:  # [75, 112, 112]
            # Stack to create RGB channels: [75, 112, 112] -> [3, 75, 112, 112]
            video_tensor = video_tensor.unsqueeze(0).expand(3, -1, -1, -1)
            
            # Resize to model expected dimensions: [3, 75, 64, 128]
            video_tensor = F.interpolate(
                video_tensor.unsqueeze(0), 
                size=(75, 64, 128), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Ensure video tensor is float32 and normalized
        video_tensor = video_tensor.float()
        
        # Check if tensor needs normalization (values > 1.0 means it's in 0-255 range)
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        
        # If tensor is all zeros or very small, it might need different handling
        if video_tensor.max() < 0.01:
            # The tensor might already be normalized but very dark
            # Add small epsilon to avoid all-zero tensors
            video_tensor = video_tensor + 1e-6
        
        # Encode sentence to indices
        sentence_indices = []
        for char in sentence.upper():
            if char in self.char_to_idx:
                sentence_indices.append(self.char_to_idx[char])
            else:
                sentence_indices.append(self.char_to_idx[' '])  # Use space for unknown chars
        
        return {
            'video': video_tensor,
            'sentence': sentence,
            'sentence_indices': torch.tensor(sentence_indices, dtype=torch.long),
            'video_length': torch.tensor(75, dtype=torch.long),
            'sentence_length': torch.tensor(len(sentence_indices), dtype=torch.long)
        }


class EnhancedLipNetTrainer:
    """
    Enhanced trainer for your LipNet system
    Integrates with language corrector and provides comprehensive monitoring
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing Enhanced LipNet Trainer on {self.device}")
        
        # Initialize components
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_criterion()
        self.corrector = LipReadingOptimizedCorrector()
        
        # Training tracking
        self.train_history = defaultdict(list)
        self.best_wer = float('inf')
        self.start_epoch = 0
        
        # Create output directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        print("Enhanced LipNet Trainer ready!")
    
    def _setup_model(self):
        """Setup and load model"""
        model = LipNet()
        
        # Load pre-trained weights if specified
        if self.config.get('pretrained_path') and os.path.exists(self.config['pretrained_path']):
            print(f"Loading pre-trained weights: {self.config['pretrained_path']}")
            model.load_state_dict(torch.load(self.config['pretrained_path'], map_location=self.device))
        else:
            print("No pre-trained weights found, starting from scratch")
        
        model.to(self.device)
        return model
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
    
    def _setup_criterion(self):
        """Setup CTC loss"""
        return nn.CTCLoss(blank=len('ABCDEFGHIJKLMNOPQRSTUVWXYZ '), reduction='mean', zero_infinity=True)
    
    def setup_datasets(self, data_dir):
        """Setup datasets with train/val split"""
        print("Setting up datasets...")
        
        # Load full dataset
        full_dataset = EnhancedGRIDDataset(data_dir, split='train')
        
        # Split into train/val (80/20)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset split: {train_size} train, {val_size} validation")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Data loaders ready: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            videos = batch['video'].to(self.device)  # [B, 3, 75, 64, 128]
            sentence_indices = batch['sentence_indices']  # [B, seq_len]
            video_lengths = batch['video_length']  # [B]
            sentence_lengths = batch['sentence_length']  # [B]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model outputs [B, T, C], but CTC expects [T, B, C]
            outputs = self.model(videos)  # [B, T, C] = [B, 75, 28]
            outputs = outputs.transpose(0, 1)  # [T, B, C] = [75, B, 28]
            
            # Prepare for CTC loss
            log_probs = F.log_softmax(outputs, dim=2)
            
            # Input lengths should be the sequence length (75)
            input_lengths = torch.full((videos.size(0),), outputs.size(0), dtype=torch.long)
            
            # Flatten sentence indices for CTC - handle variable length sequences
            targets = []
            target_lengths = []
            
            for i in range(len(sentence_indices)):
                seq_len = sentence_lengths[i].item()
                target_seq = sentence_indices[i][:seq_len]
                targets.extend(target_seq.tolist())
                target_lengths.append(seq_len)
            
            targets = torch.tensor(targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            
            # Calculate loss
            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_samples += videos.size(0)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                videos = batch['video'].to(self.device)
                sentence_indices = batch['sentence_indices']
                video_lengths = batch['video_length']
                sentence_lengths = batch['sentence_length']
                sentences = batch['sentence']
                
                # Forward pass
                outputs = self.model(videos)  # [B, T, C]
                outputs = outputs.transpose(0, 1)  # [T, B, C]
                
                # Calculate loss (same as training)
                log_probs = F.log_softmax(outputs, dim=2)
                input_lengths = torch.full((videos.size(0),), outputs.size(0), dtype=torch.long)
                
                targets = []
                target_lengths = []
                for i in range(len(sentence_indices)):
                    seq_len = sentence_lengths[i].item()
                    target_seq = sentence_indices[i][:seq_len]
                    targets.extend(target_seq.tolist())
                    target_lengths.append(seq_len)
                
                targets = torch.tensor(targets, dtype=torch.long)
                target_lengths = torch.tensor(target_lengths, dtype=torch.long)
                
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Decode predictions
                for i in range(videos.size(0)):
                    pred_str = self._ctc_decode(outputs[:, i, :])  # outputs is now [T, B, C]
                    predictions.append(pred_str)
                    ground_truths.append(sentences[i])
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        wer, cer = self._calculate_metrics(predictions, ground_truths)
        
        # Test language corrector
        corrected_predictions = []
        for pred in predictions[:10]:  # Test on first 10 predictions
            correction_result = self.corrector.correct_prediction(pred)
            corrected_predictions.append(correction_result['corrected'])
        
        print(f"\nValidation Results (Epoch {epoch+1}):")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   WER: {wer:.4f}")
        print(f"   CER: {cer:.4f}")
        print(f"\nSample Predictions:")
        for i in range(min(3, len(predictions))):
            print(f"   GT: '{ground_truths[i]}'")
            print(f"   Raw: '{predictions[i]}'")
            if i < len(corrected_predictions):
                print(f"   Corrected: '{corrected_predictions[i]}'")
            print()
        
        return avg_loss, wer, cer
    
    def _ctc_decode(self, outputs):
        """CTC decode outputs to string"""
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        arg_maxes = torch.argmax(outputs, dim=1)
        
        decoded = []
        prev_char = None
        
        for char_idx in arg_maxes:
            char_idx = char_idx.item()
            if char_idx < len(chars):
                char = chars[char_idx]
                if char != prev_char:
                    decoded.append(char)
                prev_char = char
            else:
                prev_char = None
        
        return ''.join(decoded).strip()
    
    def _calculate_metrics(self, predictions, ground_truths):
        """Calculate WER and CER"""
        total_wer = 0
        total_cer = 0
        
        for pred, gt in zip(predictions, ground_truths):
            wer = self._word_error_rate(pred, gt)
            cer = self._character_error_rate(pred, gt)
            total_wer += wer
            total_cer += cer
        
        return total_wer / len(predictions), total_cer / len(predictions)
    
    def _word_error_rate(self, pred, gt):
        """Calculate word error rate"""
        pred_words = pred.strip().split()
        gt_words = gt.strip().split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        return self._edit_distance(pred_words, gt_words) / len(gt_words)
    
    def _character_error_rate(self, pred, gt):
        """Calculate character error rate"""
        if len(gt) == 0:
            return 1.0 if len(pred) > 0 else 0.0
        
        return self._edit_distance(list(pred), list(gt)) / len(gt)
    
    def _edit_distance(self, s1, s2):
        """Calculate edit distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def save_checkpoint(self, epoch, loss, wer, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'wer': wer,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f'enhanced_lipnet_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'enhanced_lipnet_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model: WER {wer:.4f}")
    
    def train(self, data_dir, num_epochs):
        """Main training loop"""
        print("Starting Enhanced LipNet Training!")
        print("="*70)
        
        # Setup datasets
        self.setup_datasets(data_dir)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, wer, cer = self.validate_epoch(epoch)
            
            # Save training history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['wer'].append(wer)
            self.train_history['cer'].append(cer)
            
            # Save checkpoint
            is_best = wer < self.best_wer
            if is_best:
                self.best_wer = wer
            
            self.save_checkpoint(epoch, val_loss, wer, is_best)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   WER: {wer:.4f} {'(BEST!)' if is_best else ''}")
            print(f"   CER: {cer:.4f}")
        
        print("\nTraining completed!")
        print(f"Best WER achieved: {self.best_wer:.4f}")
        
        # Save final training history
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(dict(self.train_history), f, indent=2)


def main():
    """Main training function"""
    print("Enhanced LipNet Training Script")
    print("="*70)
    
    # Training configuration - optimized for RTX 3060 6GB
    config = {
        'batch_size': 1,  # Conservative for 6GB GPU
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'num_epochs': 10,  # Start with 10 epochs for initial test
        'checkpoint_dir': 'checkpoints/enhanced_lipnet',
        'log_dir': 'logs/enhanced_lipnet',
        'pretrained_path': 'models/pretrained/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'
    }
    
    # Data directory
    data_dir = 'data/GRID'
    
    # Check if data exists
    if not os.path.exists(os.path.join(data_dir, 'processed', 'train')):
        print(f"Processed data not found at: {data_dir}/processed/train")
        print("Please ensure your preprocessed data is available!")
        return
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Initialize trainer
    trainer = EnhancedLipNetTrainer(config)
    
    # Start training
    trainer.train(data_dir, config['num_epochs'])
    
    print("\nTraining completed! Next steps:")
    print("1. Check logs/enhanced_lipnet/ for training metrics")
    print("2. Best model saved in checkpoints/enhanced_lipnet/")
    print("3. Test the trained model on new videos")
    print("4. Consider adding more GRID speakers for better performance")


if __name__ == "__main__":
    main()