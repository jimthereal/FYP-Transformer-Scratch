"""
Production Transformer-Based LipNet
Designed for 1-week deadline with strong fundamentals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
from tqdm import tqdm
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=75):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ProductionLipNet(nn.Module):
    """
    Production-ready Transformer-based LipNet
    - Proven architecture for generalization
    - Efficient training with limited data
    - Real-world deployment ready
    """
    def __init__(self, vocab_size=28, d_model=512):
        super(ProductionLipNet, self).__init__()
        
        self.d_model = d_model
        
        # Efficient 3D CNN backbone
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv3d(1, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            
            # Second conv block  
            nn.Conv3d(64, 128, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            
            # Third conv block
            nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling to fixed size
            nn.AdaptiveAvgPool3d((75, 4, 2))  # [B, 256, 75, 4, 2]
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(256 * 4 * 2, d_model),  # 2048 -> 512
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=75)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=6,
            enable_nested_tensor=False
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, vocab_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize transformer weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        Args:
            x: [B, 1, T, H, W] video tensor
            mask: [B, T] attention mask (optional)
        """
        batch_size = x.size(0)
        
        # 3D CNN feature extraction
        features = self.backbone(x)  # [B, 256, 75, 4, 2]
        
        # Reshape for sequence processing
        B, C, T, H, W = features.size()
        features = features.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        features = features.reshape(B, T, -1)  # [B, T, C*H*W]
        
        # Project to model dimension
        features = self.feature_proj(features)  # [B, T, d_model]
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Create attention mask if not provided
        if mask is None:
            mask = torch.zeros(batch_size, T, dtype=torch.bool, device=x.device)
        
        # Transformer encoding
        encoded = self.transformer(features, src_key_padding_mask=mask)  # [B, T, d_model]
        
        # Output projection
        output = self.output_proj(encoded)  # [B, T, vocab_size]
        
        return output

class AdvancedDataset(Dataset):
    """Enhanced dataset with better augmentation"""
    def __init__(self, data_path, max_samples=100, augment=True):
        self.data_path = data_path
        self.augment = augment
        
        # Load sample files (reuse your existing logic)
        if not os.path.exists(data_path):
            print(f"ERROR: Data path does not exist: {data_path}")
            self.sample_files = []
            return
        
        all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pkl')])        
        self.sample_files = all_files[:max_samples]
        print(f"Loading {len(self.sample_files)} samples from multi-dataset")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        sample_file = self.sample_files[idx]
        
        with open(os.path.join(self.data_path, sample_file), 'rb') as f:
            sample = pickle.load(f)
        
        video = sample['video']  # [1, T, H, W]
        sentence = sample['sentence']
        
        # Apply augmentation during training
        if self.augment:
            video = self.apply_augmentation(video)
        
        return {
            'video': video.float(),
            'sentence': sentence,
            'video_length': torch.tensor(video.shape[1], dtype=torch.long)
        }
    
    def apply_augmentation(self, video):
        """Advanced augmentation for better generalization"""
        # Random brightness adjustment
        if torch.rand(1) > 0.6:
            brightness = 0.8 + torch.rand(1) * 0.4  # 0.8-1.2x
            video = video * brightness
        
        # Gaussian noise
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(video) * 0.03
            video = video + noise
        
        # Random temporal dropout (simulate missing frames)
        if torch.rand(1) > 0.8:
            dropout_frames = torch.randint(1, 4, (1,)).item()
            dropout_indices = torch.randperm(video.size(1))[:dropout_frames]
            for idx in dropout_indices:
                if idx > 0:
                    video[:, idx] = video[:, idx-1]  # Copy previous frame
        
        return torch.clamp(video, 0, 1)

def create_char_mappings():
    """Create character mappings (reuse your existing function)"""
    chars = [' '] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['<BLANK>']
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def text_to_sequence(text, char_to_idx):
    """Convert text to sequence (reuse your existing function)"""
    return [char_to_idx.get(char.upper(), char_to_idx[' ']) for char in text]

def sequence_to_text(sequence, idx_to_char):
    """Convert sequence to text (reuse your existing function)"""
    return ''.join([idx_to_char.get(idx, '?') for idx in sequence])

def decode_predictions(output_probs, blank_idx=27):
    """CTC decoding (reuse your existing function)"""
    batch_size, seq_len, vocab_size = output_probs.shape
    pred_sequences = []
    
    for b in range(batch_size):
        probs = output_probs[b]
        pred_sequence = torch.argmax(probs, dim=1).cpu().numpy()
        
        # CTC collapse
        collapsed = []
        prev_char = None
        
        for char_idx in pred_sequence:
            if char_idx != blank_idx and char_idx != prev_char:
                collapsed.append(char_idx)
            prev_char = char_idx
        
        pred_sequences.append(collapsed)
    
    return pred_sequences

def calculate_wer(predicted, target):
    """Calculate WER (reuse your existing function)"""
    pred_words = predicted.strip().split()
    target_words = target.strip().split()
    
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    # Edit distance calculation
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / len(target_words)

def train_production_model():
    """Train the production transformer model"""
    print("TRAINING PRODUCTION TRANSFORMER LIPNET")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Character mappings
    char_to_idx, idx_to_char = create_char_mappings()
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Use new multi-dataset path
    data_path = "data/processed_windows_safe/unified/train"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Multi-dataset not found at {data_path}")
        print("Please run windows_safe_processor.py first")
        return
    
    print(f"Using multi-dataset: {data_path}")
    
    if data_path is None:
        print("ERROR: No processed data found!")
        return
    
    # Create datasets with appropriate settings for large dataset
    train_dataset = AdvancedDataset(data_path, max_samples=3500, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Create model
    model = ProductionLipNet(vocab_size=vocab_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with different learning rates for different components
    backbone_params = list(model.backbone.parameters())
    transformer_params = list(model.transformer.parameters()) + list(model.pos_encoding.parameters())
    other_params = list(model.feature_proj.parameters()) + list(model.output_proj.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 0.0001, 'weight_decay': 0.01},
        {'params': transformer_params, 'lr': 0.0005, 'weight_decay': 0.005},
        {'params': other_params, 'lr': 0.001, 'weight_decay': 0.01}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=0.00001
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=vocab_size-1, reduction='mean', zero_infinity=True)
    
    print(f"Training on {len(train_dataset)} samples")
    print("Starting training...")
    print()
    
    # Training loop
    model.train()
    best_wer = 1.0
    patience = 0
    
    for epoch in range(50):  # Reduced epochs for large dataset
        total_loss = 0
        predictions = []
        targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100", leave=False)
        
        for batch in progress_bar:
            video = batch['video'].to(device)
            sentences = batch['sentence']
            video_lengths = batch['video_length'].to(device)
            
            batch_size = video.size(0)
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                output = model(video)  # [B, T, vocab_size]
                
                # Prepare for CTC
                output = output.permute(1, 0, 2)  # [T, B, vocab_size]
                output = torch.log_softmax(output, dim=2)
                
                # Calculate loss
                batch_loss = 0
                for b in range(batch_size):
                    sentence = sentences[b]
                    target_seq = text_to_sequence(sentence, char_to_idx)
                    target_tensor = torch.tensor(target_seq, dtype=torch.long).to(device)
                    target_length = torch.tensor([len(target_seq)], dtype=torch.long).to(device)
                    video_length = video_lengths[b:b+1]
                    
                    loss = ctc_loss(output[:, b:b+1, :], target_tensor, video_length, target_length)
                    batch_loss += loss
                
                batch_loss = batch_loss / batch_size
            
            # Backward pass
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += batch_loss.item()
            
            # Decode predictions
            with torch.no_grad():
                output_probs = torch.softmax(output.permute(1, 0, 2), dim=2)
                pred_sequences = decode_predictions(output_probs)
                
                for b in range(batch_size):
                    pred_text = sequence_to_text(pred_sequences[b], idx_to_char)
                    predictions.append(pred_text)
                    targets.append(sentences[b])
            
            progress_bar.set_postfix({'Loss': f"{batch_loss.item():.4f}"})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        wer_scores = [calculate_wer(pred, target) for pred, target in zip(predictions, targets)]
        avg_wer = np.mean(wer_scores)
        
        # Save best model
        if avg_wer < best_wer:
            best_wer = avg_wer
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'epoch': epoch,
                'best_wer': best_wer
            }, 'best_transformer_model.pth')
        else:
            patience += 1
        
        # Print progress
        if epoch % 5 == 0 or avg_wer < 0.4:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, WER={avg_wer:.4f} (Best: {best_wer:.4f})")
            
            # Show best predictions
            best_indices = np.argsort(wer_scores)[:2]
            for idx in best_indices:
                wer = wer_scores[idx]
                print(f"  '{targets[idx]}' -> '{predictions[idx]}' (WER: {wer:.3f})")
            print()
        
        # Early stopping
        if avg_wer == 0.0:
            print(f"Perfect accuracy at epoch {epoch+1}!")
            break
        
        if patience >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("=" * 60)
    print("TRANSFORMER TRAINING COMPLETE!")
    print(f"Best WER: {best_wer:.4f}")
    print(f"Model saved: best_transformer_model.pth")
    
    return best_wer

if __name__ == "__main__":
    train_production_model()