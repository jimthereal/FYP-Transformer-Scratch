"""
Stable Transformer LipNet - Simplified for reliable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
import pickle
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=75):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class StableLipNet(nn.Module):
    """Simplified, stable transformer model"""
    def __init__(self, vocab_size=28, d_model=256):
        super(StableLipNet, self).__init__()
        
        self.d_model = d_model
        
        # Simpler 3D CNN backbone
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv3d(1, 32, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            # Conv Block 2
            nn.Conv3d(32, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            # Conv Block 3
            nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling to fixed size
            nn.AdaptiveAvgPool3d((75, 4, 2))
        )
        
        # Feature projection with layer norm
        self.feature_size = 128 * 4 * 2
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder (fewer layers for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights carefully
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Careful weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [B, 1, T, H, W]
        batch_size = x.size(0)
        
        # 3D CNN feature extraction
        features = self.backbone(x)  # [B, 128, 75, 4, 2]
        
        # Reshape for sequence processing
        B, C, T, H, W = features.size()
        features = features.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        features = features.reshape(B, T, -1)  # [B, T, C*H*W]
        
        # Project to model dimension
        features = self.feature_proj(features)  # [B, T, d_model]
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Transformer encoding
        encoded = self.transformer(features)  # [B, T, d_model]
        
        # Output projection
        output = self.output_proj(encoded)  # [B, T, vocab_size]
        
        return output

class GRIDDataset(Dataset):
    """Simple dataset loader"""
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.sample_files = sorted(list(self.data_path.glob("*.pkl")))
        print(f"Found {len(self.sample_files)} samples in {data_path}")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        with open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)
        
        return {
            'video': sample['video'].float(),
            'sentence': sample['sentence'],
            'speaker_id': sample.get('speaker_id', 'unknown')
        }

def create_char_mappings():
    """Create character mappings"""
    chars = [' '] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['<BLANK>']
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def text_to_sequence(text, char_to_idx):
    """Convert text to sequence of indices"""
    return [char_to_idx.get(char.upper(), 0) for char in text]

def ctc_decode(output_probs, blank_idx=27):
    """Simple CTC decoding"""
    # Get most likely character at each time step
    predictions = torch.argmax(output_probs, dim=-1)
    
    # Collapse repeated characters and remove blanks
    decoded = []
    prev_char = blank_idx
    
    for char in predictions:
        if char != blank_idx and char != prev_char:
            decoded.append(char.item())
        prev_char = char
    
    return decoded

def calculate_wer(pred_text, target_text):
    """Calculate Word Error Rate"""
    pred_words = pred_text.strip().split()
    target_words = target_text.strip().split()
    
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    # Simple edit distance
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

def train_stable_model():
    """Train the stable model"""
    print("TRAINING STABLE TRANSFORMER LIPNET")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    train_path = "data/processed_grid_only/train"
    test_path = "data/processed_grid_only/test"
    
    # Character mappings
    char_to_idx, idx_to_char = create_char_mappings()
    vocab_size = len(char_to_idx)
    blank_idx = vocab_size - 1
    
    # Create datasets
    train_dataset = GRIDDataset(train_path)
    test_dataset = GRIDDataset(test_path)
    
    # DataLoaders with smaller batch size for stability
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = StableLipNet(vocab_size=vocab_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Loss function
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    print("\nStarting training...")
    best_wer = 1.0
    
    for epoch in range(50):
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50")
        
        for batch_idx, batch in enumerate(progress_bar):
            videos = batch['video'].to(device)
            sentences = batch['sentence']
            
            # Debug first batch
            if epoch == 0 and batch_idx == 0:
                print(f"\nVideo shape: {videos.shape}")
                print(f"Expected: [batch_size, 1, 75, 128, 64]")
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(videos)  # [B, T, vocab_size]
            
            # Prepare for CTC loss
            output_log = F.log_softmax(output, dim=2)
            output_log = output_log.permute(1, 0, 2)  # [T, B, vocab_size]
            
            # Calculate CTC loss
            input_lengths = torch.full((len(sentences),), output_log.size(0), dtype=torch.long)
            target_lengths = []
            targets = []
            
            for sentence in sentences:
                target_seq = text_to_sequence(sentence, char_to_idx)
                targets.extend(target_seq)
                target_lengths.append(len(target_seq))
            
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            
            loss = ctc_loss(output_log, targets, input_lengths, target_lengths)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\nNaN loss detected at batch {batch_idx}")
                print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Decode predictions for this batch
            with torch.no_grad():
                output_probs = torch.softmax(output, dim=2)
                
                for i in range(len(sentences)):
                    # Decode
                    decoded = ctc_decode(output_probs[i], blank_idx)
                    pred_text = ''.join([idx_to_char.get(idx, '?') for idx in decoded])
                    
                    train_predictions.append(pred_text)
                    train_targets.append(sentences[i])
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        
        # Testing phase
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                videos = batch['video'].to(device)
                sentences = batch['sentence']
                
                output = model(videos)
                output_probs = torch.softmax(output, dim=2)
                
                for i in range(len(sentences)):
                    decoded = ctc_decode(output_probs[i], blank_idx)
                    pred_text = ''.join([idx_to_char.get(idx, '?') for idx in decoded])
                    
                    test_predictions.append(pred_text)
                    test_targets.append(sentences[i])
        
        # Calculate WER
        train_wer = np.mean([calculate_wer(p, t) for p, t in zip(train_predictions, train_targets)])
        test_wer = np.mean([calculate_wer(p, t) for p, t in zip(test_predictions, test_targets)])
        
        # Update scheduler
        scheduler.step(avg_train_loss)
        
        # Print results
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, WER: {train_wer:.4f}")
        print(f"  Test WER: {test_wer:.4f}")
        
        # Show sample predictions
        if epoch % 5 == 0 or test_wer < best_wer:
            print("\nSample predictions:")
            indices = np.random.choice(len(test_predictions), min(3, len(test_predictions)), replace=False)
            for idx in indices:
                print(f"  Target: '{test_targets[idx]}'")
                print(f"  Pred:   '{test_predictions[idx]}'")
        
        # Save best model
        if test_wer < best_wer:
            best_wer = test_wer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wer': best_wer,
                'vocab_size': vocab_size,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, 'best_stable_transformer.pth')
            print(f"  Saved new best model (WER: {best_wer:.4f})")
        
        print("-" * 70)
    
    print(f"\nTraining complete! Best WER: {best_wer:.4f}")

if __name__ == "__main__":
    train_stable_model()