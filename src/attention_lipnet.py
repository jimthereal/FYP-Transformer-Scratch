"""
Enhanced LipNet with Multiple Attention Mechanisms
- Spatial Attention: Focus on important lip regions
- Temporal Attention: Weight important frames
- Multi-Head Attention: Transformer-like processing
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

class SpatialAttention(nn.Module):
    """Spatial attention to focus on important lip regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        attention = self.conv(x)  # [B, 1, H, W]
        attention = self.sigmoid(attention)
        return x * attention

class TemporalAttention(nn.Module):
    """Temporal attention to weight important frames"""
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x):
        # x shape: [B, T, F]
        weights = self.attention(x)  # [B, T, 1]
        weights = F.softmax(weights, dim=1)
        return x * weights

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attended)
        
        return output + x  # Residual connection

class EnhancedLipNet(nn.Module):
    """Enhanced LipNet with attention mechanisms"""
    def __init__(self, vocab_size=28):
        super(EnhancedLipNet, self).__init__()
        
        # 3D CNN with spatial attention
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.spatial_att_1 = SpatialAttention(32)
        self.pool_1 = nn.MaxPool3d((1, 2, 2))
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.spatial_att_2 = SpatialAttention(64)
        self.pool_2 = nn.MaxPool3d((1, 2, 2))
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
        )
        self.spatial_att_3 = SpatialAttention(96)
        self.pool_3 = nn.MaxPool3d((1, 2, 2))
        
        # Feature dimensions: 96 * 8 * 4 = 3072
        self.feature_size = 96 * 8 * 4
        
        # Feature projection
        self.feature_proj = nn.Linear(self.feature_size, 512)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(512)
        
        # Multi-head attention
        self.multihead_attention = MultiHeadAttention(512, n_heads=8)
        
        # Enhanced RNN
        self.rnn = nn.LSTM(512, 256, num_layers=2, batch_first=True, 
                          bidirectional=True, dropout=0.2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, vocab_size)
        )
        
    def forward(self, x):
        # x shape: [B, 1, T, H, W] = [B, 1, 75, 128, 64]
        batch_size, _, seq_len, _, _ = x.size()
        
        # 3D CNN with spatial attention
        x = self.conv3d_1(x)  # [B, 32, 75, 64, 32]
        
        # Apply spatial attention frame by frame
        x_attended = []
        for t in range(x.size(2)):
            frame = x[:, :, t, :, :]  # [B, 32, 64, 32]
            frame_att = self.spatial_att_1(frame)
            x_attended.append(frame_att)
        x = torch.stack(x_attended, dim=2)  # [B, 32, 75, 64, 32]
        x = self.pool_1(x)  # [B, 32, 75, 32, 16]
        
        x = self.conv3d_2(x)  # [B, 64, 75, 16, 8]
        
        # Apply spatial attention frame by frame
        x_attended = []
        for t in range(x.size(2)):
            frame = x[:, :, t, :, :]
            frame_att = self.spatial_att_2(frame)
            x_attended.append(frame_att)
        x = torch.stack(x_attended, dim=2)
        x = self.pool_2(x)  # [B, 64, 75, 16, 8]
        
        x = self.conv3d_3(x)  # [B, 96, 75, 8, 4]
        
        # Apply spatial attention frame by frame
        x_attended = []
        for t in range(x.size(2)):
            frame = x[:, :, t, :, :]
            frame_att = self.spatial_att_3(frame)
            x_attended.append(frame_att)
        x = torch.stack(x_attended, dim=2)
        x = self.pool_3(x)  # [B, 96, 75, 8, 4]
        
        # Reshape for sequence processing
        x = x.permute(0, 2, 1, 3, 4)  # [B, 75, 96, 8, 4]
        x = x.reshape(batch_size, seq_len, -1)  # [B, 75, 3072]
        
        # Project features
        x = self.feature_proj(x)  # [B, 75, 512]
        
        # Apply temporal attention
        x = self.temporal_attention(x)  # [B, 75, 512]
        
        # Apply multi-head attention
        x = self.multihead_attention(x)  # [B, 75, 512]
        
        # RNN processing
        x, _ = self.rnn(x)  # [B, 75, 512]
        
        # Classification
        x = self.classifier(x)  # [B, 75, vocab_size]
        
        return x

class RealDataset(Dataset):
    """Load real processed data with improved batching"""
    def __init__(self, data_path, max_samples=50):
        self.data_path = data_path
        self.samples = []
        
        # Load sample files
        all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pkl')])
        sample_files = [f for f in all_files if f.startswith('mediapipe_sample_')][:max_samples]
        
        print(f"Loading {len(sample_files)} REAL samples...")
        
        for sample_file in sample_files:
            with open(os.path.join(data_path, sample_file), 'rb') as f:
                sample = pickle.load(f)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Show sample info
        for i, sample in enumerate(self.samples[:5]):  # Show first 5
            video = sample['video']
            sentence = sample['sentence']
            print(f"Sample {i}: '{sentence}'")
            print(f"  Video shape: {video.shape}")
            print(f"  Video range: {video.min():.3f} - {video.max():.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video = sample['video']  # Shape: [1, 75, 128, 64]
        sentence = sample['sentence']
        
        return {
            'video': video.float(),
            'sentence': sentence,
            'video_length': torch.tensor(video.shape[1], dtype=torch.long)
        }

def text_to_sequence(text, char_to_idx):
    """Convert text to sequence of indices"""
    return [char_to_idx.get(char.upper(), char_to_idx[' ']) for char in text]

def create_char_mappings():
    """Create character to index mappings"""
    chars = [' '] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['<BLANK>']
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def sequence_to_text(sequence, idx_to_char):
    """Convert sequence of indices to text"""
    return ''.join([idx_to_char.get(idx, '?') for idx in sequence])

def calculate_wer(predicted, target):
    """Calculate Word Error Rate"""
    pred_words = predicted.split()
    target_words = target.split()
    
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    substitutions = sum(p != t for p, t in zip(pred_words, target_words))
    insertions = max(0, len(pred_words) - len(target_words))
    deletions = max(0, len(target_words) - len(pred_words))
    
    return (substitutions + insertions + deletions) / len(target_words)

def train_attention_model():
    """Train the enhanced attention model"""
    print("TRAINING ENHANCED ATTENTION MODEL")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Character mappings
    char_to_idx, idx_to_char = create_char_mappings()
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load real data
    data_path = "data/GRID/processed_mediapipe/train"
    if not os.path.exists(data_path):
        print(f"ERROR: Real data not found at {data_path}")
        print("Run simple_preprocessor.py first!")
        return
    
    # Check available samples
    available_samples = len([f for f in os.listdir(data_path) if f.startswith('mediapipe_sample_')])
    print(f"Available samples: {available_samples}")
    
    # Use as many samples as available (up to 50)
    max_samples = min(50, available_samples)
    dataset = RealDataset(data_path, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Enhanced model
    model = EnhancedLipNet(vocab_size=vocab_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with different learning rates for different parts
    cnn_params = list(model.conv3d_1.parameters()) + list(model.conv3d_2.parameters()) + list(model.conv3d_3.parameters())
    attention_params = list(model.spatial_att_1.parameters()) + list(model.spatial_att_2.parameters()) + list(model.spatial_att_3.parameters()) + list(model.temporal_attention.parameters()) + list(model.multihead_attention.parameters())
    other_params = list(model.feature_proj.parameters()) + list(model.rnn.parameters()) + list(model.classifier.parameters())
    
    optimizer = optim.Adam([
        {'params': cnn_params, 'lr': 0.0005},
        {'params': attention_params, 'lr': 0.001},
        {'params': other_params, 'lr': 0.001}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=vocab_size-1, reduction='mean', zero_infinity=True)
    
    print(f"Training on {len(dataset)} REAL samples")
    print("Enhanced model with attention mechanisms!")
    print()
    
    # Training loop
    model.train()
    best_wer = 1.0
    patience_counter = 0
    
    for epoch in range(200):  # More epochs for complex model
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_idx, batch in enumerate(dataloader):
            video = batch['video'].to(device)
            sentence = batch['sentence'][0]
            video_length = batch['video_length'].to(device)
            
            # Convert to sequence
            target_sequence = text_to_sequence(sentence, char_to_idx)
            target_tensor = torch.tensor(target_sequence, dtype=torch.long).to(device)
            target_length = torch.tensor([len(target_sequence)], dtype=torch.long).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(video)
            
            # CTC loss format
            output = output.permute(1, 0, 2)  # [T, B, C]
            output = torch.log_softmax(output, dim=2)
            
            loss = ctc_loss(output, target_tensor, video_length, target_length)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Decode prediction
            with torch.no_grad():
                output_probs = torch.softmax(output.permute(1, 0, 2), dim=2)
                pred_sequence = torch.argmax(output_probs, dim=2).squeeze().cpu().numpy()
                
                # CTC decoding
                pred_clean = []
                prev_char = None
                for char_idx in pred_sequence:
                    if char_idx != vocab_size - 1 and char_idx != prev_char:
                        pred_clean.append(char_idx)
                    prev_char = char_idx
                
                pred_text = sequence_to_text(pred_clean, idx_to_char)
                predictions.append(pred_text)
                targets.append(sentence)
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        wer_scores = [calculate_wer(pred, target) for pred, target in zip(predictions, targets)]
        avg_wer = np.mean(wer_scores)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        if avg_wer < best_wer:
            best_wer = avg_wer
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_attention_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 20 == 0 or epoch < 10 or avg_wer < 0.5:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, WER={avg_wer:.4f} (Best: {best_wer:.4f})")
            
            for i in range(min(3, len(predictions))):  # Show first 3
                wer = calculate_wer(predictions[i], targets[i])
                status = "PERFECT!" if wer == 0 else "EXCELLENT!" if wer < 0.2 else "GREAT!" if wer < 0.4 else "GOOD" if wer < 0.7 else "LEARNING"
                print(f"  {status}: '{targets[i]}' -> '{predictions[i]}' (WER: {wer:.3f})")
            print()
        
        # Early stopping
        if avg_wer == 0.0:
            print(f"PERFECT LEARNING achieved at epoch {epoch}!")
            break
        
        if patience_counter >= 30:
            print(f"Early stopping at epoch {epoch} (no improvement for 30 epochs)")
            break
    
    print("=" * 60)
    print("ATTENTION MODEL TRAINING COMPLETE!")
    print(f"Best WER achieved: {best_wer:.4f}")
    
    if best_wer < 0.2:
        print("EXCELLENT: Attention model working perfectly!")
        print("Ready for production deployment!")
    elif best_wer < 0.4:
        print("GREAT: Attention model learning very well!")
        print("Scale up to more data for production quality")
    elif best_wer < 0.7:
        print("GOOD: Attention model improving significantly")
        print("Continue training with more data")
    else:
        print("LEARNING: Model needs more training time")
        print("Consider architecture adjustments")
    
    return best_wer

if __name__ == "__main__":
    train_attention_model()