"""
Test Diagnostic with Real Working Data
Now that we have real video data, test if model can learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
import pickle

class SimpleLipNet(nn.Module):
    """Simplified LipNet for the new data format [1, 75, 128, 64]"""
    def __init__(self, vocab_size=28):
        super(SimpleLipNet, self).__init__()
        
        # 3D CNN for input [B, 1, 75, 128, 64]
        self.conv3d = nn.Sequential(
            # [B, 1, 75, 128, 64] -> [B, 32, 75, 64, 32]
            nn.Conv3d(1, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # -> [B, 32, 75, 32, 16]
            
            # [B, 32, 75, 32, 16] -> [B, 64, 75, 16, 8]
            nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # -> [B, 64, 75, 16, 8]
            
            # [B, 64, 75, 16, 8] -> [B, 96, 75, 8, 4]
            nn.Conv3d(64, 96, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))  # -> [B, 96, 75, 8, 4]
        )
        
        # Calculate feature size: 96 * 8 * 4 = 3072
        self.feature_size = 96 * 8 * 4
        
        # RNN
        self.rnn = nn.LSTM(self.feature_size, 256, batch_first=True, bidirectional=True)
        
        # Classification
        self.classifier = nn.Linear(512, vocab_size)
        
    def forward(self, x):
        # x shape: [B, 1, T, H, W] = [B, 1, 75, 128, 64]
        batch_size, _, seq_len, _, _ = x.size()
        
        # CNN features
        x = self.conv3d(x)  # [B, 96, 75, 8, 4]
        
        # Reshape for RNN
        x = x.permute(0, 2, 1, 3, 4)  # [B, 75, 96, 8, 4]
        x = x.reshape(batch_size, seq_len, -1)  # [B, 75, 3072]
        
        # RNN
        x, _ = self.rnn(x)  # [B, 75, 512]
        
        # Classification
        x = self.classifier(x)  # [B, 75, vocab_size]
        
        return x

class RealDataset(Dataset):
    """Load real processed data"""
    def __init__(self, data_path, max_samples=3):
        self.data_path = data_path
        self.samples = []
        
        # Load sample files
        all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pkl')])
        sample_files = [f for f in all_files if f.startswith('sample_')][:max_samples]
        
        print(f"Loading {len(sample_files)} REAL samples...")
        
        for sample_file in sample_files:
            with open(os.path.join(data_path, sample_file), 'rb') as f:
                sample = pickle.load(f)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Show sample info
        for i, sample in enumerate(self.samples):
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

def test_real_data():
    """Test training with real video data"""
    print("TESTING WITH REAL VIDEO DATA")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Character mappings
    char_to_idx, idx_to_char = create_char_mappings()
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load real data
    data_path = "data/GRID/processed_simple/train"
    if not os.path.exists(data_path):
        print(f"ERROR: Real data not found at {data_path}")
        print("Run simple_preprocessor.py first!")
        return
    
    dataset = RealDataset(data_path, max_samples=3)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Model for new data dimensions
    model = SimpleLipNet(vocab_size=vocab_size).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=vocab_size-1, reduction='mean', zero_infinity=True)
    
    print(f"Training on {len(dataset)} REAL samples")
    print("Expected: Model should learn much faster with real data!")
    print()
    
    # Training loop
    model.train()
    best_wer = 1.0
    
    for epoch in range(100):
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
        
        if avg_wer < best_wer:
            best_wer = avg_wer
        
        # Print progress
        if epoch % 10 == 0 or epoch < 5 or avg_wer < 0.5:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, WER={avg_wer:.4f} (Best: {best_wer:.4f})")
            
            for i in range(len(predictions)):
                wer = calculate_wer(predictions[i], targets[i])
                status = "PERFECT!" if wer == 0 else "GREAT!" if wer < 0.3 else "GOOD" if wer < 0.7 else "LEARNING"
                print(f"  {status}: '{targets[i]}' -> '{predictions[i]}' (WER: {wer:.3f})")
            print()
        
        # Early stopping
        if avg_wer == 0.0:
            print(f"PERFECT LEARNING achieved at epoch {epoch}!")
            break
    
    print("=" * 60)
    print("REAL DATA TEST COMPLETE!")
    print(f"Best WER achieved: {best_wer:.4f}")
    
    if best_wer < 0.3:
        print("EXCELLENT: Model learns well with real data!")
        print("Next: Scale up to full dataset")
    elif best_wer < 0.7:
        print("GOOD: Model is learning, needs more data/epochs")
        print("Next: Add more samples and train longer")
    else:
        print("LEARNING: Model improving but slowly")
        print("Next: Try different architecture or parameters")
    
    return best_wer

if __name__ == "__main__":
    test_real_data()