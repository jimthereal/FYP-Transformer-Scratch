# src/debug_tensors.py
# Quick debug script to check tensor shapes

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_enhanced_lipnet import EnhancedGRIDDataset
from src.models.lipnet_base import LipNet

def debug_dataset():
    """Debug dataset loading and tensor shapes"""
    print("Debugging Dataset...")
    
    dataset = EnhancedGRIDDataset('data/GRID', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Check first sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Video dtype: {sample['video'].dtype}")
    print(f"Video min/max: {sample['video'].min():.3f}/{sample['video'].max():.3f}")
    print(f"Sentence: '{sample['sentence']}'")
    print(f"Sentence indices shape: {sample['sentence_indices'].shape}")
    print(f"Sentence length: {sample['sentence_length']}")
    
    return sample

def debug_model(sample):
    """Debug model forward pass"""
    print("\nDebugging Model...")
    
    model = LipNet()
    model.eval()
    
    # Add batch dimension
    video_batch = sample['video'].unsqueeze(0)  # [1, 3, 75, 64, 128]
    print(f"Video batch shape: {video_batch.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(video_batch)
        print(f"Model output shape: {outputs.shape}")
        print("Model forward pass successful!")
        return outputs
    except Exception as e:
        print(f"Model forward pass failed: {e}")
        return None

def debug_ctc_inputs(outputs, sample):
    """Debug CTC loss inputs"""
    print("\nDebugging CTC Loss...")
    
    if outputs is None:
        print("No outputs to debug")
        return
    
    # Model outputs [B, T, C], need to transpose to [T, B, C] for CTC
    print(f"Original outputs shape: {outputs.shape}")  # [1, 75, 28]
    outputs = outputs.transpose(0, 1)  # [75, 1, 28]
    print(f"Transposed outputs shape: {outputs.shape}")  # [75, 1, 28]
    
    batch_size = outputs.size(1)  # B
    sequence_length = outputs.size(0)  # T
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    
    # Prepare CTC inputs
    log_probs = torch.log_softmax(outputs, dim=2)
    input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long)
    
    sentence_indices = sample['sentence_indices']
    target_lengths = torch.tensor([len(sentence_indices)], dtype=torch.long)
    
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Input lengths: {input_lengths}")
    print(f"Target lengths: {target_lengths}")
    print(f"Targets shape: {sentence_indices.shape}")
    
    # Test CTC loss
    try:
        criterion = torch.nn.CTCLoss(blank=27, reduction='mean', zero_infinity=True)
        loss = criterion(log_probs, sentence_indices, input_lengths, target_lengths)
        print(f"CTC loss: {loss.item()}")
        print("CTC loss calculation successful!")
    except Exception as e:
        print(f"CTC loss calculation failed: {e}")

def main():
    print("Tensor Shape Debugging")
    print("="*50)
    
    sample = debug_dataset()
    outputs = debug_model(sample)
    debug_ctc_inputs(outputs, sample)
    
    print("\nDebugging complete!")

if __name__ == "__main__":
    main()