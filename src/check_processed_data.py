#!/usr/bin/env python3
"""
Check Processed GRID Dataset Status
Verify the preprocessing results and dataset statistics
"""

import pickle
from pathlib import Path
import torch

def check_processed_dataset():
    project_root = Path("C:/Users/Jimmy/APU/Year3 Sem2/LipNet-FYP")
    processed_dir = project_root / "data" / "GRID" / "processed"
    
    print("Checking Processed GRID Dataset")
    print("=" * 50)
    
    # Check overall statistics
    stats_file = processed_dir / "dataset_stats.pkl"
    if stats_file.exists():
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        
        print("Dataset Statistics:")
        print(f"  Speakers: {stats['speakers']}")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Target size: {stats['target_size']}")
        print(f"  Sequence length: {stats['sequence_length']}")
        print()
        
        print("Split Information:")
        for split_name, count in stats['splits'].items():
            print(f"  {split_name}: {count} samples")
    else:
        print("No dataset_stats.pkl found")
    
    print()
    
    # Check each split
    splits = ['train', 'val', 'test']
    
    for split_name in splits:
        split_dir = processed_dir / split_name
        
        if split_dir.exists():
            # Count sample files
            sample_files = list(split_dir.glob("sample_*.pkl"))
            metadata_file = split_dir / "metadata.pkl"
            
            print(f"{split_name.upper()} Split:")
            print(f"  Sample files: {len(sample_files)}")
            print(f"  Metadata exists: {metadata_file.exists()}")
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    print(f"  Metadata samples: {len(metadata)}")
                    
                    # Show first few samples
                    if len(metadata) > 0:
                        print("  Sample sentences:")
                        for i, sample in enumerate(metadata[:3]):
                            print(f"    {i+1}: {sample['sentence']}")
                        if len(metadata) > 3:
                            print(f"    ... and {len(metadata)-3} more")
                            
                except Exception as e:
                    print(f"  Error reading metadata: {e}")
            
            # Test loading a sample
            if len(sample_files) > 0:
                try:
                    sample_file = sample_files[0]
                    with open(sample_file, 'rb') as f:
                        sample = pickle.load(f)
                    
                    print(f"  Sample data structure:")
                    print(f"    Video tensor shape: {sample['video'].shape}")
                    print(f"    Sentence: {sample['sentence']}")
                    print(f"    Original frames: {sample.get('original_frames', 'N/A')}")
                    
                except Exception as e:
                    print(f"  Error loading sample: {e}")
        else:
            print(f"{split_name.upper()}: Directory not found")
        
        print()
    
    print("=" * 50)
    
    # Summary
    total_samples = 0
    for split_name in splits:
        split_dir = processed_dir / split_name
        if split_dir.exists():
            sample_count = len(list(split_dir.glob("sample_*.pkl")))
            total_samples += sample_count
    
    print(f"TOTAL PROCESSED SAMPLES: {total_samples}")
    
    if total_samples > 0:
        print("SUCCESS: Dataset preprocessing completed!")
        print("Ready for model training!")
    else:
        print("WARNING: No processed samples found!")
    
    return total_samples > 0

if __name__ == "__main__":
    check_processed_dataset()