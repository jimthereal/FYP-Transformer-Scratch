"""
Check the shape of saved preprocessed data
"""

import pickle
import torch
from pathlib import Path

def check_data_shapes(data_dir="data/processed_grid_only/train"):
    """Check shapes of saved data"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        return
    
    sample_files = list(data_path.glob("*.pkl"))
    
    if not sample_files:
        print(f"ERROR: No pkl files found in {data_path}")
        return
    
    print(f"Checking data shapes in: {data_path}")
    print(f"Found {len(sample_files)} files")
    print("=" * 60)
    
    # Check first few samples
    for i, sample_file in enumerate(sample_files[:5]):
        print(f"\nSample {i+1}: {sample_file.name}")
        
        try:
            with open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            video = data['video']
            
            print(f"  Video type: {type(video)}")
            
            if isinstance(video, torch.Tensor):
                print(f"  Video shape: {video.shape}")
                print(f"  Video dtype: {video.dtype}")
                print(f"  Video range: [{video.min():.3f}, {video.max():.3f}]")
            else:
                print(f"  WARNING: Video is not a tensor!")
            
            print(f"  Sentence: '{data['sentence']}'")
            print(f"  Speaker ID: {data.get('speaker_id', 'N/A')}")
            
            # Check all keys
            print(f"  Keys: {list(data.keys())}")
            
        except Exception as e:
            print(f"  ERROR reading file: {e}")
    
    print("\n" + "=" * 60)
    print("EXPECTED SHAPE: [1, 75, 128, 64] or [75, 128, 64]")
    print("  [channels, frames, height, width]")

if __name__ == "__main__":
    # Check train data
    print("CHECKING TRAIN DATA")
    check_data_shapes("data/processed_grid_only/train")
    
    # Check test data
    print("\n\nCHECKING TEST DATA")
    check_data_shapes("data/processed_grid_only/test")