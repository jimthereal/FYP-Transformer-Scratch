"""
Data Quality Visualizer
Check if different videos actually contain different visual information
"""

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_samples(data_path, num_samples=3):
    """Load and return sample data"""
    samples = []
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.pkl')])
    sample_files = [f for f in all_files if f.startswith('sample_')][:num_samples]
    
    for sample_file in sample_files:
        with open(os.path.join(data_path, sample_file), 'rb') as f:
            sample = pickle.load(f)
            samples.append(sample)
    
    return samples

def analyze_video_differences(samples):
    """Analyze if videos are actually different"""
    print("VIDEO DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    videos = []
    sentences = []
    
    for i, sample in enumerate(samples):
        video = sample['video']  # [1, 75, 112, 112]
        sentence = sample['sentence']
        
        videos.append(video)
        sentences.append(sentence)
        
        print(f"Sample {i}: '{sentence}'")
        print(f"  Video shape: {video.shape}")
        print(f"  Video min/max: {video.min():.3f} / {video.max():.3f}")
        print(f"  Video mean: {video.mean():.3f}")
        print(f"  Video std: {video.std():.3f}")
    
    # Compare videos pairwise
    print("\nPAIRWISE VIDEO COMPARISONS:")
    print("-" * 30)
    
    for i in range(len(videos)):
        for j in range(i+1, len(videos)):
            video1, video2 = videos[i], videos[j]
            
            # Calculate differences
            diff = torch.abs(video1 - video2)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            
            # Calculate correlation
            v1_flat = video1.flatten()
            v2_flat = video2.flatten()
            correlation = torch.corrcoef(torch.stack([v1_flat, v2_flat]))[0,1].item()
            
            print(f"Video {i} vs Video {j}:")
            print(f"  Sentences: '{sentences[i]}' vs '{sentences[j]}'")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Correlation: {correlation:.6f}")
            
            if mean_diff < 0.001:
                print("  WARNING: Videos are nearly identical!")
            elif correlation > 0.95:
                print("  WARNING: Videos are highly correlated!")
            else:
                print("  OK: Videos are sufficiently different")
            print()
    
    return videos, sentences

def visualize_sample_frames(videos, sentences, output_dir="debug_frames"):
    """Visualize sample frames from each video"""
    print("CREATING FRAME VISUALIZATIONS...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Frame indices to visualize
    frame_indices = [0, 15, 30, 45, 60, 74]  # Start, middle, end frames
    
    for video_idx, (video, sentence) in enumerate(zip(videos, sentences)):
        # video shape: [1, 75, 112, 112]
        video_frames = video.squeeze(0)  # [75, 112, 112]
        
        # Create subplot for this video
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Video {video_idx}: '{sentence}'", fontsize=14)
        
        for i, frame_idx in enumerate(frame_indices):
            row = i // 3
            col = i % 3
            
            frame = video_frames[frame_idx].numpy()
            
            axes[row, col].imshow(frame, cmap='gray')
            axes[row, col].set_title(f"Frame {frame_idx}")
            axes[row, col].axis('off')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/video_{video_idx}_frames.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_dir}/video_{video_idx}_frames.png")
    
    print(f"Frame visualizations saved to: {output_dir}/")

def analyze_temporal_changes(videos, sentences):
    """Analyze how much each video changes over time"""
    print("TEMPORAL CHANGE ANALYSIS")
    print("-" * 30)
    
    for video_idx, (video, sentence) in enumerate(zip(videos, sentences)):
        # video shape: [1, 75, 112, 112]
        video_frames = video.squeeze(0)  # [75, 112, 112]
        
        # Calculate frame-to-frame differences
        frame_diffs = []
        for t in range(1, 75):
            diff = torch.abs(video_frames[t] - video_frames[t-1]).mean().item()
            frame_diffs.append(diff)
        
        mean_temporal_change = np.mean(frame_diffs)
        max_temporal_change = np.max(frame_diffs)
        
        print(f"Video {video_idx}: '{sentence}'")
        print(f"  Mean temporal change: {mean_temporal_change:.6f}")
        print(f"  Max temporal change: {max_temporal_change:.6f}")
        
        if mean_temporal_change < 0.001:
            print("  WARNING: Very little temporal change - static video!")
        elif mean_temporal_change < 0.01:
            print("  CAUTION: Low temporal change - limited motion")
        else:
            print("  OK: Good temporal dynamics")
        print()

def check_lip_region_quality(videos, sentences):
    """Check if lip regions contain meaningful information"""
    print("LIP REGION QUALITY CHECK")
    print("-" * 30)
    
    for video_idx, (video, sentence) in enumerate(zip(videos, sentences)):
        # video shape: [1, 75, 112, 112]
        video_frames = video.squeeze(0)  # [75, 112, 112]
        
        # Focus on center region where lips should be
        h, w = video_frames.shape[1], video_frames.shape[2]
        lip_region = video_frames[:, h//3:2*h//3, w//4:3*w//4]  # Center region
        
        # Calculate statistics
        lip_mean = lip_region.mean().item()
        lip_std = lip_region.std().item()
        lip_range = (lip_region.max() - lip_region.min()).item()
        
        print(f"Video {video_idx}: '{sentence}'")
        print(f"  Lip region mean: {lip_mean:.3f}")
        print(f"  Lip region std: {lip_std:.3f}")
        print(f"  Lip region range: {lip_range:.3f}")
        
        # Check for issues
        if lip_std < 0.05:
            print("  WARNING: Very low variation in lip region!")
        elif lip_range < 0.1:
            print("  WARNING: Limited intensity range in lip region!")
        else:
            print("  OK: Reasonable lip region variation")
        print()

def main():
    """Run comprehensive data quality analysis"""
    print("DATA QUALITY DIAGNOSTIC")
    print("=" * 60)
    
    # Load samples
    data_path = "data/GRID/processed/train"
    samples = load_samples(data_path, num_samples=3)
    
    if not samples:
        print("ERROR: No samples loaded!")
        return
    
    # Analyze video differences
    videos, sentences = analyze_video_differences(samples)
    
    # Visualize frames
    visualize_sample_frames(videos, sentences)
    
    # Analyze temporal changes
    analyze_temporal_changes(videos, sentences)
    
    # Check lip region quality
    check_lip_region_quality(videos, sentences)
    
    print("=" * 60)
    print("RECOMMENDATIONS:")
    print("1. Check the generated frame images in debug_frames/")
    print("2. If videos look identical, reprocess with better lip detection")
    print("3. If little temporal change, check video extraction pipeline")
    print("4. If low lip region quality, improve face landmark detection")

if __name__ == "__main__":
    main()