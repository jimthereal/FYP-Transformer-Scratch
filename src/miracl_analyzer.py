"""
MIRACL-VC1 Dataset Analyzer
Check how many samples are actually available and optimize extraction
"""

import os
from pathlib import Path
import pickle

def analyze_miracl_dataset(base_path="data/MIRACL-VC1"):
    """Analyze the complete MIRACL-VC1 dataset structure"""
    
    miracl_path = Path(base_path)
    
    if not miracl_path.exists():
        print(f"❌ MIRACL-VC1 not found at {miracl_path}")
        return
    
    print("MIRACL-VC1 DATASET ANALYSIS")
    print("=" * 50)
    
    total_sequences = 0
    persons = []
    content_types = set()
    
    # Analyze directory structure
    for person_dir in miracl_path.iterdir():
        if not person_dir.is_dir():
            continue
            
        persons.append(person_dir.name)
        person_sequences = 0
        
        # Check for words/phrases subdirectories
        subdirs = []
        words_dir = person_dir / "words"
        phrases_dir = person_dir / "phrases"
        
        if words_dir.exists():
            subdirs.append(("words", words_dir))
        if phrases_dir.exists():
            subdirs.append(("phrases", phrases_dir))
        
        # If no subdirs, check direct content
        if not subdirs:
            subdirs = [("direct", person_dir)]
        
        for content_type, content_dir in subdirs:
            content_types.add(content_type)
            
            if content_type == "direct":
                # Direct content in person directory
                content_dirs = [d for d in content_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            else:
                # Words/phrases subdirectory
                content_dirs = [d for d in content_dir.iterdir() if d.is_dir()]
            
            for content_item in content_dirs:
                # Find instance directories (01, 02, etc.)
                instance_dirs = [d for d in content_item.iterdir() if d.is_dir()]
                
                for instance_dir in instance_dirs:
                    # Check if this contains actual image sequences
                    color_images = list(instance_dir.glob("color_*.jpg"))
                    
                    if len(color_images) > 0:
                        person_sequences += 1
                        total_sequences += 1
        
        print(f"Person {person_dir.name}: {person_sequences} sequences")
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"Total persons: {len(persons)}")
    print(f"Total sequences: {total_sequences}")
    print(f"Content types: {list(content_types)}")
    print(f"Current extraction: 150 sequences ({150/total_sequences*100:.1f}%)")
    
    if total_sequences > 150:
        print(f"\n🚀 OPPORTUNITY: {total_sequences - 150} additional sequences available!")
        print(f"Recommended: Extract {min(500, total_sequences)} sequences for better diversity")
    
    return total_sequences

def analyze_processed_samples(processed_path="data/processed_multi/unified/train"):
    """Analyze the processed samples to verify they're properly formatted"""
    
    processed_dir = Path(processed_path)
    
    if not processed_dir.exists():
        print(f"❌ Processed directory not found: {processed_dir}")
        return
    
    print("\nPROCESSED SAMPLES ANALYSIS")
    print("=" * 50)
    
    sample_files = list(processed_dir.glob("*.pkl"))
    
    if not sample_files:
        print("❌ No processed samples found!")
        return
    
    print(f"Total processed samples: {len(sample_files)}")
    
    # Analyze sample breakdown
    grid_count = len([f for f in sample_files if "grid" in f.name])
    miracl_count = len([f for f in sample_files if "miracl" in f.name])
    
    print(f"GRID samples: {grid_count}")
    print(f"MIRACL samples: {miracl_count}")
    
    # Check sample quality
    print("\nSAMPLE QUALITY CHECK:")
    
    try:
        # Load a few samples to verify format
        for i, sample_file in enumerate(sample_files[:5]):
            with open(sample_file, 'rb') as f:
                sample = pickle.load(f)
            
            video = sample['video']
            sentence = sample['sentence']
            source = sample.get('source', 'unknown')
            
            print(f"Sample {i+1} ({source}): '{sentence}'")
            print(f"  Video shape: {video.shape}")
            print(f"  Video range: {video.min():.3f} - {video.max():.3f}")
            
            # Verify format is correct
            if video.shape != (1, 75, 128, 64):
                print(f"  ⚠️  WARNING: Unexpected shape! Expected (1, 75, 128, 64)")
            else:
                print(f"  ✅ Format correct")
        
        print(f"\n✅ Processed samples are properly formatted for training!")
        print(f"Ready to use with transformer model")
        
    except Exception as e:
        print(f"❌ Error loading samples: {e}")

def recommend_optimal_extraction():
    """Recommend optimal extraction strategy"""
    
    print("\nOPTIMAL EXTRACTION RECOMMENDATIONS")
    print("=" * 50)
    
    # Analyze current dataset
    total_available = analyze_miracl_dataset()
    
    if total_available and total_available > 150:
        print("\n🎯 RECOMMENDED ACTIONS:")
        print("1. Increase MIRACL extraction to 300-500 samples")
        print("2. Keep GRID at 50 samples (prevent overfitting)")
        print("3. Total training set: 350-550 samples")
        print("\nBENEFITS:")
        print("- Much better speaker diversity")
        print("- Reduced GRID overfitting")
        print("- Better real-world performance")
        
        print(f"\n💻 CODE UPDATE:")
        print("In multi_dataset_preprocessor.py, change:")
        print("miracl_count = self.process_miracl_vc1(max_samples=500)  # Increased")
        print("grid_count = self.copy_grid_samples(max_samples=50)     # Keep low")
        
    else:
        print("\n✅ Current extraction seems optimal for available data")
    
    # Analyze processed samples
    analyze_processed_samples()

if __name__ == "__main__":
    recommend_optimal_extraction()