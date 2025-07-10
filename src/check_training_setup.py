# src/check_training_setup.py
# Quick checker to ensure everything is ready for training

import torch
import os
import sys
import pickle
from collections import Counter

def check_cuda_setup():
    """Check CUDA availability and memory"""
    print("Checking CUDA Setup...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        print(f"GPU Available: {gpu_name}")
        print(f"Memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB")
        
        # Test tensor operations
        test_tensor = torch.randn(1, 3, 75, 64, 128).cuda()
        print(f"CUDA tensor operations working")
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
    else:
        print("CUDA not available - will use CPU (much slower)")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\nChecking Dependencies...")
    
    # Map package names to their import names
    package_imports = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'numpy': 'numpy',
        'opencv-python': 'cv2',  # opencv-python imports as cv2
        'tqdm': 'tqdm',
        'matplotlib': 'matplotlib',
        'spellchecker': 'spellchecker'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"Found: {package}")
        except ImportError:
            print(f"MISSING: {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """Check for required model files"""
    print("\nChecking Model Files...")
    
    # Check base model files
    base_model_path = "src/models/lipnet_base.py"
    if os.path.exists(base_model_path):
        print(f"Base model found: {base_model_path}")
    else:
        print(f"Base model missing: {base_model_path}")
        return False
    
    # Check pre-trained weights
    pretrained_path = "models/pretrained/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt"
    if os.path.exists(pretrained_path):
        print(f"Pre-trained weights found: {pretrained_path}")
    else:
        print(f"Pre-trained weights not found: {pretrained_path}")
        print("   Training will start from scratch")
    
    return True

def check_processed_data():
    """Check processed training data"""
    print("\nChecking Processed Data...")
    
    data_dir = "data/GRID/processed/train"
    
    if not os.path.exists(data_dir):
        print(f"Data directory missing: {data_dir}")
        return False
    
    # Count sample files
    pkl_files = [f for f in os.listdir(data_dir) if f.startswith('sample_') and f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} sample files")
    
    if len(pkl_files) < 100:
        print("Less than 100 samples - consider processing more data")
    
    # Test loading a sample
    if pkl_files:
        try:
            sample_path = os.path.join(data_dir, pkl_files[0])
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
            
            print(f"Sample structure valid:")
            print(f"   Video tensor shape: {sample['video'].shape}")
            print(f"   Sentence: '{sample['sentence']}'")
            
            return True
            
        except Exception as e:
            print(f"Error loading sample: {e}")
            return False
    
    return False

def estimate_training_time(num_samples, batch_size=2, num_epochs=10):
    """Estimate training time"""
    print(f"\nTraining Time Estimation:")
    
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * num_epochs
    
    # Rough estimates based on RTX 3060
    seconds_per_batch_gpu = 3.0  # seconds (conservative for batch_size=2)
    seconds_per_batch_cpu = 20.0  # seconds
    
    gpu_available = torch.cuda.is_available()
    seconds_per_batch = seconds_per_batch_gpu if gpu_available else seconds_per_batch_cpu
    
    total_seconds = total_batches * seconds_per_batch
    hours = total_seconds / 3600
    
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total batches: {total_batches}")
    print(f"Device: {'GPU' if gpu_available else 'CPU'}")
    print(f"Estimated time: {hours:.1f} hours")
    
    if hours > 8:
        print("Consider using Google Colab for faster training")

def recommend_batch_size():
    """Recommend batch size based on available memory"""
    print(f"\nBatch Size Recommendations:")
    
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if memory_gb >= 12:
            recommended_batch = 8
        elif memory_gb >= 8:
            recommended_batch = 4
        elif memory_gb >= 6:
            recommended_batch = 2
        else:
            recommended_batch = 1
        
        print(f"GPU Memory: {memory_gb:.1f}GB")
        print(f"Recommended batch size: {recommended_batch}")
        
        if memory_gb < 8:
            print("Consider using Google Colab T4 (15GB) for larger batches")
    else:
        print("CPU training - use batch size 1")

def main():
    """Main setup checker"""
    print("Enhanced LipNet Training Setup Checker")
    print("="*70)
    
    all_good = True
    
    # Run all checks
    all_good &= check_dependencies()
    cuda_available = check_cuda_setup()
    all_good &= check_model_files()
    data_ready = check_processed_data()
    all_good &= data_ready
    
    if data_ready:
        # Count samples for estimation
        data_dir = "data/GRID/processed/train"
        pkl_files = [f for f in os.listdir(data_dir) if f.startswith('sample_') and f.endswith('.pkl')]
        estimate_training_time(len(pkl_files))
    
    recommend_batch_size()
    
    print("\n" + "="*70)
    
    if all_good:
        print("ALL CHECKS PASSED!")
        print("Ready to start training!")
        print("\nNext step:")
        print("python src/train_enhanced_lipnet.py")
    else:
        print("Some issues found - please fix them before training")
    
    print("="*70)

if __name__ == "__main__":
    main()