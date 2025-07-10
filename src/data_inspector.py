"""
Data Inspector - Check what's actually in your processed data files
"""

import os
import pickle
import torch
import numpy as np

def inspect_data_directory(data_path):
    """Inspect the contents of processed data directory"""
    print("=" * 60)
    print("DATA DIRECTORY INSPECTION")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"ERROR: Directory not found: {data_path}")
        return
    
    # List all files
    files = os.listdir(data_path)
    pkl_files = [f for f in files if f.endswith('.pkl')]
    
    print(f"Directory: {data_path}")
    print(f"Total files: {len(files)}")
    print(f"Pickle files: {len(pkl_files)}")
    print()
    
    if not pkl_files:
        print("ERROR: No pickle files found!")
        return
    
    # Check first few files in detail
    print("DETAILED FILE INSPECTION:")
    print("-" * 40)
    
    for i, filename in enumerate(pkl_files[:5]):  # Check first 5 files
        filepath = os.path.join(data_path, filename)
        file_size = os.path.getsize(filepath)
        
        print(f"\nFile {i+1}: {filename}")
        print(f"Size: {file_size} bytes")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Type: {type(data)}")
            
            if isinstance(data, list):
                print(f"List length: {len(data)}")
                if len(data) > 0:
                    print(f"First element type: {type(data[0])}")
                    if hasattr(data[0], 'shape'):
                        print(f"First element shape: {data[0].shape}")
                else:
                    print("WARNING: Empty list!")
                    
            elif isinstance(data, dict):
                print(f"Dictionary keys: {list(data.keys())}")
                for key, value in data.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {type(value)} shape {value.shape}")
                    else:
                        print(f"  {key}: {type(value)} = {value}")
                        
            elif isinstance(data, tuple):
                print(f"Tuple length: {len(data)}")
                for j, item in enumerate(data):
                    if hasattr(item, 'shape'):
                        print(f"  Item {j}: {type(item)} shape {item.shape}")
                    else:
                        print(f"  Item {j}: {type(item)} = {item}")
                        
            else:
                print(f"Unknown data type: {type(data)}")
                
        except Exception as e:
            print(f"ERROR loading file: {e}")
    
    print("\n" + "=" * 60)

def check_original_grid_data():
    """Check if original GRID data exists"""
    print("\nORIGINAL GRID DATA CHECK:")
    print("-" * 40)
    
    grid_paths = [
        "data/GRID",
        "data/GRID/train",
        "data/GRID/val", 
        "data/GRID/raw"
    ]
    
    for path in grid_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"{path}: {len(files)} items")
            if files:
                print(f"  Sample files: {files[:3]}")
        else:
            print(f"{path}: NOT FOUND")

def check_preprocessing_script():
    """Check if preprocessing script exists and look for clues"""
    print("\nPREPROCESSING SCRIPT CHECK:")
    print("-" * 40)
    
    potential_scripts = [
        "src/grid_preprocessor.py",
        "src/preprocess_grid.py",
        "src/data_preprocessing.py"
    ]
    
    for script in potential_scripts:
        if os.path.exists(script):
            print(f"Found: {script}")
            # Quick look at the file
            with open(script, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"  Lines: {len(lines)}")
                # Look for save operations
                save_lines = [line for line in lines if 'pickle' in line or '.pkl' in line]
                if save_lines:
                    print("  Pickle operations found:")
                    for line in save_lines[:3]:
                        print(f"    {line.strip()}")
        else:
            print(f"Not found: {script}")

if __name__ == "__main__":
    # Check processed data
    inspect_data_directory("data/GRID/processed/train")
    
    # Check validation data too
    if os.path.exists("data/GRID/processed/val"):
        print("\n")
        inspect_data_directory("data/GRID/processed/val")
    
    # Check original data
    check_original_grid_data()
    
    # Check preprocessing scripts
    check_preprocessing_script()
    
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    print("1. If pickle files are empty, re-run preprocessing")
    print("2. Check preprocessing script for errors")
    print("3. Verify original GRID data is present")
    print("4. Consider regenerating processed data")