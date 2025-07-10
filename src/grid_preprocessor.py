#!/usr/bin/env python3
"""
GRID Dataset Preprocessor
Processes GRID corpus for enhanced lip-reading training
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm
import dlib
import shutil

class GRIDPreprocessor:
    def __init__(self, project_root="C:/Users/Jimmy/APU/Year3 Sem2/LipNet-FYP"):
        self.project_root = Path(project_root)
        self.grid_dir = self.project_root / "data" / "GRID"
        self.video_dir = self.grid_dir / "video"
        self.align_dir = self.grid_dir / "align"
        self.processed_dir = self.grid_dir / "processed"
        
        print(f"GRID Preprocessor for Enhanced LipNet")
        print(f"Project: {self.project_root}")
        print(f"GRID: {self.grid_dir}")
        
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = self.project_root / "models" / "shape_predictor_68_face_landmarks.dat"
        
        if predictor_path.exists():
            print("Face landmark predictor found!")
            self.predictor = dlib.shape_predictor(str(predictor_path))
        else:
            print("Face landmark predictor not found")
            print("Please download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"Save to: {predictor_path}")
            self.predictor = None
        
        # Processing parameters
        self.target_size = (112, 112)  # Lip region size
        self.sequence_length = 75      # Max frames per video
    
    def scan_dataset(self):
        """Scan available speakers and videos"""
        speakers = []
        total_videos = 0
        
        print("Scanning GRID dataset...")
        
        for speaker_dir in self.video_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                video_files = list(speaker_dir.glob("*.mpg"))
                
                # Handle different alignment folder naming patterns
                # Try standard naming first (s1, s10, s23)
                align_speaker_dir = self.align_dir / speaker_id
                if not align_speaker_dir.exists():
                    # Try with _align suffix (s1_align, s10_align, s23_align)
                    align_speaker_dir = self.align_dir / f"{speaker_id}_align"
                
                if align_speaker_dir.exists():
                    align_files = list(align_speaker_dir.glob("*.align"))
                else:
                    align_files = []
                
                print(f"Speaker {speaker_id}: {len(video_files)} videos, {len(align_files)} alignments")
                print(f"  Video dir: {speaker_dir}")
                print(f"  Align dir: {align_speaker_dir}")
                
                speakers.append({
                    'id': speaker_id,
                    'video_count': len(video_files),
                    'align_count': len(align_files),
                    'video_files': video_files,
                    'align_files': align_files,
                    'align_dir': align_speaker_dir
                })
                total_videos += len(video_files)
        
        print(f"Found {len(speakers)} speakers, {total_videos} total videos")
        return speakers
    
    def extract_lip_region(self, frame):
        """Extract lip region from frame using facial landmarks"""
        try:
            # Ensure frame is valid and convert to proper format
            if frame is None or frame.size == 0:
                return None
            
            # Convert to 8-bit if needed
            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Fallback if no predictor
            if self.predictor is None:
                h, w = gray.shape
                lip_region = gray[int(h*0.6):int(h*0.9), int(w*0.25):int(w*0.75)]
                if lip_region.size > 0:
                    return cv2.resize(lip_region, self.target_size)
                else:
                    return np.zeros(self.target_size, dtype=np.uint8)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) == 0:
                # No face detected, use lower center region
                h, w = gray.shape
                lip_region = gray[int(h*0.6):int(h*0.9), int(w*0.25):int(w*0.75)]
                if lip_region.size > 0:
                    return cv2.resize(lip_region, self.target_size)
                else:
                    return np.zeros(self.target_size, dtype=np.uint8)
            
            # Use the first detected face
            face = faces[0]
            landmarks = self.predictor(gray, face)
            
            # Extract lip landmarks (points 48-67)
            lip_points = []
            for i in range(48, 68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                lip_points.append([x, y])
            
            lip_points = np.array(lip_points)
            
            # Calculate bounding box around lips with padding
            min_x, min_y = np.min(lip_points, axis=0)
            max_x, max_y = np.max(lip_points, axis=0)
            
            # Add padding
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(gray.shape[1], max_x + padding)
            max_y = min(gray.shape[0], max_y + padding)
            
            # Extract and resize lip region
            lip_region = gray[min_y:max_y, min_x:max_x]
            if lip_region.size > 0:
                return cv2.resize(lip_region, self.target_size)
            else:
                return np.zeros(self.target_size, dtype=np.uint8)
                
        except Exception as e:
            print(f"Error in lip extraction: {e}")
            # Return blank region as fallback
            return np.zeros(self.target_size, dtype=np.uint8)
    
    def parse_alignment(self, align_file):
        """Parse GRID alignment file to get word timings"""
        words = []
        with open(align_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = int(parts[0])
                    end_time = int(parts[1])
                    word = parts[2]
                    words.append({
                        'start': start_time,
                        'end': end_time,
                        'word': word
                    })
        return words
    
    def process_video(self, video_path, align_path):
        """Process a single video file"""
        try:
            # Read video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Read alignment if available
            if align_path.exists():
                try:
                    alignment = self.parse_alignment(align_path)
                    # Create sentence from words (excluding 'sil')
                    sentence = ' '.join([w['word'] for w in alignment if w['word'] != 'sil']).upper()
                except Exception as e:
                    print(f"Error reading alignment {align_path}: {e}")
                    # Generate sentence from filename if alignment fails
                    sentence = self.generate_sentence_from_filename(video_path.stem)
            else:
                # Generate sentence from filename
                sentence = self.generate_sentence_from_filename(video_path.stem)
            
            # Extract frames and lip regions
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract lip region
                lip_region = self.extract_lip_region(frame)
                if lip_region is not None:
                    frames.append(lip_region)
                    frame_count += 1
                
                # Limit sequence length
                if frame_count >= self.sequence_length:
                    break
            
            cap.release()
            
            if len(frames) == 0:
                print(f"No valid frames extracted from: {video_path}")
                return None
            
            # Pad or truncate to fixed length
            if len(frames) < self.sequence_length:
                # Pad with last frame
                last_frame = frames[-1]
                while len(frames) < self.sequence_length:
                    frames.append(last_frame.copy())
            else:
                frames = frames[:self.sequence_length]
            
            # Convert to tensor
            video_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add channel dimension
            video_tensor = video_tensor / 255.0  # Normalize to [0,1]
            
            return {
                'video': video_tensor,
                'sentence': sentence,
                'video_path': str(video_path),
                'align_path': str(align_path),
                'fps': fps,
                'original_frames': frame_count
            }
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
    
    def generate_sentence_from_filename(self, filename):
        """Generate a sentence from GRID filename pattern"""
        # GRID filenames follow pattern like: bbaf2n, lgir5s, etc.
        # This is a simplified mapping - you might need actual GRID word mappings
        grid_words = {
            'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set',
            'a': 'at', 'g': 'green', 'r': 'red', 'i': 'in',
            'f': 'four', 'n': 'now', 'z': 'zero', 't': 'two'
        }
        
        words = []
        for char in filename.lower():
            if char in grid_words:
                words.append(grid_words[char])
        
        return ' '.join(words).upper() if words else "UNKNOWN SEQUENCE"
    
    def create_splits(self, speakers, train_ratio=0.8, val_ratio=0.1):
        """Create train/val/test splits"""
        import random
        random.seed(42)  # For reproducibility
        
        all_samples = []
        
        # Collect all video-alignment pairs
        for speaker in speakers:
            speaker_id = speaker['id']
            video_files = speaker['video_files']
            align_dir = speaker['align_dir']  # Use the actual align directory found
            
            for video_file in video_files:
                # Find corresponding alignment file
                align_file = align_dir / (video_file.stem + '.align')
                
                all_samples.append({
                    'video_path': video_file,
                    'align_path': align_file,
                    'speaker': speaker_id
                })
        
        # Shuffle samples
        random.shuffle(all_samples)
        
        # Create splits
        total_samples = len(all_samples)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end],
            'test': all_samples[val_end:]
        }
        
        print(f"Dataset splits:")
        for split_name, samples in splits.items():
            print(f"  {split_name}: {len(samples)} samples")
        
        return splits
    
    def process_split(self, samples, split_name):
        """Process all samples in a split"""
        split_dir = self.processed_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        processed_samples = []
        
        print(f"Processing {split_name} split ({len(samples)} samples)...")
        
        for i, sample in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
            result = self.process_video(sample['video_path'], sample['align_path'])
            
            if result is not None:
                # Save processed sample
                sample_file = split_dir / f"sample_{i:04d}.pkl"
                with open(sample_file, 'wb') as f:
                    pickle.dump(result, f)
                
                processed_samples.append({
                    'file': str(sample_file),
                    'sentence': result['sentence'],
                    'speaker': sample['speaker']
                })
        
        # Save split metadata
        metadata_file = split_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(processed_samples, f)
        
        print(f"{split_name} split: {len(processed_samples)} samples processed")
        return processed_samples
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("Starting GRID dataset preprocessing...")
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan dataset
        speakers = self.scan_dataset()
        
        if len(speakers) == 0:
            print("No speakers found! Check your GRID dataset structure.")
            return False
        
        # Create splits
        splits = self.create_splits(speakers)
        
        # Process each split
        all_processed = {}
        for split_name, samples in splits.items():
            processed = self.process_split(samples, split_name)
            all_processed[split_name] = processed
        
        # Save overall statistics
        stats = {
            'speakers': [s['id'] for s in speakers],
            'total_videos': sum(len(s['video_files']) for s in speakers),
            'splits': {name: len(samples) for name, samples in all_processed.items()},
            'target_size': self.target_size,
            'sequence_length': self.sequence_length
        }
        
        stats_file = self.processed_dir / "dataset_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        print("=" * 60)
        print("GRID dataset preprocessing COMPLETE!")
        print(f"Statistics: {stats}")
        print("Processed data saved to:", self.processed_dir)
        print("Ready for model training!")
        print("=" * 60)
        
        return True

def main():
    preprocessor = GRIDPreprocessor()
    
    # Check if face landmark predictor exists
    predictor_path = Path("C:/Users/Jimmy/APU/Year3 Sem2/LipNet-FYP/models/shape_predictor_68_face_landmarks.dat")
    if not predictor_path.exists():
        print("Downloading face landmark predictor...")
        predictor_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("Please download manually:")
        print("1. Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("2. Extract to:", predictor_path)
        print("3. Run this script again")
        return
    
    # Run preprocessing
    success = preprocessor.run_preprocessing()
    
    if success:
        print("Preprocessing completed successfully!")
        print("Next step: Start model training")
    else:
        print("Preprocessing failed!")

if __name__ == "__main__":
    main()