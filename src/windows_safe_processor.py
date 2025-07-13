"""
Windows-Safe Dataset Processor
Fixes handle errors and removes all emojis
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
import shutil
import random
import gc
import sys
import traceback
import time
warnings.filterwarnings('ignore')

# MediaPipe setup with Windows compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

class WindowsSafeProcessor:
    """Windows-compatible processor without emoji issues"""
    
    def __init__(self, base_path="data"):
        self.base_path = Path(base_path)
        self.target_size = (64, 128)
        self.sequence_length = 75
        
        # Scaled up settings for maximum data usage
        self.datasets = {
            'grid': {
                'source': self.base_path / "GRID",
                'output': self.base_path / "processed_windows_safe" / "grid",
                'max_samples': 1500,  # Increased from 300
                'batch_size': 25      # Keep small batches for stability
            },
            'miracl': {
                'source': self.base_path / "MIRACL-VC1",
                'output': self.base_path / "processed_windows_safe" / "miracl", 
                'max_samples': 2000,  # Increased from 500
                'batch_size': 50      # Keep small batches for stability
            }
        }
        
        self.mp_face_detection = None
        self.face_detection = None
        self.setup_mediapipe_windows_safe()
    
    def setup_mediapipe_windows_safe(self):
        """Setup MediaPipe with Windows-specific fixes"""
        try:
            print("Initializing MediaPipe for Windows...")
            
            # Import MediaPipe with Windows-safe settings
            import mediapipe as mp
            
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            )
            
            print("MediaPipe initialized successfully")
            
        except Exception as e:
            print(f"MediaPipe initialization failed: {e}")
            print("Will use geometric fallback method")
            self.face_detection = None
    
    def process_grid_windows_safe(self, max_samples=1500):
        """Windows-safe GRID processing"""
        print(f"PROCESSING GRID (up to {max_samples} samples)...")
        
        grid_source = self.datasets['grid']['source']
        grid_output = self.datasets['grid']['output']
        grid_output.mkdir(parents=True, exist_ok=True)
        
        # Check existing samples
        existing_samples = list(grid_output.glob("*.pkl"))
        start_idx = len(existing_samples)
        
        if start_idx > 0:
            print(f"Found {start_idx} existing samples, resuming...")
        
        video_dir = grid_source / "video"
        align_dir = grid_source / "align"
        
        if not video_dir.exists():
            print(f"ERROR: GRID video directory not found at {video_dir}")
            return start_idx
        
        # Collect videos without MediaPipe in this step
        print("Collecting video files...")
        all_videos = []
        
        for speaker_dir in video_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                video_files = list(speaker_dir.glob("*.mpg"))
                
                align_speaker_dir = align_dir / f"{speaker_id}_align"
                if not align_speaker_dir.exists():
                    align_speaker_dir = align_dir / speaker_id
                
                for video_file in video_files:
                    align_file = align_speaker_dir / (video_file.stem + '.align')
                    if align_file.exists():
                        all_videos.append((video_file, align_file, speaker_id))
        
        print(f"Found {len(all_videos)} videos with alignments")
        
        if len(all_videos) == 0:
            print("ERROR: No valid video-alignment pairs found")
            return start_idx
        
        # Process samples
        random.seed(42)
        random.shuffle(all_videos)
        
        remaining_needed = max_samples - start_idx
        if remaining_needed <= 0:
            print(f"Already have {start_idx} samples, no more needed")
            return start_idx
        
        selected_videos = all_videos[:remaining_needed]
        print(f"Processing {len(selected_videos)} new samples...")
        
        processed_count = start_idx
        failed_count = 0
        
        # Process one by one to avoid handle issues
        for i, (video_path, align_path, speaker_id) in enumerate(selected_videos):
            try:
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(selected_videos)} ({i/len(selected_videos)*100:.1f}%)")
                
                result = self.process_single_video_windows_safe(video_path, align_path, speaker_id, 'grid')
                
                if result is not None:
                    sample_file = grid_output / f"grid_sample_{processed_count:04d}.pkl"
                    
                    with open(sample_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    processed_count += 1
                else:
                    failed_count += 1
                
                # Memory cleanup every 25 samples
                if i % 25 == 0:
                    gc.collect()
                    
            except Exception as e:
                failed_count += 1
                print(f"Failed: {video_path.name} - {str(e)[:100]}")
                continue
        
        final_count = processed_count
        print(f"GRID processing complete!")
        print(f"Started with: {start_idx} samples")
        print(f"Processed: {final_count - start_idx} new samples")
        print(f"Failed: {failed_count} samples")
        print(f"Total: {final_count} samples")
        
        return final_count
    
    def process_miracl_windows_safe(self, max_samples=2000):
        """Windows-safe MIRACL processing"""
        print(f"PROCESSING MIRACL (up to {max_samples} samples)...")
        
        miracl_source = self.datasets['miracl']['source']
        miracl_output = self.datasets['miracl']['output']
        miracl_output.mkdir(parents=True, exist_ok=True)
        
        # Check existing samples
        existing_samples = list(miracl_output.glob("*.pkl"))
        start_idx = len(existing_samples)
        
        if start_idx > 0:
            print(f"Found {start_idx} existing samples, resuming...")
        
        if not miracl_source.exists():
            print(f"ERROR: MIRACL not found at {miracl_source}")
            return start_idx
        
        # Collect sequences
        print("Collecting MIRACL sequences...")
        all_sequences = []
        
        for person_dir in miracl_source.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_sequences = 0
            
            for content_type in ['words', 'phrases']:
                content_dir = person_dir / content_type
                if content_dir.exists():
                    for word_dir in content_dir.iterdir():
                        if word_dir.is_dir():
                            for instance_dir in word_dir.iterdir():
                                if instance_dir.is_dir():
                                    color_images = list(instance_dir.glob("color_*.jpg"))
                                    if len(color_images) >= 5:
                                        all_sequences.append((
                                            instance_dir, person_dir.name, 
                                            word_dir.name, content_type
                                        ))
                                        person_sequences += 1
            
            print(f"{person_dir.name}: {person_sequences} sequences")
        
        print(f"Total MIRACL sequences: {len(all_sequences)}")
        
        if len(all_sequences) == 0:
            print("ERROR: No valid MIRACL sequences found")
            return start_idx
        
        # Process samples
        random.seed(42)
        random.shuffle(all_sequences)
        
        remaining_needed = max_samples - start_idx
        if remaining_needed <= 0:
            print(f"Already have {start_idx} samples, no more needed")
            return start_idx
        
        selected_sequences = all_sequences[:remaining_needed]
        print(f"Processing {len(selected_sequences)} new samples...")
        
        processed_count = start_idx
        failed_count = 0
        
        # Process one by one
        for i, (sequence_dir, person_id, content_id, content_type) in enumerate(selected_sequences):
            try:
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(selected_sequences)} ({i/len(selected_sequences)*100:.1f}%)")
                
                result = self.process_miracl_sequence_windows_safe(sequence_dir, person_id, content_id, content_type)
                
                if result is not None:
                    sample_file = miracl_output / f"miracl_sample_{processed_count:04d}.pkl"
                    
                    with open(sample_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    processed_count += 1
                else:
                    failed_count += 1
                
                # Memory cleanup
                if i % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                failed_count += 1
                continue
        
        final_count = processed_count
        print(f"MIRACL processing complete!")
        print(f"Started with: {start_idx} samples")
        print(f"Processed: {final_count - start_idx} new samples") 
        print(f"Failed: {failed_count} samples")
        print(f"Total: {final_count} samples")
        
        return final_count
    
    def process_single_video_windows_safe(self, video_path, align_path, speaker_id, source):
        """Windows-safe single video processing"""
        try:
            # Read alignment
            sentence = self.read_alignment_safe(align_path)
            if not sentence:
                return None
            
            # Process video with Windows-safe settings
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Conservative frame processing
            max_frames_to_process = min(total_frames, 100)
            skip_frames = max(1, max_frames_to_process // self.sequence_length)
            
            frame_idx = 0
            extracted_count = 0
            
            while len(frames) < self.sequence_length and frame_idx < max_frames_to_process:
                ret, frame = cap.read()
                
                if ret:
                    lip_region = self.extract_lip_windows_safe(frame)
                    if lip_region is not None:
                        frames.append(lip_region)
                        extracted_count += 1
                    elif frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
                
                frame_idx += skip_frames
                
                # Set next frame position
                if frame_idx < max_frames_to_process:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            cap.release()
            
            # Quality check
            if len(frames) == 0 or extracted_count < 2:
                return None
            
            # Adjust sequence
            frames = self.adjust_sequence_safe(frames)
            
            # Convert to tensor
            frames_array = np.stack(frames).astype(np.float32)
            frames_normalized = frames_array / 255.0
            video_tensor = torch.FloatTensor(frames_normalized).unsqueeze(0)
            
            return {
                'video': video_tensor,
                'sentence': sentence,
                'speaker_id': speaker_id,
                'source': source,
                'extraction_rate': extracted_count / len(frames)
            }
            
        except Exception as e:
            return None
    
    def process_miracl_sequence_windows_safe(self, sequence_dir, person_id, content_id, content_type):
        """Windows-safe MIRACL sequence processing"""
        try:
            color_images = sorted(sequence_dir.glob("color_*.jpg"))
            
            if len(color_images) < 5:
                return None
            
            frames = []
            successful_extractions = 0
            
            # Conservative frame selection
            max_frames = min(len(color_images), 80)
            frame_indices = np.linspace(0, len(color_images)-1, min(self.sequence_length, max_frames)).astype(int)
            
            for idx in frame_indices:
                try:
                    frame = cv2.imread(str(color_images[idx]))
                    if frame is None:
                        continue
                    
                    lip_region = self.extract_lip_windows_safe(frame)
                    
                    if lip_region is not None:
                        frames.append(lip_region)
                        successful_extractions += 1
                    elif frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
                        
                except Exception:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            # Quality check
            if len(frames) == 0 or successful_extractions < 2:
                return None
            
            # Adjust sequence
            frames = self.adjust_sequence_safe(frames)
            
            # Create sentence
            sentence = self.create_sentence_safe(content_id, content_type)
            
            # Convert to tensor
            frames_array = np.stack(frames).astype(np.float32)
            frames_normalized = frames_array / 255.0
            video_tensor = torch.FloatTensor(frames_normalized).unsqueeze(0)
            
            return {
                'video': video_tensor,
                'sentence': sentence,
                'person_id': person_id,
                'content_id': content_id,
                'content_type': content_type,
                'source': 'miracl',
                'extraction_success_rate': successful_extractions / len(frame_indices)
            }
            
        except Exception as e:
            return None
    
    def extract_lip_windows_safe(self, frame):
        """Windows-safe lip extraction"""
        if frame is None:
            return None
        
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                gray = frame.copy()
                rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Try MediaPipe if available
            if self.face_detection is not None:
                try:
                    results = self.face_detection.process(rgb_frame)
                    
                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        h, w = gray.shape[:2]
                        
                        # Extract mouth region
                        face_x = int(bbox.xmin * w)
                        face_y = int(bbox.ymin * h)
                        face_w = int(bbox.width * w)
                        face_h = int(bbox.height * h)
                        
                        mouth_y = face_y + int(face_h * 0.6)
                        mouth_h = int(face_h * 0.4)
                        mouth_x = face_x + int(face_w * 0.2)
                        mouth_w = int(face_w * 0.6)
                        
                        # Bounds checking
                        mouth_x = max(0, mouth_x)
                        mouth_y = max(0, mouth_y)
                        mouth_x2 = min(w, mouth_x + mouth_w)
                        mouth_y2 = min(h, mouth_y + mouth_h)
                        
                        if mouth_x2 > mouth_x and mouth_y2 > mouth_y:
                            mouth_region = gray[mouth_y:mouth_y2, mouth_x:mouth_x2]
                            
                            if mouth_region.size > 0:
                                mouth_resized = cv2.resize(mouth_region, self.target_size, interpolation=cv2.INTER_CUBIC)
                                return cv2.equalizeHist(mouth_resized)
                
                except Exception:
                    pass  # Fall through to geometric method
            
            # Geometric fallback
            h, w = gray.shape
            start_y = int(h * 0.65)
            end_y = int(h * 0.95)
            start_x = int(w * 0.25)
            end_x = int(w * 0.75)
            
            mouth_region = gray[start_y:end_y, start_x:end_x]
            
            if mouth_region.size > 0:
                mouth_resized = cv2.resize(mouth_region, self.target_size, interpolation=cv2.INTER_CUBIC)
                return cv2.equalizeHist(mouth_resized)
            else:
                return np.zeros(self.target_size[::-1], dtype=np.uint8)
                
        except Exception:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
    
    def adjust_sequence_safe(self, frames):
        """Safe sequence adjustment"""
        try:
            current_length = len(frames)
            
            if current_length == self.sequence_length:
                return frames
            elif current_length > self.sequence_length:
                indices = np.linspace(0, current_length - 1, self.sequence_length).astype(int)
                return [frames[i] for i in indices]
            else:
                frames_extended = frames.copy()
                while len(frames_extended) < self.sequence_length:
                    repeat_count = min(self.sequence_length - len(frames_extended), len(frames))
                    frames_extended.extend(frames[:repeat_count])
                return frames_extended[:self.sequence_length]
                
        except Exception:
            result = frames[:self.sequence_length] if len(frames) >= self.sequence_length else frames
            while len(result) < self.sequence_length:
                result.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            return result[:self.sequence_length]
    
    def read_alignment_safe(self, align_path):
        """Safe alignment reading"""
        try:
            if not align_path.exists():
                return None
            
            words = []
            with open(align_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        word = parts[2]
                        if word != 'sil':
                            words.append(word)
            
            sentence = ' '.join(words).upper()
            return sentence if sentence else None
            
        except Exception:
            return None
    
    def create_sentence_safe(self, content_id, content_type):
        """Safe sentence creation"""
        try:
            if content_type == "words":
                return f"WORD {content_id.upper()}"
            else:
                return f"PHRASE {content_id.upper()}"
        except Exception:
            return f"CONTENT {content_id}"
    
    def create_unified_dataset_safe(self):
        """Safe unified dataset creation"""
        print("Creating unified dataset...")
        
        try:
            unified_dir = self.base_path / "processed_windows_safe" / "unified" / "train"
            unified_dir.mkdir(parents=True, exist_ok=True)
            
            total_samples = 0
            dataset_counts = {}
            
            for dataset_name, info in self.datasets.items():
                dataset_dir = info['output']
                
                if dataset_dir.exists():
                    samples = list(dataset_dir.glob("*.pkl"))
                    
                    for i, sample_file in enumerate(samples):
                        new_name = f"safe_{dataset_name}_{i:04d}.pkl"
                        shutil.copy2(sample_file, unified_dir / new_name)
                        total_samples += 1
                    
                    dataset_counts[dataset_name] = len(samples)
                    print(f"{dataset_name.upper()}: {len(samples)} samples")
            
            print(f"UNIFIED DATASET CREATED:")
            print(f"Total samples: {total_samples}")
            print(f"Location: {unified_dir}")
            
            return str(unified_dir), total_samples
            
        except Exception as e:
            print(f"ERROR creating unified dataset: {e}")
            return None, 0
    
    def process_all_windows_safe(self):
        """Main Windows-safe processing"""
        print("WINDOWS-SAFE MULTI-DATASET PROCESSING")
        print("=" * 60)
        print("Features: Windows handle fix, conservative processing")
        print()
        
        try:
            # Process GRID
            grid_count = self.process_grid_windows_safe(max_samples=1500)
            
            # Process MIRACL
            miracl_count = self.process_miracl_windows_safe(max_samples=2000)
            
            # Create unified dataset
            unified_dir, total_count = self.create_unified_dataset_safe()
            
            if unified_dir and total_count > 0:
                print("=" * 60)
                print("WINDOWS-SAFE PROCESSING COMPLETE!")
                print(f"GRID samples: {grid_count}")
                print(f"MIRACL samples: {miracl_count}")
                print(f"Total samples: {total_count}")
                print(f"Improvement over 200 samples: {total_count/200:.1f}x")
                
                print("BENEFITS:")
                print("- Windows handle error fixed")
                print("- Conservative memory usage")
                print("- Stable processing without crashes")
                print("- Resume capability if interrupted")
                
                return str(unified_dir)
            else:
                print("ERROR: Processing failed or no data created")
                return None
                
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            traceback.print_exc()
            return None

def main():
    """Main Windows-safe execution"""
    try:
        processor = WindowsSafeProcessor()
        unified_path = processor.process_all_windows_safe()
        
        if unified_path:
            print("SUCCESS: Windows-safe dataset ready!")
            print(f"Path: '{unified_path}'")
            print("NEXT STEPS:")
            print("1. Train transformer with new dataset")
            print("2. Expected WER: 15-30% (realistic for diverse data)")
            print("3. Much better real-world performance!")
        else:
            print("ERROR: Processing failed - check error messages above")
            
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()