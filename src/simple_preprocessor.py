"""
Simple Bulletproof GRID Preprocessor
No fancy face detection - just extract center region that contains mouth
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm

class SimpleBulletproofPreprocessor:
    def __init__(self, project_root="C:/Users/Jimmy/APU/Year3 Sem2/LipNet-FYP"):
        self.project_root = Path(project_root)
        self.grid_dir = self.project_root / "data" / "GRID"
        self.video_dir = self.grid_dir / "video"
        self.align_dir = self.grid_dir / "align"
        self.processed_dir = self.grid_dir / "processed_simple"  # New directory
        
        print("Simple Bulletproof GRID Preprocessor")
        print(f"Project: {self.project_root}")
        print(f"GRID: {self.grid_dir}")
        
        # Simple parameters - no fancy detection
        self.target_size = (64, 128)  # Smaller, easier to process
        self.sequence_length = 75
        
    def extract_mouth_region_simple(self, frame):
        """
        Extract mouth region using simple geometric approach
        No face detection - just crop lower center portion
        """
        if frame is None or frame.size == 0:
            return np.zeros(self.target_size, dtype=np.uint8)
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        h, w = gray.shape
        
        # Extract lower center region (where mouth typically is)
        # Take bottom 40% and center 60% of frame
        start_y = int(h * 0.6)  # Start from 60% down
        end_y = h              # Go to bottom
        start_x = int(w * 0.2)  # Start from 20% from left
        end_x = int(w * 0.8)    # End at 80% from left
        
        mouth_region = gray[start_y:end_y, start_x:end_x]
        
        # Resize to target size
        if mouth_region.size > 0:
            mouth_resized = cv2.resize(mouth_region, self.target_size)
            return mouth_resized
        else:
            return np.zeros(self.target_size, dtype=np.uint8)
    
    def test_video_reading(self, video_path):
        """Test if we can read a video and extract frames"""
        print(f"Testing video reading: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            return False
        
        # Read first few frames
        frames_read = 0
        while frames_read < 5:
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"Frame {frames_read}: shape={frame.shape}, dtype={frame.dtype}")
            print(f"  Min/Max: {frame.min()}/{frame.max()}")
            
            # Test mouth extraction
            mouth = self.extract_mouth_region_simple(frame)
            print(f"  Mouth region: shape={mouth.shape}, min/max={mouth.min()}/{mouth.max()}")
            
            # Save first frame for visual inspection
            if frames_read == 0:
                test_dir = self.project_root / "debug_test"
                test_dir.mkdir(exist_ok=True)
                
                cv2.imwrite(str(test_dir / "original_frame.png"), frame)
                cv2.imwrite(str(test_dir / "mouth_region.png"), mouth)
                print(f"  Saved test images to: {test_dir}")
            
            frames_read += 1
        
        cap.release()
        print(f"Successfully read {frames_read} frames")
        return frames_read > 0
    
    def process_single_video(self, video_path, align_path):
        """Process a single video with detailed logging"""
        print(f"Processing: {video_path.name}")
        
        # Read video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Video info: {total_frames} frames, {fps} fps")
        
        # Read alignment for sentence
        sentence = self.read_alignment_simple(align_path)
        print(f"  Sentence: '{sentence}'")
        
        # Extract frames
        frames = []
        frame_count = 0
        
        while frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                print(f"  Reached end of video at frame {frame_count}")
                break
            
            # Extract mouth region
            mouth_region = self.extract_mouth_region_simple(frame)
            frames.append(mouth_region)
            frame_count += 1
        
        cap.release()
        
        print(f"  Extracted {len(frames)} frames")
        
        if len(frames) == 0:
            print(f"  ERROR: No frames extracted")
            return None
        
        # Pad or truncate to fixed length
        while len(frames) < self.sequence_length:
            frames.append(frames[-1].copy())  # Repeat last frame
        
        frames = frames[:self.sequence_length]
        
        # Convert to tensor
        frames_array = np.stack(frames)  # [T, H, W]
        frames_array = frames_array.astype(np.float32) / 255.0  # Normalize
        
        # Add channel dimension: [T, H, W] -> [1, T, H, W]
        video_tensor = torch.FloatTensor(frames_array).unsqueeze(0)
        
        print(f"  Final tensor: {video_tensor.shape}")
        print(f"  Tensor min/max: {video_tensor.min():.3f}/{video_tensor.max():.3f}")
        print(f"  Tensor mean: {video_tensor.mean():.3f}")
        
        return {
            'video': video_tensor,
            'sentence': sentence,
            'video_path': str(video_path),
            'fps': fps,
            'original_frames': frame_count
        }
    
    def read_alignment_simple(self, align_path):
        """Read alignment file and create sentence"""
        if not align_path.exists():
            # Generate from filename if no alignment
            return self.generate_sentence_from_filename(align_path.stem)
        
        try:
            words = []
            with open(align_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        word = parts[2]
                        if word != 'sil':  # Skip silence markers
                            words.append(word)
            
            sentence = ' '.join(words).upper()
            return sentence if sentence else "UNKNOWN"
            
        except Exception as e:
            print(f"  Warning: Error reading alignment: {e}")
            return self.generate_sentence_from_filename(align_path.stem)
    
    def generate_sentence_from_filename(self, filename):
        """Simple filename to sentence mapping"""
        # This is a fallback - real alignments are better
        return f"SEQUENCE {filename.upper()}"
    
    def scan_and_process(self, max_samples=10):
        """Scan dataset and process a few samples for testing"""
        print("Scanning GRID dataset...")
        
        # Find all video files
        all_videos = []
        for speaker_dir in self.video_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                video_files = list(speaker_dir.glob("*.mpg"))
                
                # Find corresponding alignment directory
                align_dir = self.align_dir / f"{speaker_id}_align"
                if not align_dir.exists():
                    align_dir = self.align_dir / speaker_id
                
                for video_file in video_files:
                    align_file = align_dir / (video_file.stem + '.align')
                    all_videos.append((video_file, align_file, speaker_id))
        
        print(f"Found {len(all_videos)} total videos")
        
        # Test first video
        if all_videos:
            test_video, test_align, test_speaker = all_videos[0]
            print(f"\nTesting video reading with: {test_video}")
            
            if self.test_video_reading(test_video):
                print("Video reading test PASSED")
            else:
                print("Video reading test FAILED - check video files")
                return False
        
        # Process limited samples
        sample_videos = all_videos[:max_samples]
        print(f"\nProcessing {len(sample_videos)} samples...")
        
        # Create output directory
        train_dir = self.processed_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        for i, (video_path, align_path, speaker_id) in enumerate(tqdm(sample_videos)):
            result = self.process_single_video(video_path, align_path)
            
            if result is not None:
                # Save processed sample
                sample_file = train_dir / f"sample_{processed_count:04d}.pkl"
                with open(sample_file, 'wb') as f:
                    pickle.dump(result, f)
                
                print(f"  Saved: {sample_file}")
                processed_count += 1
            else:
                print(f"  Failed: {video_path}")
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count} samples")
        print(f"Saved to: {train_dir}")
        
        return processed_count > 0

def main():
    """Run simple preprocessing"""
    print("SIMPLE BULLETPROOF PREPROCESSING")
    print("=" * 50)
    
    preprocessor = SimpleBulletproofPreprocessor()
    
    # Check if GRID data exists
    if not preprocessor.video_dir.exists():
        print(f"ERROR: GRID video directory not found: {preprocessor.video_dir}")
        print("Please ensure GRID dataset is properly extracted")
        return
    
    # Run preprocessing
    success = preprocessor.scan_and_process(max_samples=5)  # Start with just 5 samples
    
    if success:
        print("\nSUCCESS: Preprocessing completed!")
        print("Next step: Test with diagnostic training")
    else:
        print("\nFAILED: Check video files and paths")

if __name__ == "__main__":
    main()