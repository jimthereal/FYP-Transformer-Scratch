"""
GRID-Only Dataset Preprocessor with Train/Test Split
Focuses exclusively on GRID corpus with proper data splitting
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
import random
import gc
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# MediaPipe setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

class GRIDOnlyProcessor:
    """GRID-focused processor with train/test split"""
    
    def __init__(self, base_path="data"):
        self.base_path = Path(base_path)
        self.grid_path = self.base_path / "GRID"
        self.output_path = self.base_path / "processed_grid_only"
        
        self.target_size = (64, 128)
        self.sequence_length = 75
        self.train_ratio = 0.8
        
        # Initialize MediaPipe
        self.mp_face_detection = None
        self.face_detection = None
        self.setup_mediapipe()
        
        print("GRID-Only Processor Initialized")
        print(f"GRID path: {self.grid_path}")
        print(f"Output path: {self.output_path}")
        print(f"Train/Test split: {self.train_ratio:.0%}/{(1-self.train_ratio):.0%}")
    
    def setup_mediapipe(self):
        """Setup MediaPipe for face detection"""
        try:
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
    
    def extract_lip_region(self, frame):
        """Extract lip region from frame"""
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
            
            # Try MediaPipe detection
            if self.face_detection is not None:
                try:
                    results = self.face_detection.process(rgb_frame)
                    
                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        h, w = gray.shape[:2]
                        
                        # Calculate mouth region
                        face_x = int(bbox.xmin * w)
                        face_y = int(bbox.ymin * h)
                        face_w = int(bbox.width * w)
                        face_h = int(bbox.height * h)
                        
                        # Mouth is in lower part of face
                        mouth_y = face_y + int(face_h * 0.6)
                        mouth_h = int(face_h * 0.4)
                        mouth_x = face_x + int(face_w * 0.2)
                        mouth_w = int(face_w * 0.6)
                        
                        # Ensure valid bounds
                        mouth_x = max(0, mouth_x)
                        mouth_y = max(0, mouth_y)
                        mouth_x2 = min(w, mouth_x + mouth_w)
                        mouth_y2 = min(h, mouth_y + mouth_h)
                        
                        if mouth_x2 > mouth_x and mouth_y2 > mouth_y:
                            mouth_region = gray[mouth_y:mouth_y2, mouth_x:mouth_x2]
                            
                            if mouth_region.size > 0:
                                mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                                         interpolation=cv2.INTER_CUBIC)
                                return cv2.equalizeHist(mouth_resized)
                
                except Exception:
                    pass  # Fall through to geometric method
            
            # Geometric fallback
            return self.extract_geometric(gray)
            
        except Exception:
            return self.extract_geometric(gray) if gray is not None else None
    
    def extract_geometric(self, gray):
        """Geometric fallback for lip extraction"""
        if gray is None:
            return None
        
        h, w = gray.shape
        
        # Approximate mouth region
        start_y = int(h * 0.65)
        end_y = int(h * 0.95)
        start_x = int(w * 0.25)
        end_x = int(w * 0.75)
        
        mouth_region = gray[start_y:end_y, start_x:end_x]
        
        if mouth_region.size > 0:
            mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                     interpolation=cv2.INTER_CUBIC)
            return cv2.equalizeHist(mouth_resized)
        else:
            return None
    
    def read_alignment(self, align_path):
        """Read GRID alignment file"""
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
            
        except Exception as e:
            print(f"Error reading alignment: {e}")
            return None
    
    def process_video(self, video_path, align_path, speaker_id):
        """Process single GRID video with better error handling"""
        try:
            # Read alignment first
            sentence = self.read_alignment(align_path)
            if not sentence:
                print(f"    Warning: Empty alignment for {video_path.name}")
                return None
            
            # Open video with error checking
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"    Error: Cannot open video {video_path.name}")
                # Try alternative codec
                cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print(f"    Error: Still cannot open video with FFMPEG")
                    return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                print(f"    Error: Video has 0 frames: {video_path.name}")
                cap.release()
                return None
            
            # Calculate frame sampling
            frame_indices = np.linspace(0, max(0, total_frames - 1), self.sequence_length).astype(int)
            
            frames = []
            successful_extractions = 0
            failed_reads = 0
            
            for idx, target_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    lip_region = self.extract_lip_region(frame)
                    if lip_region is not None:
                        frames.append(lip_region)
                        successful_extractions += 1
                    else:
                        # Use last valid frame or zeros
                        if frames:
                            frames.append(frames[-1].copy())
                        else:
                            frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
                else:
                    failed_reads += 1
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            cap.release()
            
            # Quality check
            if successful_extractions < self.sequence_length * 0.3:  # Less than 30% success
                print(f"    Warning: Low extraction rate ({successful_extractions}/{self.sequence_length}) for {video_path.name}")
                return None
            
            # Ensure correct length
            frames = frames[:self.sequence_length]
            while len(frames) < self.sequence_length:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            # Convert to tensor
            frames_array = np.stack(frames).astype(np.float32)
            frames_normalized = frames_array / 255.0
            video_tensor = torch.FloatTensor(frames_normalized).unsqueeze(0)  # [1, T, H, W]
            
            return {
                'video': video_tensor,
                'sentence': sentence,
                'speaker_id': speaker_id,
                'video_path': str(video_path),
                'fps': fps,
                'extraction_rate': successful_extractions / self.sequence_length
            }
            
        except Exception as e:
            print(f"    Error processing video {video_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def collect_all_videos(self):
        """Collect all GRID videos with alignments"""
        print("Collecting GRID videos...")
        
        video_dir = self.grid_path / "video"
        align_dir = self.grid_path / "align"
        
        if not video_dir.exists():
            print(f"ERROR: Video directory not found at {video_dir}")
            return []
        
        all_videos = []
        
        # Process each speaker
        for speaker_dir in sorted(video_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            print(f"Processing speaker {speaker_id}...")
            
            # Find corresponding align directory
            align_speaker_dir = align_dir / speaker_id
            if not align_speaker_dir.exists():
                align_speaker_dir = align_dir / f"{speaker_id}_align"
            
            if not align_speaker_dir.exists():
                print(f"  Warning: No alignment directory for {speaker_id}")
                continue
            
            # Collect videos - look for .mpg files
            video_files = list(speaker_dir.glob("*.mpg"))
            
            if len(video_files) == 0:
                # Try other common video formats just in case
                for ext in ['*.mp4', '*.avi', '*.mov']:
                    video_files.extend(list(speaker_dir.glob(ext)))
            
            video_files = sorted(video_files)
            valid_videos = 0
            
            for video_file in video_files:
                align_file = align_speaker_dir / (video_file.stem + '.align')
                
                if align_file.exists():
                    all_videos.append({
                        'video_path': video_file,
                        'align_path': align_file,
                        'speaker_id': speaker_id,
                        'video_id': video_file.stem
                    })
                    valid_videos += 1
                else:
                    print(f"    Warning: No alignment for {video_file.name}")
            
            print(f"  Found {len(video_files)} videos ({valid_videos} with alignments) for {speaker_id}")
        
        print(f"Total videos found: {len(all_videos)}")
        return all_videos
    
    def process_dataset(self, max_samples_per_speaker=None):
        """Process GRID dataset with train/test split"""
        # Create output directories
        train_dir = self.output_path / "train"
        test_dir = self.output_path / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all videos
        all_videos = self.collect_all_videos()
        
        if not all_videos:
            print("ERROR: No videos found!")
            return
        
        # Group by speaker for balanced splitting
        videos_by_speaker = {}
        for video_info in all_videos:
            speaker_id = video_info['speaker_id']
            if speaker_id not in videos_by_speaker:
                videos_by_speaker[speaker_id] = []
            videos_by_speaker[speaker_id].append(video_info)
        
        print(f"\nFound {len(videos_by_speaker)} speakers:")
        for speaker, videos in videos_by_speaker.items():
            print(f"  {speaker}: {len(videos)} videos")
        
        # Process each speaker with train/test split
        all_train_samples = []
        all_test_samples = []
        
        for speaker_id, speaker_videos in videos_by_speaker.items():
            print(f"\nProcessing speaker {speaker_id}...")
            
            # Limit samples if specified
            if max_samples_per_speaker:
                speaker_videos = speaker_videos[:max_samples_per_speaker]
            
            # Split this speaker's videos
            if len(speaker_videos) > 5:  # Need enough for meaningful split
                train_videos, test_videos = train_test_split(
                    speaker_videos, 
                    train_size=self.train_ratio,
                    random_state=42
                )
            else:
                # Too few videos, put all in train
                train_videos = speaker_videos
                test_videos = []
            
            print(f"  Train: {len(train_videos)}, Test: {len(test_videos)}")
            
            # Process train videos
            print(f"  Processing {len(train_videos)} training videos...")
            train_success = 0
            train_failed = 0
            
            for i, video_info in enumerate(train_videos):
                if i % 50 == 0:
                    print(f"    Progress: {i}/{len(train_videos)} ({train_success} successful, {train_failed} failed)")
                
                result = self.process_video(
                    video_info['video_path'],
                    video_info['align_path'],
                    video_info['speaker_id']
                )
                
                if result is not None:
                    all_train_samples.append(result)
                    train_success += 1
                else:
                    train_failed += 1
                    print(f"    Failed: {video_info['video_path'].name}")
            
            print(f"  Train complete: {train_success} successful, {train_failed} failed")
            
            # Process test videos
            print(f"  Processing {len(test_videos)} test videos...")
            test_success = 0
            test_failed = 0
            
            for i, video_info in enumerate(test_videos):
                if i % 50 == 0:
                    print(f"    Progress: {i}/{len(test_videos)} ({test_success} successful, {test_failed} failed)")
                
                result = self.process_video(
                    video_info['video_path'],
                    video_info['align_path'],
                    video_info['speaker_id']
                )
                
                if result is not None:
                    all_test_samples.append(result)
                    test_success += 1
                else:
                    test_failed += 1
                    print(f"    Failed: {video_info['video_path'].name}")
            
            print(f"  Test complete: {test_success} successful, {test_failed} failed")
            
            # Memory cleanup
            gc.collect()
        
        # Save all samples
        print("\nSaving processed samples...")
        
        # Save train samples
        for i, sample in enumerate(all_train_samples):
            sample_file = train_dir / f"grid_train_{i:05d}.pkl"
            with open(sample_file, 'wb') as f:
                pickle.dump(sample, f)
        
        # Save test samples
        for i, sample in enumerate(all_test_samples):
            sample_file = test_dir / f"grid_test_{i:05d}.pkl"
            with open(sample_file, 'wb') as f:
                pickle.dump(sample, f)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print(f"Train samples: {len(all_train_samples)}")
        print(f"Test samples: {len(all_test_samples)}")
        print(f"Total samples: {len(all_train_samples) + len(all_test_samples)}")
        print(f"Train/Test ratio: {len(all_train_samples)/(len(all_train_samples)+len(all_test_samples)):.1%}/{len(all_test_samples)/(len(all_train_samples)+len(all_test_samples)):.1%}")
        
        # Show vocabulary statistics
        self.analyze_vocabulary(all_train_samples, all_test_samples)
        
        return train_dir, test_dir
    
    def analyze_vocabulary(self, train_samples, test_samples):
        """Analyze vocabulary distribution"""
        print("\nVocabulary Analysis:")
        
        # Collect all sentences
        train_sentences = [s['sentence'] for s in train_samples]
        test_sentences = [s['sentence'] for s in test_samples]
        
        # Word frequency
        train_words = ' '.join(train_sentences).split()
        test_words = ' '.join(test_sentences).split()
        
        train_vocab = set(train_words)
        test_vocab = set(test_words)
        
        print(f"Train vocabulary size: {len(train_vocab)} unique words")
        print(f"Test vocabulary size: {len(test_vocab)} unique words")
        print(f"Vocabulary overlap: {len(train_vocab & test_vocab)} words")
        
        # Show most common words
        from collections import Counter
        train_counter = Counter(train_words)
        print("\nMost common words in training:")
        for word, count in train_counter.most_common(10):
            print(f"  {word}: {count}")
        
        # Check for test words not in train
        unseen_words = test_vocab - train_vocab
        if unseen_words:
            print(f"\nWarning: {len(unseen_words)} words in test not seen in train:")
            print(f"  {list(unseen_words)[:10]}...")

def main():
    """Main execution"""
    print("GRID-ONLY DATASET PROCESSING")
    print("="*60)
    
    processor = GRIDOnlyProcessor()
    
    # Process with optional limit per speaker
    # Set to None to process all videos, or a number to limit
    # For faster testing, you can set max_samples_per_speaker=100
    train_dir, test_dir = processor.process_dataset(max_samples_per_speaker=None)
    
    if train_dir and test_dir:
        print("\nSUCCESS! Dataset ready for training")
        print(f"Train data: {train_dir}")
        print(f"Test data: {test_dir}")
        print("\nNext steps:")
        print("1. Run training: python improved_transformer.py")
        print("2. Update Flask backend with new model")
        print("3. Open web interface for testing")
    else:
        print("\nERROR: Processing failed")

if __name__ == "__main__":
    main()