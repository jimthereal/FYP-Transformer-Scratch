"""
MediaPipe-based Lip Extraction Preprocessor
Uses Google's MediaPipe for accurate face detection and lip landmark extraction
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

class MediaPipeLipPreprocessor:
    def __init__(self, project_root="C:/Users/Jimmy/APU/Year3 Sem2/LipNet-FYP"):
        self.project_root = Path(project_root)
        self.grid_dir = self.project_root / "data" / "GRID"
        self.video_dir = self.grid_dir / "video"
        self.align_dir = self.grid_dir / "align"
        self.processed_dir = self.grid_dir / "processed_mediapipe"  # New directory
        
        print("MediaPipe Lip Extraction Preprocessor")
        print(f"Project: {self.project_root}")
        
        # Parameters
        self.target_size = (64, 128)  # Width, Height for lips
        self.sequence_length = 75
        
        # Initialize MediaPipe
        self.setup_mediapipe()
        
    def setup_mediapipe(self):
        """Initialize MediaPipe Face Mesh for precise lip detection"""
        try:
            # MediaPipe Face Mesh for detailed facial landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,  # More precise landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # MediaPipe Face Detection (backup)
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for close-range (< 2m), 1 for full-range
                min_detection_confidence=0.5
            )
            
            # Drawing utilities for visualization
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            print("✓ MediaPipe initialized successfully")
            
            # Define lip landmark indices (MediaPipe 468 landmark model)
            # Outer lip landmarks
            self.OUTER_LIP_INDICES = [
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
                269, 270, 267, 271, 272, 271, 268, 12, 15, 16, 17, 18, 200,
                199, 175, 0, 13, 82, 81, 80, 78
            ]
            
            # Inner lip landmarks  
            self.INNER_LIP_INDICES = [
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318
            ]
            
            # Simplified lip region (most reliable points)
            self.LIP_INDICES = [
                # Outer lip boundary
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                # Upper lip
                13, 82, 81, 80, 78,
                # Lower lip  
                14, 87, 178, 88, 95,
                # Corners
                61, 291
            ]
            
        except Exception as e:
            print(f"MediaPipe setup error: {e}")
            raise
    
    def extract_lip_landmarks(self, frame):
        """Extract precise lip landmarks using MediaPipe Face Mesh"""
        if frame is None:
            return None
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Process with Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract lip landmarks
            lip_points = []
            h, w = frame.shape[:2]
            
            for idx in self.LIP_INDICES:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    lip_points.append([x, y])
            
            if len(lip_points) > 4:  # Need at least 4 points for bounding box
                lip_points = np.array(lip_points)
                return lip_points
        
        return None
    
    def extract_lip_region_mediapipe(self, frame):
        """Extract lip region using MediaPipe landmarks"""
        if frame is None:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        # Convert to grayscale for final processing
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Get lip landmarks
        lip_points = self.extract_lip_landmarks(frame)
        
        if lip_points is not None:
            # Calculate bounding box with margin
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            
            # Add margin (50% bigger for context)
            lip_width = x_max - x_min
            lip_height = y_max - y_min
            margin_x = int(lip_width * 0.5)
            margin_y = int(lip_height * 0.5)
            
            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(gray.shape[1], x_max + margin_x)
            y_max = min(gray.shape[0], y_max + margin_y)
            
            # Extract lip region
            if x_max > x_min and y_max > y_min:
                lip_region = gray[y_min:y_max, x_min:x_max]
                
                if lip_region.size > 0:
                    # High-quality resize
                    lip_resized = cv2.resize(lip_region, self.target_size, 
                                           interpolation=cv2.INTER_CUBIC)
                    
                    # Enhance contrast using CLAHE
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                    lip_enhanced = clahe.apply(lip_resized)
                    
                    # Gentle noise reduction
                    lip_smooth = cv2.bilateralFilter(lip_enhanced, 5, 50, 50)
                    
                    return lip_smooth
        
        # Fallback to face detection
        return self.extract_lip_region_face_detection(frame)
    
    def extract_lip_region_face_detection(self, frame):
        """Fallback using MediaPipe Face Detection"""
        if frame is None:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        # Convert to RGB
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Detect face
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = gray.shape[:2]
            
            # Convert relative coordinates to absolute
            face_x = int(bbox.xmin * w)
            face_y = int(bbox.ymin * h)
            face_w = int(bbox.width * w)
            face_h = int(bbox.height * h)
            
            # Extract mouth region (bottom third of face)
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
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                    mouth_enhanced = clahe.apply(mouth_resized)
                    mouth_smooth = cv2.bilateralFilter(mouth_enhanced, 5, 50, 50)
                    
                    return mouth_smooth
        
        # Final fallback to geometric method
        return self.extract_mouth_region_geometric(gray)
    
    def extract_mouth_region_geometric(self, gray):
        """Geometric fallback method"""
        if gray is None:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        h, w = gray.shape
        
        # Improved geometric estimation
        start_y = int(h * 0.65)
        end_y = int(h * 0.95)
        start_x = int(w * 0.25)
        end_x = int(w * 0.75)
        
        mouth_region = gray[start_y:end_y, start_x:end_x]
        
        if mouth_region.size > 0:
            mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                     interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            mouth_enhanced = clahe.apply(mouth_resized)
            
            return mouth_enhanced
        else:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
    
    def test_mediapipe_quality(self, video_path, num_frames=5):
        """Test MediaPipe quality and save comparison images"""
        print(f"Testing MediaPipe quality: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("Cannot open video for testing")
            return False
        
        test_dir = self.project_root / "debug_mediapipe"
        test_dir.mkdir(exist_ok=True)
        
        landmark_success = 0
        face_detection_success = 0
        
        for frame_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Test landmark extraction
            lip_points = self.extract_lip_landmarks(frame)
            landmark_detected = lip_points is not None
            if landmark_detected:
                landmark_success += 1
            
            # Test face detection fallback
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            face_results = self.face_detection.process(rgb_frame)
            face_detected = face_results.detections is not None and len(face_results.detections) > 0
            if face_detected:
                face_detection_success += 1
            
            # Extract lip regions using different methods
            mediapipe_lip = self.extract_lip_region_mediapipe(frame)
            geometric_lip = self.extract_mouth_region_geometric(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            )
            
            # Create comparison image
            original_resized = cv2.resize(frame, (128, 128))
            
            # Draw landmarks on original if detected
            if landmark_detected and lip_points is not None:
                for point in lip_points:
                    cv2.circle(original_resized, 
                             (int(point[0] * 128 / frame.shape[1]), 
                              int(point[1] * 128 / frame.shape[0])), 
                             1, (0, 255, 0), -1)
            
            # Create side-by-side comparison
            comparison = np.hstack([
                original_resized,
                cv2.resize(cv2.cvtColor(mediapipe_lip, cv2.COLOR_GRAY2BGR), (128, 128)),
                cv2.resize(cv2.cvtColor(geometric_lip, cv2.COLOR_GRAY2BGR), (128, 128))
            ])
            
            cv2.imwrite(str(test_dir / f"mediapipe_comparison_frame_{frame_idx}.png"), comparison)
            
            print(f"Frame {frame_idx}:")
            print(f"  Landmarks detected: {landmark_detected}")
            print(f"  Face detected: {face_detected}")
            print(f"  MediaPipe lip: shape={mediapipe_lip.shape}, range={mediapipe_lip.min()}-{mediapipe_lip.max()}")
        
        cap.release()
        
        print(f"\nMediaPipe Test Results:")
        print(f"  Landmark detection rate: {landmark_success}/{num_frames} ({landmark_success/num_frames*100:.1f}%)")
        print(f"  Face detection rate: {face_detection_success}/{num_frames} ({face_detection_success/num_frames*100:.1f}%)")
        print(f"  Test images saved to: {test_dir}")
        
        return landmark_success > 0 or face_detection_success > 0
    
    def process_single_video(self, video_path, align_path):
        """Process single video with MediaPipe"""
        print(f"Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sentence = self.read_alignment_simple(align_path)
        print(f"  Video: {total_frames} frames, {fps} fps")
        print(f"  Sentence: '{sentence}'")
        
        frames = []
        landmark_detections = 0
        face_detections = 0
        geometric_fallbacks = 0
        
        for frame_idx in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track detection method used
            lip_points = self.extract_lip_landmarks(frame)
            if lip_points is not None:
                landmark_detections += 1
            else:
                # Check if face detection works
                if len(frame.shape) == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                face_results = self.face_detection.process(rgb_frame)
                if face_results.detections:
                    face_detections += 1
                else:
                    geometric_fallbacks += 1
            
            # Extract lip region
            lip_region = self.extract_lip_region_mediapipe(frame)
            frames.append(lip_region)
        
        cap.release()
        
        total_processed = len(frames)
        print(f"  Extracted {total_processed} frames")
        print(f"  Landmark detections: {landmark_detections} ({landmark_detections/total_processed*100:.1f}%)")
        print(f"  Face detections: {face_detections} ({face_detections/total_processed*100:.1f}%)")
        print(f"  Geometric fallbacks: {geometric_fallbacks} ({geometric_fallbacks/total_processed*100:.1f}%)")
        
        if not frames:
            return None
        
        # Pad/truncate to fixed length
        while len(frames) < self.sequence_length:
            frames.append(frames[-1].copy())
        frames = frames[:self.sequence_length]
        
        # Advanced normalization
        frames_array = np.stack(frames).astype(np.float32)
        normalized_frames = []
        
        for frame in frames_array:
            # Per-frame normalization for better contrast
            frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)
            frame_norm = np.clip(frame_norm, -2, 2)
            frame_norm = (frame_norm + 2) / 4  # Scale to [0, 1]
            normalized_frames.append(frame_norm)
        
        frames_array = np.stack(normalized_frames)
        video_tensor = torch.FloatTensor(frames_array).unsqueeze(0)  # [1, T, H, W]
        
        print(f"  Final tensor: {video_tensor.shape}")
        print(f"  Tensor range: {video_tensor.min():.3f} - {video_tensor.max():.3f}")
        
        quality_score = (landmark_detections + face_detections) / total_processed
        
        return {
            'video': video_tensor,
            'sentence': sentence,
            'video_path': str(video_path),
            'fps': fps,
            'landmark_detections': landmark_detections,
            'face_detections': face_detections,
            'geometric_fallbacks': geometric_fallbacks,
            'quality_score': quality_score
        }
    
    def read_alignment_simple(self, align_path):
        """Read alignment file"""
        if not align_path.exists():
            return f"SEQUENCE {align_path.stem.upper()}"
        
        try:
            words = []
            with open(align_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        word = parts[2]
                        if word != 'sil':
                            words.append(word)
            
            sentence = ' '.join(words).upper()
            return sentence if sentence else "UNKNOWN"
            
        except Exception as e:
            print(f"  Warning: Error reading alignment: {e}")
            return f"SEQUENCE {align_path.stem.upper()}"
    
    def scan_and_process(self, max_samples=50):
        """Scan and process with MediaPipe"""
        print("Scanning GRID dataset with MediaPipe...")
        
        # Find all videos
        all_videos = []
        for speaker_dir in self.video_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                video_files = list(speaker_dir.glob("*.mpg"))
                
                align_dir = self.align_dir / f"{speaker_id}_align"
                if not align_dir.exists():
                    align_dir = self.align_dir / speaker_id
                
                for video_file in video_files:
                    align_file = align_dir / (video_file.stem + '.align')
                    all_videos.append((video_file, align_file, speaker_id))
        
        print(f"Found {len(all_videos)} total videos")
        
        # Test MediaPipe on first video
        if all_videos:
            test_video, _, _ = all_videos[0]
            print(f"\nTesting MediaPipe quality...")
            self.test_mediapipe_quality(test_video)
        
        # Process samples
        sample_videos = all_videos[:max_samples]
        print(f"\nProcessing {len(sample_videos)} samples with MediaPipe...")
        
        train_dir = self.processed_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        total_quality = 0
        total_landmarks = 0
        total_faces = 0
        total_geometric = 0
        
        for i, (video_path, align_path, speaker_id) in enumerate(sample_videos):
            print(f"Processing {i+1}/{len(sample_videos)}: {video_path.name}")
            result = self.process_single_video(video_path, align_path)
            
            if result is not None:
                sample_file = train_dir / f"mediapipe_sample_{processed_count:04d}.pkl"
                with open(sample_file, 'wb') as f:
                    pickle.dump(result, f)
                
                total_quality += result['quality_score']
                total_landmarks += result['landmark_detections']
                total_faces += result['face_detections']
                total_geometric += result['geometric_fallbacks']
                
                processed_count += 1
                print(f"  Saved: {sample_file} (Quality: {result['quality_score']:.2f})")
            else:
                print(f"  Failed: {video_path}")
        
        if processed_count > 0:
            avg_quality = total_quality / processed_count
            total_frames = processed_count * self.sequence_length
            
            print(f"\nMediaPipe processing complete!")
            print(f"Successfully processed: {processed_count} samples")
            print(f"Average quality score: {avg_quality:.2f}")
            print(f"Detection statistics:")
            print(f"  Landmark detections: {total_landmarks}/{total_frames} ({total_landmarks/total_frames*100:.1f}%)")
            print(f"  Face detections: {total_faces}/{total_frames} ({total_faces/total_frames*100:.1f}%)")
            print(f"  Geometric fallbacks: {total_geometric}/{total_frames} ({total_geometric/total_frames*100:.1f}%)")
            print(f"Saved to: {train_dir}")
        
        return processed_count > 0

def main():
    print("MEDIAPIPE LIP EXTRACTION PREPROCESSING")
    print("=" * 50)
    
    try:
        preprocessor = MediaPipeLipPreprocessor()
        
        if not preprocessor.video_dir.exists():
            print(f"ERROR: GRID video directory not found")
            return
        
        # Start with smaller batch for testing
        success = preprocessor.scan_and_process(max_samples=20)
        
        if success:
            print("\nSUCCESS: MediaPipe preprocessing completed!")
            print("Check debug_mediapipe/ folder for quality comparison images")
            print("Next: Test with attention model using processed_mediapipe data")
        else:
            print("\nFAILED: Check MediaPipe setup")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install MediaPipe: pip install mediapipe")

if __name__ == "__main__":
    main()