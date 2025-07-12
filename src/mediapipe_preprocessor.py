"""
Windows-Safe MediaPipe Preprocessor
Simplified version to avoid Windows handle issues
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set environment variables before importing MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import mediapipe as mp

class WindowsSafePreprocessor:
    def __init__(self, project_root="C:/Users/Jimmy/APU/Year3 Sem2/LipNet-FYP"):
        self.project_root = Path(project_root)
        self.grid_dir = self.project_root / "data" / "GRID"
        self.video_dir = self.grid_dir / "video"
        self.align_dir = self.grid_dir / "align"
        self.processed_dir = self.grid_dir / "processed_mediapipe_safe"
        
        print("Windows-Safe MediaPipe Preprocessor")
        print(f"Project: {self.project_root}")
        
        # Parameters
        self.target_size = (64, 128)
        self.sequence_length = 75
        
        # Initialize MediaPipe with safe settings
        self.setup_mediapipe_safe()
        
    def setup_mediapipe_safe(self):
        """Initialize MediaPipe with Windows-safe settings"""
        try:
            print("Initializing MediaPipe with safe settings...")
            
            # Simple face detection only (more stable on Windows)
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            # Optional face mesh (disable if causing issues)
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,  # Static mode for stability
                    max_num_faces=1,
                    refine_landmarks=False,  # Disable for stability
                    min_detection_confidence=0.5
                )
                self.use_face_mesh = True
                print("✓ Face mesh enabled")
            except:
                self.use_face_mesh = False
                print("! Face mesh disabled (using face detection only)")
            
            print("✓ MediaPipe initialized successfully")
            
        except Exception as e:
            print(f"MediaPipe setup error: {e}")
            raise
    
    def extract_lip_region_safe(self, frame):
        """Safe lip extraction with fallbacks"""
        if frame is None:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            gray = frame.copy()
            rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        try:
            # Try face detection first (more stable)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = gray.shape[:2]
                
                # Extract mouth region from face detection
                face_x = int(bbox.xmin * w)
                face_y = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)
                
                # Mouth region (bottom part of face)
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
                        # Resize and enhance
                        mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                                 interpolation=cv2.INTER_CUBIC)
                        
                        # Simple enhancement
                        mouth_enhanced = cv2.equalizeHist(mouth_resized)
                        
                        return mouth_enhanced
            
            # Try face mesh if available and face detection failed
            if self.use_face_mesh:
                try:
                    mesh_results = self.face_mesh.process(rgb_frame)
                    if mesh_results.multi_face_landmarks:
                        # Extract lip landmarks
                        landmarks = mesh_results.multi_face_landmarks[0]
                        lip_points = []
                        
                        # Key lip landmark indices
                        lip_indices = [61, 84, 17, 314, 405, 320, 13, 82, 81, 78, 14, 87, 178, 88, 95, 291]
                        
                        for idx in lip_indices:
                            if idx < len(landmarks.landmark):
                                landmark = landmarks.landmark[idx]
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                lip_points.append([x, y])
                        
                        if len(lip_points) > 4:
                            lip_points = np.array(lip_points)
                            x_min, y_min = lip_points.min(axis=0)
                            x_max, y_max = lip_points.max(axis=0)
                            
                            # Add margin
                            margin = 20
                            x_min = max(0, x_min - margin)
                            y_min = max(0, y_min - margin)
                            x_max = min(w, x_max + margin)
                            y_max = min(h, y_max + margin)
                            
                            if x_max > x_min and y_max > y_min:
                                lip_region = gray[y_min:y_max, x_min:x_max]
                                
                                if lip_region.size > 0:
                                    lip_resized = cv2.resize(lip_region, self.target_size, 
                                                           interpolation=cv2.INTER_CUBIC)
                                    lip_enhanced = cv2.equalizeHist(lip_resized)
                                    
                                    return lip_enhanced
                except:
                    pass  # Fall through to geometric method
            
        except Exception as e:
            print(f"    MediaPipe processing failed: {e}")
        
        # Geometric fallback
        return self.extract_geometric(gray)
    
    def extract_geometric(self, gray):
        """Geometric fallback method"""
        if gray is None:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
        
        h, w = gray.shape
        
        # Simple geometric mouth region
        start_y = int(h * 0.65)
        end_y = int(h * 0.95)
        start_x = int(w * 0.25)
        end_x = int(w * 0.75)
        
        mouth_region = gray[start_y:end_y, start_x:end_x]
        
        if mouth_region.size > 0:
            mouth_resized = cv2.resize(mouth_region, self.target_size, 
                                     interpolation=cv2.INTER_CUBIC)
            mouth_enhanced = cv2.equalizeHist(mouth_resized)
            
            return mouth_enhanced
        else:
            return np.zeros(self.target_size[::-1], dtype=np.uint8)
    
    def process_video_safe(self, video_path, align_path):
        """Safe video processing"""
        print(f"Processing: {video_path.name}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"  ERROR: Cannot open video")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            sentence = self.read_alignment(align_path)
            print(f"  Video: {total_frames} frames, {fps} fps")
            print(f"  Sentence: '{sentence}'")
            
            frames = []
            processed_count = 0
            
            # Process frames sequentially (safer than random access)
            while processed_count < self.sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    lip_region = self.extract_lip_region_safe(frame)
                    frames.append(lip_region)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"    Frame {processed_count} failed: {e}")
                    # Use blank frame for failed extractions
                    frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
                    processed_count += 1
            
            cap.release()
            
            print(f"  Extracted {len(frames)} frames")
            
            if not frames:
                return None
            
            # Pad/truncate to fixed length
            while len(frames) < self.sequence_length:
                frames.append(frames[-1].copy())
            frames = frames[:self.sequence_length]
            
            # Simple normalization
            frames_array = np.stack(frames).astype(np.float32)
            frames_normalized = frames_array / 255.0  # Simple [0,1] normalization
            
            video_tensor = torch.FloatTensor(frames_normalized).unsqueeze(0)  # [1, T, H, W]
            
            print(f"  Final tensor: {video_tensor.shape}")
            print(f"  Tensor range: {video_tensor.min():.3f} - {video_tensor.max():.3f}")
            
            return {
                'video': video_tensor,
                'sentence': sentence,
                'video_path': str(video_path),
                'fps': fps
            }
            
        except Exception as e:
            print(f"  ERROR: Video processing failed: {e}")
            return None
    
    def read_alignment(self, align_path):
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
    
    def process_dataset_safe(self, max_samples=100):
        """Safe dataset processing"""
        print("Processing GRID dataset safely...")
        
        # Find videos
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
        
        # Select samples
        selected_videos = all_videos[:max_samples]
        print(f"Processing {len(selected_videos)} samples...")
        
        # Create output directory
        train_dir = self.processed_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        for i, (video_path, align_path, speaker_id) in enumerate(selected_videos):
            print(f"\n[{i+1}/{len(selected_videos)}]")
            
            try:
                result = self.process_video_safe(video_path, align_path)
                
                if result is not None:
                    sample_file = train_dir / f"safe_sample_{processed_count:04d}.pkl"
                    
                    with open(sample_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    processed_count += 1
                    print(f"  ✓ Saved: {sample_file.name}")
                else:
                    print(f"  ✗ Failed: {video_path.name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {video_path.name}: {e}")
                continue
        
        print(f"\nSafe processing complete!")
        print(f"Successfully processed: {processed_count} samples")
        print(f"Saved to: {train_dir}")
        
        return processed_count > 0

def main():
    print("WINDOWS-SAFE MEDIAPIPE PREPROCESSING")
    print("=" * 50)
    
    try:
        preprocessor = WindowsSafePreprocessor()
        
        if not preprocessor.video_dir.exists():
            print(f"ERROR: GRID video directory not found")
            print(f"Expected path: {preprocessor.video_dir}")
            return
        
        print(f"Video directory found: {preprocessor.video_dir}")
        
        # Start with small batch
        success = preprocessor.process_dataset_safe(max_samples=100)
        
        if success:
            print("\nSUCCESS: Safe preprocessing completed!")
            print("You can now train with the enhanced model!")
            print("Data saved in: data/GRID/processed_mediapipe_safe/train/")
        else:
            print("\nFAILED: No samples processed successfully")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()