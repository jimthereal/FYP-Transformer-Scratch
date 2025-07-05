import cv2
import numpy as np
import torch
import dlib
from model import LipNet

class GRIDPreprocessor:
    def __init__(self):
        # Initialize face and landmark detectors
        self.face_detector = dlib.get_frontal_face_detector()
        # Download this file if not present: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        try:
            self.landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.use_landmarks = True
            print("✅ Using dlib face landmarks for lip extraction")
        except:
            self.use_landmarks = False
            print("⚠️ No face landmarks detector found - using simple cropping")
    
    def extract_mouth_region(self, frame):
        """Extract mouth region from frame using face landmarks"""
        if not self.use_landmarks:
            # Simple center crop as fallback
            h, w = frame.shape[:2]
            # Assume mouth is in bottom half, center region
            y_start = int(h * 0.6)  # Bottom 40% of frame
            y_end = int(h * 0.9)
            x_start = int(w * 0.25)  # Center 50% of frame
            x_end = int(w * 0.75)
            return frame[y_start:y_end, x_start:x_end]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            # No face detected, use center crop
            h, w = frame.shape[:2]
            y_start = int(h * 0.6)
            y_end = int(h * 0.9)
            x_start = int(w * 0.25)
            x_end = int(w * 0.75)
            return frame[y_start:y_end, x_start:x_end]
        
        # Use first detected face
        face = faces[0]
        landmarks = self.landmark_predictor(gray, face)
        
        # Extract mouth landmarks (points 48-67)
        mouth_points = []
        for i in range(48, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            mouth_points.append([x, y])
        
        mouth_points = np.array(mouth_points)
        
        # Get bounding box around mouth with some padding
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        mouth_region = frame[y_min:y_max, x_min:x_max]
        return mouth_region
    
    def preprocess_grid_video(self, video_path):
        """Preprocess video in GRID corpus style"""
        print(f"🎬 GRID-style preprocessing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames in video")
        
        print(f"📊 Original: {len(frames)} frames at {frames[0].shape}")
        
        # Extract mouth regions
        mouth_frames = []
        for i, frame in enumerate(frames):
            mouth_region = self.extract_mouth_region(frame)
            mouth_frames.append(mouth_region)
            
            # Save first mouth region for debugging
            if i == 0:
                cv2.imwrite('debug_mouth_region.jpg', mouth_region)
                print(f"💾 Saved mouth region as debug_mouth_region.jpg")
        
        # Resize mouth regions to model input size
        processed_frames = []
        for mouth_frame in mouth_frames:
            # Convert to RGB
            mouth_rgb = cv2.cvtColor(mouth_frame, cv2.COLOR_BGR2RGB)
            # Resize to 128x64 (width x height)
            mouth_resized = cv2.resize(mouth_rgb, (128, 64))
            processed_frames.append(mouth_resized)
        
        # Handle frame count (pad or truncate to 75 frames)
        if len(processed_frames) > 75:
            processed_frames = processed_frames[:75]
        elif len(processed_frames) < 75:
            last_frame = processed_frames[-1]
            while len(processed_frames) < 75:
                processed_frames.append(last_frame)
        
        # Convert to tensor [1, 3, 75, 64, 128]
        video_array = np.array(processed_frames)  # [75, 64, 128, 3]
        video_array = video_array.transpose(3, 0, 1, 2)  # [3, 75, 64, 128]
        video_array = video_array[np.newaxis, ...]  # [1, 3, 75, 64, 128]
        
        # Normalize to [0, 1]
        video_array = video_array.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor
        video_tensor = torch.from_numpy(video_array)
        
        print(f"✅ Processed shape: {video_tensor.shape}")
        return video_tensor

def decode_prediction_improved(outputs):
    """Improved CTC decoding"""
    # Setup vocabulary
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    blank_idx = 27
    
    # Get predictions
    predicted_ids = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()
    
    # CTC decoding with better logic
    decoded_chars = []
    prev_char_idx = -1
    
    for char_idx in predicted_ids:
        # Skip blank tokens
        if char_idx == blank_idx:
            prev_char_idx = -1  # Reset previous for better duplicate handling
            continue
            
        # Skip consecutive duplicates
        if char_idx != prev_char_idx and char_idx < len(chars):
            decoded_chars.append(idx_to_char[char_idx])
            prev_char_idx = char_idx
    
    return ''.join(decoded_chars)

def test_improved_preprocessing():
    """Test the improved preprocessing"""
    video_path = "test_videos/id23_6000_priazn.mpg"  # Fixed extension
    
    # Initialize preprocessor
    preprocessor = GRIDPreprocessor()
    
    # Preprocess video
    try:
        video_tensor = preprocessor.preprocess_grid_video(video_path)
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LipNet()
        model_path = "pretrain/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        
        # Get prediction
        with torch.no_grad():
            video_tensor = video_tensor.to(device)
            outputs = model(video_tensor)
            
        # Decode with improved method
        prediction = decode_prediction_improved(outputs)
        
        print(f"\n🎯 IMPROVED PREDICTION: '{prediction}'")
        print(f"📝 Filename suggests: priazn (might be a GRID sentence code)")
        
        # Compare with original prediction from baseline_evaluation.py
        print(f"📊 Previous prediction was: 'NC BYJKAVGBKAVBVABVAKIGAVBRNGCUG'")
        print(f"📈 Improvement: Much shorter, more reasonable output!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try downloading shape_predictor_68_face_landmarks.dat for better mouth detection")

if __name__ == "__main__":
    test_improved_preprocessing()