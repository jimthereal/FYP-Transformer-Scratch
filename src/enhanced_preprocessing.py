"""
Phase 2A: Enhanced Preprocessing Framework
FYP Enhancement - Jimmy Yeow Kai Jim

Advanced preprocessing pipeline to bridge domain gap between GRID corpus and real-world videos.
"""

import cv2
import numpy as np
import torch
import dlib
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLipPreprocessor:
    """
    Enhanced preprocessing pipeline for robust lip-reading across various conditions.
    
    Phase 2A Improvements:
    1. Robust mouth detection with multiple fallback strategies
    2. Advanced data augmentation for better generalization
    3. Adaptive normalization based on video characteristics
    4. Multi-condition handling (lighting, quality, angles)
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (128, 64),
                 use_augmentation: bool = True,
                 face_detector_path: str = 'shape_predictor_68_face_landmarks.dat'):
        
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        
        # Initialize face detection
        self.face_detector = dlib.get_frontal_face_detector()
        
        try:
            self.landmark_predictor = dlib.shape_predictor(face_detector_path)
            self.use_landmarks = True
            logger.info("✅ Using dlib landmarks for precise mouth detection")
        except:
            self.use_landmarks = False
            logger.warning("⚠️ No landmarks detector - using backup mouth detection")
        
        # Setup data augmentation pipeline
        self.setup_augmentation()
        
        # Mouth detection strategies (fallback system)
        self.mouth_strategies = [
            self._extract_mouth_landmarks,      # Best: Face landmarks
            self._extract_mouth_cascade,        # Good: Haar cascades
            self._extract_mouth_center_crop,    # Fallback: Center crop
        ]
    
    def setup_augmentation(self):
        """Setup advanced data augmentation pipeline"""
        
        if self.use_augmentation:
            self.augmentation = A.Compose([
                # Geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1, 
                    rotate_limit=5,
                    p=0.5
                ),
                
                # Lighting augmentations (crucial for real-world robustness)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.3
                ),
                
                # Color augmentations
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                
                # Noise and blur (simulate poor video quality)
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.2),
                ], p=0.3),
                
                # Normalize to [0, 1]
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
            ])
        else:
            # Simple normalization without augmentation
            self.augmentation = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
            ])
    
    def _extract_mouth_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth using facial landmarks (most precise)"""
        if not self.use_landmarks:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None
            
        face = faces[0]
        landmarks = self.landmark_predictor(gray, face)
        
        # Extract mouth landmarks (points 48-67)
        mouth_points = []
        for i in range(48, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            mouth_points.append([x, y])
        
        mouth_points = np.array(mouth_points)
        
        # Get bounding box with adaptive padding
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # Adaptive padding based on mouth size
        mouth_width = x_max - x_min
        mouth_height = y_max - y_min
        padding_x = int(mouth_width * 0.3)  # 30% padding
        padding_y = int(mouth_height * 0.4)  # 40% padding
        
        # Apply padding with bounds checking
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(frame.shape[1], x_max + padding_x)
        y_max = min(frame.shape[0], y_max + padding_y)
        
        mouth_region = frame[y_min:y_max, x_min:x_max]
        
        # Quality check - ensure reasonable size
        if mouth_region.shape[0] < 20 or mouth_region.shape[1] < 20:
            return None
            
        return mouth_region
    
    def _extract_mouth_cascade(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth using Haar cascade (backup method)"""
        try:
            # Load mouth cascade classifier
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect mouth region
            mouths = mouth_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(mouths) > 0:
                # Use the largest detected mouth
                x, y, w, h = max(mouths, key=lambda m: m[2] * m[3])
                
                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                return frame[y:y+h, x:x+w]
                
        except Exception as e:
            logger.warning(f"Haar cascade mouth detection failed: {e}")
            
        return None
    
    def _extract_mouth_center_crop(self, frame: np.ndarray) -> np.ndarray:
        """Extract mouth using center crop (ultimate fallback)"""
        h, w = frame.shape[:2]
        
        # Assume mouth is in bottom half, center region
        y_start = int(h * 0.6)  # Bottom 40%
        y_end = int(h * 0.9)
        x_start = int(w * 0.25)  # Center 50%
        x_end = int(w * 0.75)
        
        return frame[y_start:y_end, x_start:x_end]
    
    def extract_mouth_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract mouth region using hierarchical fallback strategy
        """
        for i, strategy in enumerate(self.mouth_strategies):
            try:
                mouth_region = strategy(frame)
                if mouth_region is not None:
                    if i > 0:  # Log if using fallback
                        logger.info(f"Using fallback strategy {i+1} for mouth detection")
                    return mouth_region
            except Exception as e:
                logger.warning(f"Mouth detection strategy {i+1} failed: {e}")
                continue
        
        # Should never reach here due to center crop fallback
        raise ValueError("All mouth detection strategies failed")
    
    def adaptive_normalization(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply adaptive normalization based on frame characteristics
        """
        # Analyze frame characteristics
        mean_brightness = np.mean(frame)
        std_brightness = np.std(frame)
        
        # Adaptive histogram equalization for low contrast
        if std_brightness < 30:  # Low contrast threshold
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Brightness adjustment for very dark/bright videos
        if mean_brightness < 80:  # Too dark
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
        elif mean_brightness > 180:  # Too bright
            frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-10)
        
        return frame
    
    def preprocess_frame(self, frame: np.ndarray, apply_augmentation: bool = None) -> np.ndarray:
        """
        Enhanced preprocessing for a single frame
        """
        if apply_augmentation is None:
            apply_augmentation = self.use_augmentation
        
        # Step 1: Adaptive normalization
        frame = self.adaptive_normalization(frame)
        
        # Step 2: Extract mouth region
        mouth_region = self.extract_mouth_region(frame)
        
        # Step 3: Convert BGR to RGB
        mouth_rgb = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB)
        
        # Step 4: Resize to target size
        mouth_resized = cv2.resize(mouth_rgb, self.target_size)
        
        # Step 5: Apply augmentation if enabled
        if apply_augmentation:
            augmented = self.augmentation(image=mouth_resized)
            return augmented['image']
        else:
            # Simple normalization
            normalized = mouth_resized.astype(np.float32) / 255.0
            return normalized
    
    def preprocess_video_enhanced(self, video_path: str, target_frames: int = 75) -> torch.Tensor:
        """
        Enhanced video preprocessing with improved robustness
        """
        logger.info(f"🎬 Enhanced preprocessing: {video_path}")
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames found in video")
        
        logger.info(f"📊 Video info: {len(frames)} frames at {frames[0].shape}")
        
        # Process each frame
        processed_frames = []
        success_count = 0
        
        for i, frame in enumerate(frames):
            try:
                processed_frame = self.preprocess_frame(frame, apply_augmentation=False)
                processed_frames.append(processed_frame)
                success_count += 1
            except Exception as e:
                logger.warning(f"Frame {i} processing failed: {e}")
                # Use previous frame if available
                if processed_frames:
                    processed_frames.append(processed_frames[-1])
                else:
                    # Skip this frame
                    continue
        
        logger.info(f"✅ Successfully processed {success_count}/{len(frames)} frames")
        
        # Handle frame count (pad or truncate to target_frames)
        if len(processed_frames) > target_frames:
            # Smart sampling instead of simple truncation
            indices = np.linspace(0, len(processed_frames)-1, target_frames, dtype=int)
            processed_frames = [processed_frames[i] for i in indices]
        elif len(processed_frames) < target_frames:
            # Pad with last frame
            last_frame = processed_frames[-1]
            while len(processed_frames) < target_frames:
                processed_frames.append(last_frame)
        
        # Convert to tensor [1, 3, 75, 64, 128]
        if isinstance(processed_frames[0], np.ndarray):
            # Manual normalization path
            video_array = np.array(processed_frames)  # [75, 64, 128, 3]
            video_array = video_array.transpose(3, 0, 1, 2)  # [3, 75, 64, 128]
            video_array = video_array[np.newaxis, ...]  # [1, 3, 75, 64, 128]
            video_tensor = torch.from_numpy(video_array).float()
        else:
            # Augmentation pipeline returns normalized tensors
            video_tensor = torch.stack(processed_frames).unsqueeze(0)  # [1, 75, 3, 64, 128]
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [1, 3, 75, 64, 128]
        
        logger.info(f"✅ Final tensor shape: {video_tensor.shape}")
        return video_tensor

def test_enhanced_preprocessing():
    """Test the enhanced preprocessing pipeline"""
    
    print("🧪 Testing Enhanced Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize enhanced preprocessor
    preprocessor = EnhancedLipPreprocessor(use_augmentation=True)
    
    # Test video path
    video_path = "data/test_videos/id23_6000_priazn.mpg"
    
    try:
        # Process with enhanced pipeline
        video_tensor = preprocessor.preprocess_video_enhanced(video_path)
        
        print(f"✅ Enhanced preprocessing successful!")
        print(f"📊 Output shape: {video_tensor.shape}")
        print(f"📈 Tensor range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
        print(f"📊 Tensor mean: {video_tensor.mean():.3f}")
        
        # Test with model
        from src.models.lipnet_base import LipNet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LipNet()
        model_path = "models/pretrained/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        
        # Get prediction
        with torch.no_grad():
            video_tensor = video_tensor.to(device)
            outputs = model(video_tensor)
            
        # Simple decoding
        predicted_ids = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()
        
        # Convert to text (basic decoding)
        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        decoded_chars = []
        prev_char = -1
        
        for char_idx in predicted_ids:
            if char_idx != 27 and char_idx != prev_char and char_idx < 27:  # Skip blank and duplicates
                decoded_chars.append(chars[char_idx])
                prev_char = char_idx
        
        prediction = ''.join(decoded_chars)
        
        print(f"\n🎯 ENHANCED PREDICTION: '{prediction}'")
        print(f"📝 Compare with baseline to see improvement!")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_enhanced_preprocessing()