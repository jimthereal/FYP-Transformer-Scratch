# src/final_enhanced_lipnet.py
# Phase 2B Final: Complete Enhanced LipNet System
# FYP Enhancement: Integrated system with all necessary components

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import os
import re
from collections import Counter
import dlib
from scipy.spatial import distance as dist

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lipnet_base import LipNet

class AdvancedPreprocessor:
    """
    Advanced video preprocessing with multiple strategies
    - Face detection and mouth region extraction
    - Adaptive normalization
    - Data augmentation
    - Quality assessment
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize face detection methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Try to load dlib predictor (optional)
        self.predictor = None
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        if os.path.exists(predictor_path):
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            print("✅ Using dlib landmarks for mouth detection")
        else:
            print("⚠️  Dlib landmarks not found, using fallback methods")
    
    def process_video(self, video_path):
        """
        Complete video preprocessing pipeline
        
        Args:
            video_path (str): Path to input video
            
        Returns:
            torch.Tensor: Preprocessed video tensor [1, 3, 75, 64, 128]
        """
        try:
            print(f"📹 Processing video: {video_path}")
            
            # Step 1: Load video frames
            frames = self._load_video_frames(video_path)
            
            # Step 2: Ensure 75 frames
            frames = self._normalize_frame_count(frames, target_count=75)
            
            # Step 3: Extract mouth regions
            mouth_regions = self._extract_mouth_regions(frames)
            
            # Step 4: Apply preprocessing
            processed_frames = self._apply_preprocessing(mouth_regions)
            
            # Step 5: Convert to tensor
            video_tensor = self._frames_to_tensor(processed_frames)
            
            print(f"✅ Preprocessed video shape: {video_tensor.shape}")
            return video_tensor
            
        except Exception as e:
            print(f"❌ Error in video preprocessing: {e}")
            raise
    
    def _load_video_frames(self, video_path):
        """Load all frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames found in video")
        
        print(f"📊 Loaded {len(frames)} frames")
        return frames
    
    def _normalize_frame_count(self, frames, target_count=75):
        """Ensure exactly target_count frames"""
        if len(frames) > target_count:
            # Take middle frames
            start_idx = (len(frames) - target_count) // 2
            frames = frames[start_idx:start_idx + target_count]
        elif len(frames) < target_count:
            # Repeat last frame
            while len(frames) < target_count:
                frames.append(frames[-1])
        
        return frames
    
    def _extract_mouth_regions(self, frames):
        """Extract mouth regions using multiple strategies"""
        mouth_regions = []
        
        for i, frame in enumerate(frames):
            try:
                # Strategy 1: Dlib landmarks (most accurate)
                if self.predictor:
                    mouth_region = self._extract_mouth_dlib(frame)
                    if mouth_region is not None:
                        mouth_regions.append(mouth_region)
                        continue
                
                # Strategy 2: Haar cascade mouth detection
                mouth_region = self._extract_mouth_haar(frame)
                if mouth_region is not None:
                    mouth_regions.append(mouth_region)
                    continue
                
                # Strategy 3: Face detection + mouth estimation
                mouth_region = self._extract_mouth_face_based(frame)
                if mouth_region is not None:
                    mouth_regions.append(mouth_region)
                    continue
                
                # Strategy 4: Center crop (fallback)
                mouth_region = self._extract_mouth_center_crop(frame)
                mouth_regions.append(mouth_region)
                
            except Exception as e:
                print(f"⚠️  Frame {i}: Using fallback method due to {e}")
                mouth_region = self._extract_mouth_center_crop(frame)
                mouth_regions.append(mouth_region)
        
        return mouth_regions
    
    def _extract_mouth_dlib(self, frame):
        """Extract mouth using dlib landmarks"""
        if not self.predictor:
            return None
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.detector(gray)
            
            if len(faces) > 0:
                # Use first detected face
                face = faces[0]
                landmarks = self.predictor(gray, face)
                
                # Mouth landmarks (points 48-67)
                mouth_points = []
                for i in range(48, 68):
                    point = landmarks.part(i)
                    mouth_points.append((point.x, point.y))
                
                # Calculate mouth bounding box
                mouth_points = np.array(mouth_points)
                x, y, w, h = cv2.boundingRect(mouth_points)
                
                # Expand region slightly
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                mouth_region = frame[y:y+h, x:x+w]
                return cv2.resize(mouth_region, (128, 64))
        
        except Exception:
            pass
        
        return None
    
    def _extract_mouth_haar(self, frame):
        """Extract mouth using Haar cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect faces first
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Use first face
                (x, y, w, h) = faces[0]
                face_region = gray[y:y+h, x:x+w]
                
                # Detect mouth in lower half of face
                mouth_region_y = y + h // 2
                mouth_search_area = gray[mouth_region_y:y+h, x:x+w]
                
                mouths = self.mouth_cascade.detectMultiScale(mouth_search_area, 1.3, 5)
                
                if len(mouths) > 0:
                    (mx, my, mw, mh) = mouths[0]
                    # Adjust coordinates to full frame
                    mx += x
                    my += mouth_region_y
                    
                    # Expand mouth region
                    padding = 10
                    mx = max(0, mx - padding)
                    my = max(0, my - padding)
                    mw = min(frame.shape[1] - mx, mw + 2 * padding)
                    mh = min(frame.shape[0] - my, mh + 2 * padding)
                    
                    mouth_region = frame[my:my+mh, mx:mx+mw]
                    return cv2.resize(mouth_region, (128, 64))
        
        except Exception:
            pass
        
        return None
    
    def _extract_mouth_face_based(self, frame):
        """Extract mouth based on face detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                
                # Estimate mouth position (lower third of face)
                mouth_y = y + int(h * 0.6)
                mouth_x = x + int(w * 0.2)
                mouth_w = int(w * 0.6)
                mouth_h = int(h * 0.3)
                
                mouth_region = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
                return cv2.resize(mouth_region, (128, 64))
        
        except Exception:
            pass
        
        return None
    
    def _extract_mouth_center_crop(self, frame):
        """Fallback: center crop method"""
        h, w = frame.shape[:2]
        
        # Center crop focusing on lower face area
        crop_h, crop_w = 64, 128
        center_y = int(h * 0.65)  # Slightly below center
        center_x = w // 2
        
        y1 = max(0, center_y - crop_h // 2)
        y2 = min(h, center_y + crop_h // 2)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(w, center_x + crop_w // 2)
        
        mouth_region = frame[y1:y2, x1:x2]
        return cv2.resize(mouth_region, (128, 64))
    
    def _apply_preprocessing(self, mouth_regions):
        """Apply preprocessing to mouth regions"""
        processed_frames = []
        
        for mouth_region in mouth_regions:
            # Normalize to [0, 1]
            processed = mouth_region.astype(np.float32) / 255.0
            
            # Apply adaptive histogram equalization (optional)
            # processed = self._adaptive_histogram_equalization(processed)
            
            # Apply slight smoothing
            processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
            
            processed_frames.append(processed)
        
        return processed_frames
    
    def _frames_to_tensor(self, processed_frames):
        """Convert frames to tensor format"""
        video_tensor = np.array(processed_frames)  # [75, 64, 128, 3]
        video_tensor = video_tensor.transpose(3, 0, 1, 2)  # [3, 75, 64, 128]
        video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)  # [1, 3, 75, 64, 128]
        return video_tensor


class ComprehensiveLanguageCorrector:
    """
    Comprehensive language model correction system
    Integrates multiple correction strategies for optimal performance
    """
    
    def __init__(self):
        # Complete GRID corpus vocabulary
        self.grid_vocabulary = {
            # Commands
            'PLACE', 'LAY', 'SET',
            # Colors  
            'WHITE', 'RED', 'GREEN', 'BLUE',
            # Prepositions
            'AT', 'BY', 'WITH', 'IN', 'ON',
            # Adverbs
            'NOW', 'PLEASE', 'SOON', 'AGAIN',
            # Objects
            'BIN', 'BOX', 'TABLE',
            # Letters
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            # Numbers
            'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'
        }
        
        # Common character substitutions in lip-reading
        self.substitutions = {
            'B': ['P', 'M'], 'P': ['B', 'M'], 'M': ['B', 'P'],
            'F': ['V', 'TH'], 'V': ['F', 'TH'], 
            'T': ['D', 'TH'], 'D': ['T', 'TH'],
            'K': ['G', 'NG'], 'G': ['K', 'NG'],
            'S': ['Z', 'SH'], 'Z': ['S', 'SH'],
            'CH': ['J', 'SH'], 'J': ['CH', 'SH'],
            'L': ['R', 'W'], 'R': ['L', 'W'], 'W': ['R', 'L']
        }
        
        # GRID sentence patterns
        self.patterns = [
            r'(PLACE|LAY|SET)\s+(WHITE|RED|GREEN|BLUE)\s+(AT|BY|WITH|IN|ON)\s+([A-Z])\s*(NOW|PLEASE|SOON|AGAIN)?',
            r'(PLACE|LAY|SET)\s+([A-Z])\s+(AT|BY|WITH|IN|ON)\s+(WHITE|RED|GREEN|BLUE)\s*(NOW|PLEASE|SOON|AGAIN)?',
            r'(PLACE|LAY|SET)\s+(WHITE|RED|GREEN|BLUE)\s+([A-Z])\s*(NOW|PLEASE|SOON|AGAIN)?',
            r'(PLACE|LAY|SET)\s+([A-Z])\s+(NOW|PLEASE|SOON|AGAIN)?'
        ]
    
    def correct_prediction(self, raw_prediction, confidence_scores=None):
        """
        Apply comprehensive correction to raw prediction
        
        Args:
            raw_prediction (str): Raw CTC decoded output
            confidence_scores (list): Optional confidence scores for each character
            
        Returns:
            dict: Correction results with confidence info
        """
        if not raw_prediction or len(raw_prediction.strip()) == 0:
            return {
                'corrected': 'NO_PREDICTION',
                'confidence': 0.0,
                'method': 'none',
                'alternatives': []
            }
        
        print(f"🔍 Correcting: '{raw_prediction}'")
        
        # Apply correction strategies in order of reliability
        strategies = [
            ('pattern_matching', self._pattern_based_correction),
            ('vocabulary_reconstruction', self._vocabulary_based_correction),
            ('phonetic_correction', self._phonetic_based_correction),
            ('statistical_reconstruction', self._statistical_reconstruction)
        ]
        
        best_result = None
        best_confidence = 0.0
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(raw_prediction)
                
                if result and result != raw_prediction:
                    confidence = self._calculate_confidence(result, raw_prediction)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            'corrected': result,
                            'confidence': confidence,
                            'method': strategy_name,
                            'alternatives': []
                        }
                    
                    print(f"📊 {strategy_name}: '{result}' (confidence: {confidence:.2f})")
            
            except Exception as e:
                print(f"⚠️  {strategy_name} failed: {e}")
                continue
        
        if best_result:
            return best_result
        else:
            return {
                'corrected': raw_prediction,
                'confidence': 0.1,
                'method': 'none',
                'alternatives': []
            }
    
    def _pattern_based_correction(self, text):
        """Apply GRID pattern-based correction"""
        # Look for command words
        commands = ['PLACE', 'LAY', 'SET']
        colors = ['WHITE', 'RED', 'GREEN', 'BLUE']
        prepositions = ['AT', 'BY', 'WITH', 'IN', 'ON']
        adverbs = ['NOW', 'PLEASE', 'SOON', 'AGAIN']
        
        result_components = []
        
        # Find command
        for cmd in commands:
            if self._fuzzy_match(text, cmd, threshold=0.6):
                result_components.append(cmd)
                break
        
        # Find color
        for color in colors:
            if self._fuzzy_match(text, color, threshold=0.6):
                result_components.append(color)
                break
        
        # Find preposition
        for prep in prepositions:
            if self._fuzzy_match(text, prep, threshold=0.7):
                result_components.append(prep)
                break
        
        # Find single letter
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if letter in text:
                result_components.append(letter)
                break
        
        # Find adverb
        for adv in adverbs:
            if self._fuzzy_match(text, adv, threshold=0.6):
                result_components.append(adv)
                break
        
        if len(result_components) >= 2:
            return ' '.join(result_components)
        
        return text
    
    def _vocabulary_based_correction(self, text):
        """Vocabulary-based word reconstruction"""
        corrected_words = []
        i = 0
        
        while i < len(text):
            best_match = None
            best_score = 0
            best_length = 0
            
            # Try different word lengths
            for length in range(2, min(10, len(text) - i + 1)):
                segment = text[i:i+length]
                
                for vocab_word in self.grid_vocabulary:
                    similarity = self._similarity_score(segment, vocab_word)
                    
                    if similarity > best_score and similarity > 0.5:
                        best_score = similarity
                        best_match = vocab_word
                        best_length = length
            
            if best_match:
                corrected_words.append(best_match)
                i += best_length
            else:
                i += 1
        
        return ' '.join(corrected_words) if corrected_words else text
    
    def _phonetic_based_correction(self, text):
        """Apply phonetic-based corrections"""
        # This is a simplified phonetic correction
        # In practice, this would use phonetic algorithms
        corrected = text
        
        # Apply common substitutions
        for original, substitutes in self.substitutions.items():
            if original in corrected:
                for substitute in substitutes:
                    test_correction = corrected.replace(original, substitute)
                    if self._contains_valid_words(test_correction):
                        corrected = test_correction
                        break
        
        return corrected
    
    def _statistical_reconstruction(self, text):
        """Statistical reconstruction based on GRID patterns"""
        # Build most likely GRID sentence
        components = []
        
        # Most common patterns
        if any(c in text for c in 'PLS'):
            components.append('PLACE')
        
        if any(c in text for c in 'WHT'):
            components.append('WHITE')
        elif any(c in text for c in 'RD'):
            components.append('RED')
        
        if 'AT' in text:
            components.append('AT')
        
        # Find letters
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if letter in text:
                components.append(letter)
                break
        
        if 'NOW' in text or text.endswith('W'):
            components.append('NOW')
        
        return ' '.join(components) if components else f"UNCLEAR: {text[:10]}..."
    
    def _fuzzy_match(self, text, target, threshold=0.6):
        """Check if target fuzzy matches in text"""
        if target in text:
            return True
        
        # Character-based fuzzy matching
        target_chars = Counter(target)
        text_chars = Counter(text)
        
        matches = sum(min(target_chars[c], text_chars[c]) for c in target_chars)
        return matches / len(target) >= threshold
    
    def _similarity_score(self, s1, s2):
        """Calculate similarity score between strings"""
        if s1 == s2:
            return 1.0
        
        # Levenshtein distance-based similarity
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        distance = self._levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _contains_valid_words(self, text):
        """Check if text contains valid vocabulary words"""
        words = text.split()
        return any(word in self.grid_vocabulary for word in words)
    
    def _calculate_confidence(self, corrected, original):
        """Calculate confidence score for correction"""
        if corrected == original:
            return 0.1
        
        # Base confidence on vocabulary coverage
        corrected_words = corrected.split()
        valid_words = sum(1 for word in corrected_words if word in self.grid_vocabulary)
        
        if len(corrected_words) == 0:
            return 0.1
        
        vocab_score = valid_words / len(corrected_words)
        
        # Adjust for sentence structure
        structure_score = 0.5
        if len(corrected_words) >= 2:
            structure_score = 0.7
        if len(corrected_words) >= 3:
            structure_score = 0.9
        
        return min(1.0, vocab_score * structure_score)


class FinalEnhancedLipNet:
    """
    Final Enhanced LipNet System
    Integrates: Advanced Preprocessing + Base Model + Comprehensive Correction
    """
    
    def __init__(self, model_path='models/pretrained/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'):
        print("🚀 Initializing Final Enhanced LipNet System...")
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AdvancedPreprocessor()
        self.base_model = self._load_base_model(model_path)
        self.corrector = ComprehensiveLanguageCorrector()
        
        # Character mapping
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        
        print(f"✅ Final Enhanced LipNet ready on {self.device}")
    
    def _load_base_model(self, model_path):
        """Load the pre-trained base model"""
        model = LipNet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, video_path):
        """
        Complete prediction pipeline
        
        Args:
            video_path (str): Path to input video
            
        Returns:
            dict: Comprehensive prediction results
        """
        try:
            print(f"\n🎬 Processing: {video_path}")
            
            # Stage 1: Advanced Preprocessing
            print("📊 Stage 1: Advanced preprocessing...")
            preprocessed_tensor = self.preprocessor.process_video(video_path)
            
            # Stage 2: Base Model Inference
            print("🧠 Stage 2: Base model inference...")
            with torch.no_grad():
                preprocessed_tensor = preprocessed_tensor.to(self.device)
                model_outputs = self.base_model(preprocessed_tensor)
                raw_prediction = self._ctc_decode(model_outputs)
            
            # Stage 3: Comprehensive Correction
            print("🔧 Stage 3: Comprehensive correction...")
            correction_result = self.corrector.correct_prediction(raw_prediction)
            
            # Compile results
            result = {
                'raw_prediction': raw_prediction,
                'enhanced_prediction': correction_result['corrected'],
                'confidence': correction_result['confidence'],
                'correction_method': correction_result['method'],
                'improvement': correction_result['corrected'] != raw_prediction
            }
            
            # Display results
            self._display_results(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return {
                'raw_prediction': 'ERROR',
                'enhanced_prediction': 'ERROR',
                'confidence': 0.0,
                'correction_method': 'error',
                'improvement': False
            }
    
    def _ctc_decode(self, outputs):
        """CTC decoding from model outputs"""
        arg_maxes = torch.argmax(outputs, dim=2)[0]
        
        decoded = []
        prev_char = None
        
        for char_idx in arg_maxes:
            char_idx = char_idx.item()
            if char_idx < len(self.chars):
                char = self.chars[char_idx]
                if char != prev_char:
                    decoded.append(char)
                prev_char = char
            else:
                prev_char = None
        
        return ''.join(decoded)
    
    def _display_results(self, result):
        """Display comprehensive results"""
        print("\n" + "="*70)
        print("📋 FINAL ENHANCED LIPNET RESULTS")
        print("="*70)
        print(f"Raw Prediction:       '{result['raw_prediction']}'")
        print(f"Enhanced Prediction:  '{result['enhanced_prediction']}'")
        print(f"Confidence Score:     {result['confidence']:.2f}")
        print(f"Correction Method:    {result['correction_method']}")
        print(f"Improvement:          {'✅ Yes' if result['improvement'] else '❌ No'}")
        print("="*70)


def main():
    """Test the final enhanced system"""
    print("🧪 Testing Final Enhanced LipNet System")
    print("="*70)
    
    # Initialize system
    enhanced_lipnet = FinalEnhancedLipNet()
    
    # Test video
    test_video = "data/test_videos/id23_6000_priazn.mpg"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    # Run prediction
    result = enhanced_lipnet.predict(test_video)
    
    # Analysis
    print(f"\n📊 System Analysis:")
    print(f"✅ Advanced preprocessing: Multiple mouth detection strategies")
    print(f"✅ Base model inference: Pre-trained LipNet (4.64% WER)")
    print(f"✅ Comprehensive correction: {result['correction_method']}")
    print(f"✅ Overall improvement: {result['improvement']}")
    print(f"✅ System confidence: {result['confidence']:.2f}")
    
    print(f"\n🎯 Ready for Phase 2B: Fine-tuning!")
    print("Next steps: Dataset preparation and transfer learning")


if __name__ == "__main__":
    main()