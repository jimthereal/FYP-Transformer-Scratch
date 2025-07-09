import torch
import cv2
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lipnet_base import LipNet
import re
from collections import defaultdict, Counter
import difflib

class LanguageModelCorrector:
    """
    FYP Enhancement: Language model post-processing for LipNet predictions
    
    This class applies intelligent corrections to raw LipNet output to improve
    real-world performance without retraining the base model.
    """
    
    def __init__(self):
        # Common English words for lip-reading context
        self.common_words = {
            'PLACE', 'WHITE', 'AT', 'RED', 'NOW', 'PLEASE', 'WITH', 'BIN',
            'LAY', 'SET', 'GREEN', 'BLUE', 'AGAIN', 'SOON', 'TABLE', 'CHAIR',
            'BOOK', 'PAPER', 'WATER', 'GLASS', 'HAND', 'FACE', 'HEAD', 'EYES',
            'MOUTH', 'WORD', 'TIME', 'WORK', 'HOME', 'LOOK', 'GOOD', 'BACK',
            'COME', 'MAKE', 'TAKE', 'GIVE', 'TELL', 'KNOW', 'THINK', 'WANT',
            'NEED', 'HELP', 'STOP', 'MOVE', 'TURN', 'WALK', 'TALK', 'SPEAK'
        }
        
        # Load GRID corpus vocabulary if available
        self.grid_vocabulary = self._load_grid_vocabulary()
        
    def _load_grid_vocabulary(self):
        """Load GRID corpus vocabulary for better context"""
        grid_words = {
            'PLACE', 'LAY', 'SET', 'WHITE', 'RED', 'GREEN', 'BLUE', 'AT',
            'BY', 'WITH', 'IN', 'ON', 'NOW', 'PLEASE', 'SOON', 'AGAIN',
            'BIN', 'BOX', 'TABLE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN',
            'EIGHT', 'NINE'
        }
        return grid_words
    
    def correct_sequence(self, raw_prediction):
        """
        Apply multiple correction strategies to raw LipNet output
        
        Args:
            raw_prediction (str): Raw output from LipNet (e.g., "UGAVBIJKAGBYKAVJBEABVAKZABUQP")
            
        Returns:
            str: Corrected prediction
        """
        if not raw_prediction or len(raw_prediction.strip()) == 0:
            return "NO_PREDICTION"
        
        # Strategy 1: Remove obvious repetitive patterns
        denoised = self._remove_repetitive_patterns(raw_prediction)
        
        # Strategy 2: Split into potential words using statistical analysis
        word_candidates = self._segment_into_words(denoised)
        
        # Strategy 3: Correct individual words using similarity matching
        corrected_words = []
        for word in word_candidates:
            corrected = self._correct_word(word)
            if corrected:
                corrected_words.append(corrected)
        
        # Strategy 4: Apply context-based corrections
        final_result = self._apply_context_corrections(corrected_words)
        
        return final_result
    
    def _remove_repetitive_patterns(self, text):
        """Remove obvious repetitive character patterns"""
        # Remove consecutive repeated characters (more than 2)
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove very short repetitive patterns
        cleaned = re.sub(r'(.{1,3})\1{3,}', r'\1', cleaned)
        
        return cleaned
    
    def _segment_into_words(self, text):
        """Segment text into potential words using statistical methods"""
        if len(text) < 3:
            return [text]
        
        # Simple word segmentation based on character frequency
        words = []
        current_word = ""
        
        for i, char in enumerate(text):
            current_word += char
            
            # Check if this could be end of a word
            if len(current_word) >= 3:
                # Look ahead to see if next characters form common patterns
                if i < len(text) - 1:
                    # Simple heuristic: if we have a reasonable word length, split
                    if len(current_word) >= 4 and self._is_potential_word_boundary(text, i):
                        words.append(current_word)
                        current_word = ""
        
        if current_word:
            words.append(current_word)
        
        return words if words else [text]
    
    def _is_potential_word_boundary(self, text, position):
        """Simple heuristic to determine word boundaries"""
        # This is a simple implementation - can be enhanced
        if position < len(text) - 2:
            # Look for patterns that suggest word boundaries
            next_chars = text[position+1:position+3]
            return any(pattern in next_chars for pattern in ['PL', 'WH', 'RE', 'BL', 'GR'])
        return False
    
    def _correct_word(self, word):
        """Correct individual word using similarity matching"""
        if len(word) < 2:
            return word
        
        # First, check exact matches in known vocabularies
        if word in self.common_words or word in self.grid_vocabulary:
            return word
        
        # Find closest match using edit distance
        all_vocabulary = self.common_words.union(self.grid_vocabulary)
        
        best_match = None
        best_score = float('inf')
        
        for vocab_word in all_vocabulary:
            # Calculate similarity score (edit distance + length difference)
            distance = self._edit_distance(word, vocab_word)
            length_penalty = abs(len(word) - len(vocab_word)) * 0.5
            score = distance + length_penalty
            
            # Only consider if similarity is reasonable
            if score < best_score and score <= max(3, len(word) * 0.4):
                best_score = score
                best_match = vocab_word
        
        return best_match if best_match else word
    
    def _edit_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
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
    
    def _apply_context_corrections(self, words):
        """Apply context-based corrections to word sequence"""
        if not words:
            return "NO_CLEAR_PREDICTION"
        
        # Remove very short words that are likely noise
        filtered_words = [w for w in words if len(w) >= 2 or w in ['A', 'I']]
        
        if not filtered_words:
            return " ".join(words)  # Return original if filtering removes everything
        
        # Apply common phrase patterns from GRID corpus
        result = " ".join(filtered_words)
        
        # Common GRID patterns
        result = re.sub(r'^PLACE\s+', 'PLACE ', result)
        result = re.sub(r'\s+AT\s+', ' AT ', result)
        result = re.sub(r'\s+NOW\s*$', ' NOW', result)
        
        return result


class SimplePreprocessor:
    """
    Simple video preprocessor for testing language model corrections
    Uses basic preprocessing without advanced features
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_video(self, video_path):
        """
        Simple video preprocessing - basic version for testing
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            torch.Tensor: Preprocessed video tensor [1, 3, 75, 64, 128]
        """
        try:
            print(f"📹 Processing video: {video_path}")
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError("No frames found in video")
            
            print(f"📊 Loaded {len(frames)} frames")
            
            # Ensure exactly 75 frames
            if len(frames) > 75:
                # Take middle 75 frames
                start_idx = (len(frames) - 75) // 2
                frames = frames[start_idx:start_idx + 75]
            elif len(frames) < 75:
                # Repeat last frame to reach 75
                while len(frames) < 75:
                    frames.append(frames[-1])
            
            # Process frames
            processed_frames = []
            for frame in frames:
                # Simple mouth region extraction (center crop)
                h, w = frame.shape[:2]
                
                # Extract mouth region (bottom half, center)
                mouth_y = h // 2
                mouth_x = w // 2
                mouth_w, mouth_h = 128, 64
                
                y1 = max(0, mouth_y - mouth_h // 2)
                y2 = min(h, mouth_y + mouth_h // 2)
                x1 = max(0, mouth_x - mouth_w // 2)
                x2 = min(w, mouth_x + mouth_w // 2)
                
                mouth_region = frame[y1:y2, x1:x2]
                
                # Resize to target size
                mouth_region = cv2.resize(mouth_region, (128, 64))
                
                # Normalize to [0, 1]
                mouth_region = mouth_region.astype(np.float32) / 255.0
                
                processed_frames.append(mouth_region)
            
            # Convert to tensor [3, 75, 64, 128]
            video_tensor = np.array(processed_frames)  # [75, 64, 128, 3]
            video_tensor = video_tensor.transpose(3, 0, 1, 2)  # [3, 75, 64, 128]
            video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)  # [1, 3, 75, 64, 128]
            
            print(f"✅ Preprocessed video shape: {video_tensor.shape}")
            return video_tensor
            
        except Exception as e:
            print(f"❌ Error in video preprocessing: {e}")
            raise


class EnhancedLipNetPredictor:
    """
    FYP Enhancement: Combines base LipNet with language model correction
    
    This is our enhanced system: LipNet + Basic_Preprocessing + Language_Model
    """
    
    def __init__(self, model_path='models/pretrained/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'):
        print("🚀 Initializing Enhanced LipNet System...")
        
        # Load base components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.preprocessor = SimplePreprocessor()
        self.corrector = LanguageModelCorrector()
        
        # Character mapping
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        
        print(f"✅ Enhanced LipNet loaded on {self.device}")
        
    def _load_model(self, model_path):
        """Load the pre-trained LipNet model"""
        model = LipNet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict_enhanced(self, video_path):
        """
        Enhanced prediction pipeline:
        1. Basic preprocessing
        2. Base LipNet prediction (Phase 1) 
        3. Language model correction (Phase 2B)
        """
        try:
            print(f"\n🎬 Processing video: {video_path}")
            
            # Step 1: Basic preprocessing
            print("📊 Step 1: Basic preprocessing...")
            preprocessed_tensor = self.preprocessor.process_video(video_path)
            
            # Step 2: Base model prediction (Phase 1)
            print("🧠 Step 2: Base LipNet prediction...")
            with torch.no_grad():
                preprocessed_tensor = preprocessed_tensor.to(self.device)
                outputs = self.model(preprocessed_tensor)
                
                # CTC decoding
                raw_prediction = self._ctc_decode(outputs)
            
            # Step 3: Language model correction (Phase 2B)
            print("🔧 Step 3: Language model correction...")
            corrected_prediction = self.corrector.correct_sequence(raw_prediction)
            
            # Results
            print("\n" + "="*60)
            print("📋 ENHANCED LIPNET RESULTS")
            print("="*60)
            print(f"Raw LipNet Output:    '{raw_prediction}'")
            print(f"Enhanced Prediction:  '{corrected_prediction}'")
            print("="*60)
            
            return {
                'raw_prediction': raw_prediction,
                'enhanced_prediction': corrected_prediction,
                'improvement': 'Yes' if corrected_prediction != raw_prediction else 'No'
            }
            
        except Exception as e:
            print(f"❌ Error in enhanced prediction: {e}")
            return {
                'raw_prediction': 'ERROR',
                'enhanced_prediction': 'ERROR', 
                'improvement': 'Error'
            }
    
    def _ctc_decode(self, outputs):
        """CTC decoding from model outputs"""
        # Get the most likely character sequence
        arg_maxes = torch.argmax(outputs, dim=2)[0]
        
        # Remove consecutive duplicates and blanks
        decoded = []
        prev_char = None
        
        for char_idx in arg_maxes:
            char_idx = char_idx.item()
            if char_idx < len(self.chars):  # Not blank token
                char = self.chars[char_idx]
                if char != prev_char:  # Remove consecutive duplicates
                    decoded.append(char)
                prev_char = char
            else:
                prev_char = None  # Blank token
        
        return ''.join(decoded)


def main():
    """Test the enhanced LipNet system"""
    print("🧪 Testing Enhanced LipNet System (Phase 2B Day 1)")
    print("="*60)
    
    # Initialize enhanced predictor
    predictor = EnhancedLipNetPredictor()
    
    # Test video path
    test_video = "data/test_videos/id23_6000_priazn.mpg"
    
    # Check if test video exists
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please ensure the video file exists in the correct location")
        return
    
    # Run enhanced prediction
    results = predictor.predict_enhanced(test_video)
    
    # Analysis
    print(f"\n📊 Enhancement Analysis:")
    print(f"Improvement detected: {results['improvement']}")
    
    if results['improvement'] == 'Yes':
        print("✅ Language model correction is working!")
        print("🎯 Raw output was processed and corrected")
    else:
        print("⚠️  No improvement detected")
        print("🔧 This means either:")
        print("   - Raw output was already clean, or")
        print("   - Language model needs fine-tuning")
    
    print(f"\n🎯 Next Steps:")
    print("1. Fine-tune word segmentation algorithm")
    print("2. Expand vocabulary database")
    print("3. Add confidence scoring")
    print("4. Test on more videos")


if __name__ == "__main__":
    main()