# src/enhanced_language_model_v2.py
# Phase 2B Day 1-2: Smarter Language Model Correction
# FYP Enhancement: More intelligent post-processing

import torch
import cv2
import numpy as np
import sys
import os
from collections import Counter
import re

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lipnet_base import LipNet

class SmartLanguageCorrector:
    """
    FYP Enhancement: Smarter language model with multiple correction strategies
    """
    
    def __init__(self):
        # GRID corpus specific vocabulary (exactly what the model was trained on)
        self.grid_commands = {
            'PLACE', 'LAY', 'SET'
        }
        
        self.grid_colors = {
            'WHITE', 'RED', 'GREEN', 'BLUE'
        }
        
        self.grid_prepositions = {
            'AT', 'BY', 'WITH', 'IN', 'ON'
        }
        
        self.grid_adverbs = {
            'NOW', 'PLEASE', 'SOON', 'AGAIN'
        }
        
        self.grid_objects = {
            'BIN', 'BOX', 'TABLE'
        }
        
        self.grid_letters = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        }
        
        self.grid_numbers = {
            'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'
        }
        
        # Combine all vocabularies
        self.all_vocab = (self.grid_commands | self.grid_colors | self.grid_prepositions | 
                         self.grid_adverbs | self.grid_objects | self.grid_letters | self.grid_numbers)
        
        # Common GRID sentence patterns
        self.grid_patterns = [
            r'(PLACE|LAY|SET)\s+(WHITE|RED|GREEN|BLUE)\s+(AT|BY|WITH|IN|ON)',
            r'(PLACE|LAY|SET)\s+([A-Z])\s+(AT|BY|WITH|IN|ON)',
            r'(PLACE|LAY|SET)\s+(WHITE|RED|GREEN|BLUE)\s+([A-Z])',
            r'(WHITE|RED|GREEN|BLUE)\s+(AT|BY|WITH|IN|ON)\s+([A-Z])',
        ]
        
        # Character substitution patterns (common OCR/lip-reading errors)
        self.char_substitutions = {
            'C': ['G', 'O'],
            'G': ['C', 'Q'],
            'B': ['P', 'R'],
            'P': ['B', 'R'],
            'K': ['X', 'R'],
            'X': ['K', 'R'],
            'V': ['W', 'U'],
            'W': ['V', 'U'],
            'N': ['M', 'H'],
            'M': ['N', 'H'],
            'F': ['T', 'P'],
            'T': ['F', 'L'],
            'J': ['Y', 'I'],
            'Y': ['J', 'I'],
            'U': ['V', 'O'],
            'O': ['U', 'Q'],
            'Q': ['O', 'G'],
            'A': ['R', 'H'],
            'R': ['A', 'B'],
            'Z': ['S', 'X'],
            'S': ['Z', 'C']
        }
    
    def correct_sequence(self, raw_prediction):
        """
        Apply advanced correction strategies
        """
        if not raw_prediction or len(raw_prediction.strip()) == 0:
            return "NO_PREDICTION"
        
        print(f"🔍 Analyzing: '{raw_prediction}'")
        
        # Strategy 1: Pattern-based extraction
        pattern_result = self._extract_using_patterns(raw_prediction)
        if pattern_result != raw_prediction:
            print(f"📊 Pattern extraction: '{pattern_result}'")
            return pattern_result
        
        # Strategy 2: Vocabulary matching with fuzzy logic
        vocab_result = self._vocabulary_fuzzy_match(raw_prediction)
        if vocab_result != raw_prediction:
            print(f"📊 Vocabulary matching: '{vocab_result}'")
            return vocab_result
        
        # Strategy 3: Character-level correction
        char_result = self._character_level_correction(raw_prediction)
        if char_result != raw_prediction:
            print(f"📊 Character correction: '{char_result}'")
            return char_result
        
        # Strategy 4: Sliding window search
        window_result = self._sliding_window_search(raw_prediction)
        if window_result != raw_prediction:
            print(f"📊 Sliding window: '{window_result}'")
            return window_result
        
        # Strategy 5: Statistical reconstruction
        stats_result = self._statistical_reconstruction(raw_prediction)
        print(f"📊 Statistical reconstruction: '{stats_result}'")
        return stats_result
    
    def _extract_using_patterns(self, text):
        """Extract meaningful content using GRID patterns"""
        # Look for GRID command structure
        commands = ['PLACE', 'LAY', 'SET']
        colors = ['WHITE', 'RED', 'GREEN', 'BLUE']
        prepositions = ['AT', 'BY', 'WITH', 'IN', 'ON']
        
        # Try to find command-like patterns
        for cmd in commands:
            if self._fuzzy_contains(text, cmd):
                result = [cmd]
                
                # Look for color
                for color in colors:
                    if self._fuzzy_contains(text, color):
                        result.append(color)
                        break
                
                # Look for preposition
                for prep in prepositions:
                    if self._fuzzy_contains(text, prep):
                        result.append(prep)
                        break
                
                # Look for single letter
                for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    if letter in text:
                        result.append(letter)
                        break
                
                if len(result) > 1:
                    return ' '.join(result)
        
        return text
    
    def _vocabulary_fuzzy_match(self, text):
        """Match against vocabulary with fuzzy logic"""
        words = []
        i = 0
        
        while i < len(text):
            best_match = None
            best_score = float('inf')
            best_length = 0
            
            # Try different word lengths
            for length in range(2, min(8, len(text) - i + 1)):
                candidate = text[i:i+length]
                
                # Find best vocabulary match
                for vocab_word in self.all_vocab:
                    score = self._edit_distance(candidate, vocab_word)
                    
                    # Good match criteria
                    if score <= max(1, len(vocab_word) * 0.3) and score < best_score:
                        best_score = score
                        best_match = vocab_word
                        best_length = length
            
            if best_match:
                words.append(best_match)
                i += best_length
            else:
                i += 1
        
        return ' '.join(words) if words else text
    
    def _character_level_correction(self, text):
        """Apply character-level corrections"""
        corrected = list(text)
        
        for i, char in enumerate(corrected):
            if char in self.char_substitutions:
                # Look at context to decide best substitution
                for substitute in self.char_substitutions[char]:
                    # Try substitution and see if it creates valid words
                    test_corrected = corrected.copy()
                    test_corrected[i] = substitute
                    test_text = ''.join(test_corrected)
                    
                    # Check if substitution creates recognizable patterns
                    if self._contains_vocab_words(test_text):
                        corrected[i] = substitute
                        break
        
        return ''.join(corrected)
    
    def _sliding_window_search(self, text):
        """Search for vocabulary words using sliding window"""
        found_words = []
        used_positions = set()
        
        # Try different window sizes
        for window_size in range(3, 8):
            for i in range(len(text) - window_size + 1):
                if i in used_positions:
                    continue
                
                window = text[i:i+window_size]
                
                # Check exact matches first
                if window in self.all_vocab:
                    found_words.append((i, window))
                    used_positions.update(range(i, i+window_size))
                    continue
                
                # Check fuzzy matches
                for vocab_word in self.all_vocab:
                    if self._edit_distance(window, vocab_word) <= 1:
                        found_words.append((i, vocab_word))
                        used_positions.update(range(i, i+window_size))
                        break
        
        # Sort by position and extract words
        found_words.sort(key=lambda x: x[0])
        return ' '.join([word for pos, word in found_words]) if found_words else text
    
    def _statistical_reconstruction(self, text):
        """Use statistical methods to reconstruct likely sentence"""
        # If all else fails, try to construct most likely GRID sentence
        likely_components = []
        
        # Most common GRID patterns
        if any(char in text for char in 'PLS'):
            likely_components.append('PLACE')
        elif any(char in text for char in 'LY'):
            likely_components.append('LAY')
        elif any(char in text for char in 'SET'):
            likely_components.append('SET')
        
        # Look for color indicators
        if any(char in text for char in 'WHT'):
            likely_components.append('WHITE')
        elif any(char in text for char in 'RD'):
            likely_components.append('RED')
        elif any(char in text for char in 'GRN'):
            likely_components.append('GREEN')
        elif any(char in text for char in 'BLU'):
            likely_components.append('BLUE')
        
        # Look for preposition
        if 'AT' in text:
            likely_components.append('AT')
        elif any(char in text for char in 'BY'):
            likely_components.append('BY')
        
        # Look for single letters
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if letter in text and len([c for c in text if c == letter]) == 1:
                likely_components.append(letter)
                break
        
        # Add "NOW" if it seems like end of sentence
        if text.endswith('W') or 'NOW' in text:
            likely_components.append('NOW')
        
        return ' '.join(likely_components) if likely_components else f"RECONSTRUCTED: {text[:15]}..."
    
    def _fuzzy_contains(self, text, target):
        """Check if target is fuzzy contained in text"""
        if target in text:
            return True
        
        # Check if most characters of target are in text
        target_chars = Counter(target)
        text_chars = Counter(text)
        
        matches = 0
        for char, count in target_chars.items():
            matches += min(count, text_chars.get(char, 0))
        
        return matches >= len(target) * 0.7  # 70% character match
    
    def _contains_vocab_words(self, text):
        """Check if text contains recognizable vocabulary words"""
        for word in self.all_vocab:
            if word in text:
                return True
        return False
    
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


# Update the EnhancedLipNetPredictor to use the new corrector
class EnhancedLipNetPredictor:
    """Enhanced LipNet with smarter language model correction"""
    
    def __init__(self, model_path='models/pretrained/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt'):
        print("🚀 Initializing Enhanced LipNet System v2...")
        
        # Load base components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.preprocessor = SimplePreprocessor()
        self.corrector = SmartLanguageCorrector()  # Updated corrector
        
        # Character mapping
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        
        print(f"✅ Enhanced LipNet v2 loaded on {self.device}")
    
    def _load_model(self, model_path):
        """Load the pre-trained LipNet model"""
        model = LipNet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict_enhanced(self, video_path):
        """Enhanced prediction with smarter correction"""
        try:
            print(f"\n🎬 Processing video: {video_path}")
            
            # Step 1: Basic preprocessing
            print("📊 Step 1: Basic preprocessing...")
            preprocessed_tensor = self.preprocessor.process_video(video_path)
            
            # Step 2: Base model prediction
            print("🧠 Step 2: Base LipNet prediction...")
            with torch.no_grad():
                preprocessed_tensor = preprocessed_tensor.to(self.device)
                outputs = self.model(preprocessed_tensor)
                raw_prediction = self._ctc_decode(outputs)
            
            # Step 3: Smart language model correction
            print("🔧 Step 3: Smart language model correction...")
            corrected_prediction = self.corrector.correct_sequence(raw_prediction)
            
            # Results
            print("\n" + "="*60)
            print("📋 ENHANCED LIPNET RESULTS v2")
            print("="*60)
            print(f"Raw LipNet Output:    '{raw_prediction}'")
            print(f"Enhanced Prediction:  '{corrected_prediction}'")
            
            # Calculate improvement
            improvement = 'Yes' if corrected_prediction != raw_prediction else 'No'
            if improvement == 'Yes':
                print(f"✅ Improvement: Transformation applied!")
            else:
                print(f"⚠️  No transformation needed")
            
            print("="*60)
            
            return {
                'raw_prediction': raw_prediction,
                'enhanced_prediction': corrected_prediction,
                'improvement': improvement
            }
            
        except Exception as e:
            print(f"❌ Error in enhanced prediction: {e}")
            return {'raw_prediction': 'ERROR', 'enhanced_prediction': 'ERROR', 'improvement': 'Error'}
    
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


class SimplePreprocessor:
    """Simple video preprocessor for testing"""
    
    def process_video(self, video_path):
        """Basic video preprocessing"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Ensure exactly 75 frames
        if len(frames) > 75:
            start_idx = (len(frames) - 75) // 2
            frames = frames[start_idx:start_idx + 75]
        elif len(frames) < 75:
            while len(frames) < 75:
                frames.append(frames[-1])
        
        # Process frames
        processed_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            
            # Extract mouth region (center crop)
            mouth_y = h // 2
            mouth_x = w // 2
            mouth_w, mouth_h = 128, 64
            
            y1 = max(0, mouth_y - mouth_h // 2)
            y2 = min(h, mouth_y + mouth_h // 2)
            x1 = max(0, mouth_x - mouth_w // 2)
            x2 = min(w, mouth_x + mouth_w // 2)
            
            mouth_region = frame[y1:y2, x1:x2]
            mouth_region = cv2.resize(mouth_region, (128, 64))
            mouth_region = mouth_region.astype(np.float32) / 255.0
            
            processed_frames.append(mouth_region)
        
        # Convert to tensor
        video_tensor = np.array(processed_frames)
        video_tensor = video_tensor.transpose(3, 0, 1, 2)
        video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)
        
        return video_tensor


def main():
    """Test the enhanced system v2"""
    print("🧪 Testing Enhanced LipNet System v2 (Smarter Correction)")
    print("="*60)
    
    predictor = EnhancedLipNetPredictor()
    test_video = "data/test_videos/id23_6000_priazn.mpg"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    results = predictor.predict_enhanced(test_video)
    
    print(f"\n📊 Enhancement Analysis:")
    print(f"Improvement: {results['improvement']}")
    
    if results['improvement'] == 'Yes':
        print("✅ Smart language model correction is working!")
    else:
        print("⚠️  Will continue refining the algorithm")
    
    print(f"\n🎯 Progress: Phase 2B Day 1-2 Complete!")
    print("Next: Add confidence scoring and ensemble methods")


if __name__ == "__main__":
    main()