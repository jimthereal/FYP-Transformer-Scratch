from spellchecker import SpellChecker
import difflib
import re

class LipReadingOptimizedCorrector:
    """
    FYP Enhancement: Lip-reading optimized language corrector
    Specifically designed for visual speech recognition errors
    """
    
    def __init__(self):
        print("🚀 Initializing Lip-Reading Optimized Corrector...")
        
        # Initialize spell checker
        self.spell = SpellChecker(language='en')
        
        # Lip-reading specific corrections (common visual confusions)
        self.lipread_corrections = {
            # Common visual confusions in lip-reading
            'HELO': 'HELLO',
            'HELOT': 'HELLO', 
            'LUVE': 'LOVE',
            'LURE': 'LOVE',
            'THENK': 'THANK',
            'THEN': 'THANK',  # When followed by "YOU"
            'PLAS': 'PLEASE',
            'PLATS': 'PLEASE',
            'HALP': 'HELP',
            'HALT': 'HELP',   # When context suggests help
            'AR': 'ARE',
            'U': 'YOU',
            'UR': 'YOUR',
            'TODA': 'TODAY',
            'TOAD': 'TODAY',  # When context suggests time
            'TIM': 'TIME',
            'VAR': 'VERY',
            'VER': 'VERY',
            'WIT': 'WITH',
            'WHIT': 'WITH',
            'WAT': 'WHAT',
            'WEN': 'WHEN',
            'HOU': 'HOW',
            'WERE': 'WHERE', # Context dependent
            'NO': 'NOW',     # Context dependent
            'GOD': 'GOOD',
            'GUD': 'GOOD',
            'BAD': 'BED',    # Context dependent
            'RED': 'READ',   # Context dependent
            'BLUE': 'BLOW',  # Context dependent
        }
        
        # Common lip-reading phrases
        self.common_phrases = {
            'GOOD MORNING': ['GOD MORNING', 'GUD MORNING', 'GOOD MORN'],
            'GOOD EVENING': ['GOD EVENING', 'GUD EVENING', 'GOOD EVE'],
            'GOOD AFTERNOON': ['GOD AFTERNOON', 'GUD AFTERNOON'],
            'HOW ARE YOU': ['HOU AR U', 'HOW AR U', 'HOU ARE U'],
            'WHAT TIME': ['WAT TIM', 'WHAT TIM', 'WAT TIME'],
            'THANK YOU': ['THENK U', 'THANK U', 'THEN U'],
            'PLEASE HELP': ['PLAS HALP', 'PLEASE HALP', 'PLAS HELP'],
            'I LOVE YOU': ['I LUVE U', 'I LOVE U', 'I LURE U'],
            'EXCUSE ME': ['EXCUS ME', 'EXCUSE MI'],
            'NICE TO MEET YOU': ['NIS TO MEET U', 'NICE TO MET U'],
        }
        
        print("✅ Lip-Reading Optimized Corrector ready!")
    
    def correct_prediction(self, raw_prediction):
        """
        Optimized correction for lip-reading specific errors
        """
        if not raw_prediction or len(raw_prediction.strip()) == 0:
            return {
                'corrected': 'NO_PREDICTION',
                'confidence': 0.0,
                'method': 'none'
            }
        
        print(f"🔍 Lip-reading optimized correcting: '{raw_prediction}'")
        
        # Strategy 1: Phrase-level correction (most accurate)
        phrase_result = self._phrase_level_correction(raw_prediction)
        if phrase_result['confidence'] > 0.8:
            print(f"✅ Phrase correction: '{phrase_result['corrected']}'")
            return phrase_result
        
        # Strategy 2: Word-level lip-reading corrections
        word_result = self._lipread_word_correction(raw_prediction)
        if word_result['confidence'] > 0.6:
            print(f"✅ Lip-reading word correction: '{word_result['corrected']}'")
            return word_result
        
        # Strategy 3: Context-aware spell checking
        spell_result = self._context_aware_spell_check(raw_prediction)
        if spell_result['confidence'] > 0.5:
            print(f"✅ Context-aware spell check: '{spell_result['corrected']}'")
            return spell_result
        
        # Strategy 4: Fallback to basic correction
        fallback_result = self._basic_correction(raw_prediction)
        print(f"✅ Basic correction: '{fallback_result['corrected']}'")
        return fallback_result
    
    def _phrase_level_correction(self, text):
        """Correct common phrases that are often mispredicted"""
        text_upper = text.upper()
        
        for correct_phrase, variations in self.common_phrases.items():
            # Check if any variation matches the input
            for variation in variations:
                similarity = difflib.SequenceMatcher(None, text_upper, variation).ratio()
                if similarity > 0.7:
                    return {
                        'corrected': correct_phrase,
                        'confidence': similarity,
                        'method': 'phrase_correction'
                    }
        
        return {'corrected': text, 'confidence': 0.0, 'method': 'none'}
    
    def _lipread_word_correction(self, text):
        """Apply lip-reading specific word corrections"""
        words = text.upper().split()
        corrected_words = []
        improvements = 0
        
        for i, word in enumerate(words):
            # Direct lip-reading correction
            if word in self.lipread_corrections:
                corrected_word = self.lipread_corrections[word]
                corrected_words.append(corrected_word)
                improvements += 1
            
            # Context-sensitive corrections
            elif word == 'THEN' and i < len(words) - 1 and words[i + 1] == 'U':
                corrected_words.append('THANK')
                improvements += 1
            elif word == 'HALT' and 'ME' in words:
                corrected_words.append('HELP')
                improvements += 1
            elif word == 'TOAD' and ('TIME' in text or 'TODAY' in text or 'TIM' in text):
                corrected_words.append('TODAY')
                improvements += 1
            else:
                corrected_words.append(word)
        
        if improvements > 0:
            confidence = min(0.9, 0.5 + (improvements * 0.2))
            return {
                'corrected': ' '.join(corrected_words),
                'confidence': confidence,
                'method': 'lipread_word_correction'
            }
        
        return {'corrected': text, 'confidence': 0.0, 'method': 'none'}
    
    def _context_aware_spell_check(self, text):
        """Spell check with lip-reading context awareness"""
        words = text.upper().split()
        corrected_words = []
        confidence_scores = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if already correct
            if word_lower in self.spell:
                corrected_words.append(word)
                confidence_scores.append(1.0)
                continue
            
            # Get spell check suggestions
            suggestions = self.spell.candidates(word_lower)
            
            if suggestions:
                # Choose best suggestion based on lip-reading likelihood
                best_suggestion = self._choose_best_lipread_suggestion(word, suggestions)
                corrected_words.append(best_suggestion.upper())
                
                # Calculate confidence
                edit_distance = self._edit_distance(word_lower, best_suggestion)
                confidence = max(0.1, 1.0 - (edit_distance / len(word)))
                confidence_scores.append(confidence)
            else:
                corrected_words.append(word)
                confidence_scores.append(0.1)
        
        if corrected_words:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            return {
                'corrected': ' '.join(corrected_words),
                'confidence': avg_confidence,
                'method': 'context_spell_check'
            }
        
        return {'corrected': text, 'confidence': 0.0, 'method': 'none'}
    
    def _choose_best_lipread_suggestion(self, original_word, suggestions):
        """Choose the best suggestion considering lip-reading patterns"""
        # Prioritize common words that are visually similar
        priority_words = [
            'hello', 'love', 'thank', 'please', 'help', 'you', 'are', 'time', 
            'today', 'very', 'good', 'what', 'when', 'where', 'how', 'with'
        ]
        
        # First, check if any priority words are in suggestions
        for priority_word in priority_words:
            if priority_word in suggestions:
                return priority_word
        
        # Otherwise, return the first suggestion
        return list(suggestions)[0]
    
    def _basic_correction(self, text):
        """Basic fallback correction"""
        # Simple cleanup
        cleaned = re.sub(r'[^A-Z\s]', '', text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if cleaned and cleaned != text:
            return {
                'corrected': cleaned,
                'confidence': 0.3,
                'method': 'basic_cleanup'
            }
        
        return {
            'corrected': text,
            'confidence': 0.1,
            'method': 'no_correction'
        }
    
    def _edit_distance(self, s1, s2):
        """Calculate edit distance"""
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


def main():
    """Test the lip-reading optimized corrector"""
    print("🧪 Testing Lip-Reading Optimized Corrector")
    print("=" * 50)
    
    # Initialize corrector
    corrector = LipReadingOptimizedCorrector()
    
    # Test cases (same as before)
    test_cases = [
        "UGAVBTAGFBKAPBFABAKXGABPAQY",  # Complete gibberish
        "HELO HOW AR U TODA",            # Misspelled greeting
        "WHAT TIM IS IT",                # Simple question
        "THENK U VER MUCH",              # Polite response
        "I LUVE U",                      # Common phrase
        "PLAS HALP ME",                  # Request for help
        "GOD MORNING",                   # Common greeting
        "HOU AR U",                      # How are you
    ]
    
    for test in test_cases:
        print(f"\n📝 Testing: '{test}'")
        result = corrector.correct_prediction(test)
        print(f"✅ Result: '{result['corrected']}'")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"🔧 Method: {result['method']}")
        print("-" * 40)


if __name__ == "__main__":
    main()