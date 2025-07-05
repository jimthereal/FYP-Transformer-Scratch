import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model import LipNet
import editdistance

class BaselineEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.vocabulary = self.setup_vocabulary()
        
    def load_model(self, model_path):
        """Load the pre-trained LipNet model"""
        model = LipNet()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        print(f"✅ Model loaded from {model_path}")
        return model
    
    def setup_vocabulary(self):
        """Setup vocabulary for CTC decoding"""
        # GRID corpus vocabulary: A-Z + space + blank
        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')  # 27 characters
        chars.append('<BLANK>')  # CTC blank token
        
        # Create char-to-index and index-to-char mappings
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        return {
            'chars': chars,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'blank_idx': len(chars) - 1
        }
    
    def preprocess_video(self, video_path):
        """
        Preprocess video for LipNet input
        Returns: tensor of shape [1, 3, 75, 64, 128]
        """
        print(f"📹 Processing video: {video_path}")
        
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
        
        print(f"📊 Video info: {len(frames)} frames")
        
        # Resize frames to 64x128 (LipNet input size)
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to model input size
            frame_resized = cv2.resize(frame_rgb, (128, 64))  # (width, height)
            processed_frames.append(frame_resized)
        
        # Pad or truncate to exactly 75 frames
        target_frames = 75
        if len(processed_frames) > target_frames:
            # Truncate
            processed_frames = processed_frames[:target_frames]
        elif len(processed_frames) < target_frames:
            # Pad with last frame
            last_frame = processed_frames[-1]
            while len(processed_frames) < target_frames:
                processed_frames.append(last_frame)
        
        # Convert to tensor: [75, 64, 128, 3] -> [1, 3, 75, 64, 128]
        video_tensor = np.array(processed_frames)  # [75, 64, 128, 3]
        video_tensor = video_tensor.transpose(3, 0, 1, 2)  # [3, 75, 64, 128]
        video_tensor = video_tensor[np.newaxis, ...]  # [1, 3, 75, 64, 128]
        
        # Normalize to [0, 1]
        video_tensor = video_tensor.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor
        video_tensor = torch.from_numpy(video_tensor).to(self.device)
        
        print(f"✅ Preprocessed video shape: {video_tensor.shape}")
        return video_tensor
    
    def decode_prediction(self, outputs):
        """
        Decode CTC outputs to text
        Args:
            outputs: [1, 75, 28] - model predictions
        Returns:
            decoded_text: string
        """
        # Get the most likely character at each timestep
        predicted_ids = torch.argmax(outputs, dim=2)  # [1, 75]
        predicted_ids = predicted_ids.squeeze(0).cpu().numpy()  # [75]
        
        # Remove blanks and consecutive duplicates (CTC decoding)
        decoded_chars = []
        prev_char_idx = -1
        
        for char_idx in predicted_ids:
            # Skip blank tokens
            if char_idx == self.vocabulary['blank_idx']:
                continue
            # Skip consecutive duplicates
            if char_idx != prev_char_idx:
                if char_idx < len(self.vocabulary['chars']) - 1:  # Valid character
                    decoded_chars.append(self.vocabulary['idx_to_char'][char_idx])
                prev_char_idx = char_idx
        
        decoded_text = ''.join(decoded_chars)
        return decoded_text
    
    def evaluate_single_video(self, video_path, ground_truth=None):
        """
        Evaluate model on a single video
        """
        try:
            # Preprocess video
            video_tensor = self.preprocess_video(video_path)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(video_tensor)  # [1, 75, 28]
                
            # Decode prediction
            predicted_text = self.decode_prediction(outputs)
            
            print(f"🔮 Prediction: '{predicted_text}'")
            
            # Calculate metrics if ground truth provided
            if ground_truth:
                # Clean texts for comparison
                pred_clean = predicted_text.strip().upper()
                gt_clean = ground_truth.strip().upper()
                
                # Character Error Rate (CER)
                cer = editdistance.eval(pred_clean, gt_clean) / max(len(gt_clean), 1)
                
                # Word Error Rate (WER)
                pred_words = pred_clean.split()
                gt_words = gt_clean.split()
                wer = editdistance.eval(pred_words, gt_words) / max(len(gt_words), 1)
                
                print(f"📝 Ground truth: '{ground_truth}'")
                print(f"📊 CER: {cer:.4f} ({cer*100:.2f}%)")
                print(f"📊 WER: {wer:.4f} ({wer*100:.2f}%)")
                
                return predicted_text, cer, wer
            
            return predicted_text, None, None
            
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            return None, None, None
    
    def test_with_sample_videos(self):
        """
        Test with some sample videos (you'll need to provide these)
        """
        print("🧪 Testing with sample videos...")
        
        # Check if we have any test videos
        test_video_dir = "test_videos"
        if not os.path.exists(test_video_dir):
            print(f"📁 Creating {test_video_dir} directory...")
            os.makedirs(test_video_dir)
            print(f"📝 Please add some .mp4 test videos to {test_video_dir}/")
            print(f"💡 You can use:")
            print(f"   - Your own recorded videos")
            print(f"   - GRID corpus samples")
            print(f"   - Any lip movement videos")
            return
        
        # Look for video files (including .mpg)
        video_files = [f for f in os.listdir(test_video_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mpg', '.mkv', '.wmv'))]
        
        if not video_files:
            print(f"❌ No video files found in {test_video_dir}/")
            print(f"📝 Please add some test videos!")
            return
        
        print(f"📹 Found {len(video_files)} test videos")
        
        # Test each video
        results = []
        for video_file in video_files:
            video_path = os.path.join(test_video_dir, video_file)
            print(f"\n{'='*50}")
            print(f"🎬 Testing: {video_file}")
            
            # You can add ground truth here if you know what the video says
            predicted_text, cer, wer = self.evaluate_single_video(video_path)
            
            if predicted_text:
                print(f"✅ Successfully processed {video_file}")
                results.append((video_file, predicted_text))
            else:
                print(f"❌ Failed to process {video_file}")
                results.append((video_file, "FAILED"))
        
        # Summary of all results
        print(f"\n{'='*60}")
        print("📋 SUMMARY OF ALL PREDICTIONS:")
        print(f"{'='*60}")
        for video_file, prediction in results:
            print(f"🎬 {video_file}")
            print(f"🔮 Prediction: '{prediction}'")
            print("-" * 40)

def main():
    """Main evaluation function"""
    print("🎯 LipNet Baseline Evaluation")
    print("=" * 50)
    
    # Initialize evaluator with the best pre-trained model
    model_path = "pretrain/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt"
    
    evaluator = BaselineEvaluator(model_path)
    
    # Test with sample videos
    evaluator.test_with_sample_videos()
    
    print("\n🎯 Baseline evaluation setup complete!")
    print("📝 Next steps:")
    print("   1. Add test videos to test_videos/ folder")
    print("   2. Run evaluation on real data")
    print("   3. Compare with published benchmarks")
    print("   4. Start planning improvements!")

if __name__ == "__main__":
    main()