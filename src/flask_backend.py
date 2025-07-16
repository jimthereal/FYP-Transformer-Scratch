"""
Flask Backend for Stable Transformer LipNet
Corrected to match the trained model architecture
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import tempfile
import os
import pickle
import math
from pathlib import Path
import traceback
import warnings
warnings.filterwarnings('ignore')

# Import the EXACT model architecture you trained with
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=75):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class StableLipNet(nn.Module):
    """The EXACT model architecture from stable_transformer.py"""
    def __init__(self, vocab_size=28, d_model=256):
        super(StableLipNet, self).__init__()
        
        self.d_model = d_model
        
        # Simpler 3D CNN backbone
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv3d(1, 32, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            # Conv Block 2
            nn.Conv3d(32, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            
            # Conv Block 3
            nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling to fixed size
            nn.AdaptiveAvgPool3d((75, 4, 2))
        )
        
        # Feature projection with layer norm
        self.feature_size = 128 * 4 * 2
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder (fewer layers for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x shape: [B, 1, T, H, W]
        batch_size = x.size(0)
        
        # 3D CNN feature extraction
        features = self.backbone(x)  # [B, 128, 75, 4, 2]
        
        # Reshape for sequence processing
        B, C, T, H, W = features.size()
        features = features.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        features = features.reshape(B, T, -1)  # [B, T, C*H*W]
        
        # Project to model dimension
        features = self.feature_proj(features)  # [B, T, d_model]
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Transformer encoding
        encoded = self.transformer(features)  # [B, T, d_model]
        
        # Output projection
        output = self.output_proj(encoded)  # [B, T, vocab_size]
        
        return output

class LipNetInference:
    """Inference class for the Stable Transformer model"""
    
    def __init__(self, model_path="best_stable_transformer.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Loading model from: {model_path}")
        
        # Load model
        self.model, self.char_to_idx, self.idx_to_char, self.vocab_size = self._load_model(model_path)
        self.blank_idx = self.vocab_size - 1
        
        self.target_size = (64, 128)
        self.sequence_length = 75
        
        # Initialize MediaPipe if available
        self.face_detection = self._setup_mediapipe()
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get character mappings
            char_to_idx = checkpoint['char_to_idx']
            idx_to_char = checkpoint['idx_to_char']
            vocab_size = checkpoint['vocab_size']
            
            # Create model with correct architecture
            model = StableLipNet(vocab_size=vocab_size, d_model=256).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Model loaded successfully!")
            print(f"Best WER: {checkpoint.get('best_wer', 'N/A'):.4f}")
            print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"Vocabulary size: {vocab_size}")
            
            return model, char_to_idx, idx_to_char, vocab_size
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
    
    def _setup_mediapipe(self):
        """Setup MediaPipe for lip detection"""
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            )
            print("MediaPipe initialized for lip detection")
            return face_detection
        except Exception as e:
            print(f"MediaPipe not available: {e}")
            return None
    
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
            
            # Try MediaPipe if available
            if self.face_detection is not None:
                try:
                    results = self.face_detection.process(rgb_frame)
                    
                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        h, w = gray.shape[:2]
                        
                        # Extract mouth region
                        face_x = int(bbox.xmin * w)
                        face_y = int(bbox.ymin * h)
                        face_w = int(bbox.width * w)
                        face_h = int(bbox.height * h)
                        
                        mouth_y = face_y + int(face_h * 0.6)
                        mouth_h = int(face_h * 0.4)
                        mouth_x = face_x + int(face_w * 0.2)
                        mouth_w = int(face_w * 0.6)
                        
                        # Bounds checking
                        mouth_x = max(0, mouth_x)
                        mouth_y = max(0, mouth_y)
                        mouth_x2 = min(w, mouth_x + mouth_w)
                        mouth_y2 = min(h, mouth_y + mouth_h)
                        
                        if mouth_x2 > mouth_x and mouth_y2 > mouth_y:
                            mouth_region = gray[mouth_y:mouth_y2, mouth_x:mouth_x2]
                            
                            if mouth_region.size > 0:
                                mouth_resized = cv2.resize(mouth_region, self.target_size, interpolation=cv2.INTER_CUBIC)
                                return cv2.equalizeHist(mouth_resized)
                
                except Exception:
                    pass
            
            # Geometric fallback
            h, w = gray.shape
            start_y = int(h * 0.65)
            end_y = int(h * 0.95)
            start_x = int(w * 0.25)
            end_x = int(w * 0.75)
            
            mouth_region = gray[start_y:end_y, start_x:end_x]
            
            if mouth_region.size > 0:
                mouth_resized = cv2.resize(mouth_region, self.target_size, interpolation=cv2.INTER_CUBIC)
                return cv2.equalizeHist(mouth_resized)
            
        except Exception as e:
            print(f"Error extracting lip region: {e}")
        
        return np.zeros(self.target_size[::-1], dtype=np.uint8)
    
    def process_video(self, video_path):
        """Process video file and return transcription"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames
            frame_indices = np.linspace(0, total_frames-1, self.sequence_length).astype(int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    lip_region = self.extract_lip_region(frame)
                    frames.append(lip_region)
                else:
                    # Use last frame or zeros
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            cap.release()
            
            # Ensure correct sequence length
            while len(frames) < self.sequence_length:
                frames.append(frames[-1] if frames else np.zeros(self.target_size[::-1], dtype=np.uint8))
            
            frames = frames[:self.sequence_length]
            
            # Convert to tensor
            frames_array = np.stack(frames).astype(np.float32) / 255.0
            video_tensor = torch.FloatTensor(frames_array).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(video_tensor)
                output_probs = torch.softmax(output, dim=2)
                
                # CTC decode
                pred_sequence = self._ctc_decode(output_probs[0])
                transcription = self._sequence_to_text(pred_sequence)
                
                # Calculate confidence
                max_probs = torch.max(output_probs, dim=2)[0]
                confidence = torch.mean(max_probs).item()
            
            return {
                'transcription': transcription,
                'confidence': confidence,
                'success': True,
                'model_type': 'stable_transformer'
            }
            
        except Exception as e:
            print(f"Error processing video: {e}")
            traceback.print_exc()
            return {
                'error': str(e),
                'success': False
            }
    
    def _ctc_decode(self, output_probs):
        """CTC decoding"""
        pred_sequence = torch.argmax(output_probs, dim=1).cpu().numpy()
        
        # CTC collapse
        collapsed = []
        prev_char = None
        
        for char_idx in pred_sequence:
            if char_idx != self.blank_idx and char_idx != prev_char:
                collapsed.append(char_idx)
            prev_char = char_idx
        
        return collapsed
    
    def _sequence_to_text(self, sequence):
        """Convert sequence to text"""
        return ''.join([self.idx_to_char.get(idx, '?') for idx in sequence])

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize inference engine
try:
    lipnet = LipNetInference(model_path="best_stable_transformer.pth")
    print("LipNet inference engine initialized successfully!")
except Exception as e:
    print(f"Failed to initialize LipNet: {e}")
    lipnet = None

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html') if os.path.exists('templates/index.html') else """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LipNet Web Interface</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            .info { max-width: 600px; margin: 0 auto; text-align: left; }
            .status { color: green; font-weight: bold; }
            .error { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>LipNet Flask Backend Running</h1>
        <div class="info">
            <p class="status">Backend is running successfully!</p>
            <p>Model Type: """ + (lipnet.model.__class__.__name__ if lipnet else "Not loaded") + """</p>
            <p>Device: """ + (str(lipnet.device) if lipnet else "N/A") + """</p>
            <p>Best WER: """ + (f"{lipnet.model_info['best_wer']:.4f}" if lipnet and hasattr(lipnet, 'model_info') else "N/A") + """</p>
            
            <h3>To use the web interface:</h3>
            <ol>
                <li>Save the HTML file as 'templates/index.html'</li>
                <li>Create a 'templates' folder in the same directory as this script</li>
                <li>Move index.html to the templates folder</li>
                <li>Restart this Flask server</li>
            </ol>
            
            <h3>API Endpoints:</h3>
            <ul>
                <li>POST /api/process-video - Process uploaded video</li>
                <li>GET /api/status - Check system status</li>
                <li>GET /api/model-info - Get model information</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.route('/api/process-video', methods=['POST'])
def process_video_api():
    """API endpoint to process uploaded video"""
    try:
        if lipnet is None:
            return jsonify({
                'success': False,
                'error': 'LipNet model not initialized'
            }), 500
        
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No video file selected'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process video
            result = lipnet.process_video(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    except Exception as e:
        print(f"Error in process_video_api: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check system status"""
    return jsonify({
        'status': 'running',
        'model_loaded': lipnet is not None,
        'model_type': lipnet.model.__class__.__name__ if lipnet else 'N/A',
        'device': str(lipnet.device) if lipnet else 'N/A',
        'vocab_size': lipnet.vocab_size if lipnet else 'N/A',
        'blank_idx': lipnet.blank_idx if lipnet else 'N/A'
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if lipnet is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Count model parameters
    total_params = sum(p.numel() for p in lipnet.model.parameters())
    
    return jsonify({
        'architecture': 'Stable Transformer LipNet',
        'model_class': lipnet.model.__class__.__name__,
        'vocab_size': lipnet.vocab_size,
        'sequence_length': lipnet.sequence_length,
        'target_size': lipnet.target_size,
        'device': str(lipnet.device),
        'total_parameters': f"{total_params:,}",
        'd_model': 256,
        'transformer_layers': 4,
        'features': [
            '3D CNN backbone with 3 blocks',
            'Sinusoidal positional encoding',
            'Transformer encoder with 4 layers',
            'CTC decoding for sequence alignment',
            'MediaPipe lip detection',
            'Real-time inference capable',
            f'Best WER: 17.42%'
        ]
    })

if __name__ == '__main__':
    print("Starting Stable Transformer LipNet Flask Backend...")
    print("=" * 60)
    
    if lipnet is not None:
        print("Model loaded successfully")
        print(f"Model type: {lipnet.model.__class__.__name__}")
        print(f"Device: {lipnet.device}")
        print(f"Vocabulary size: {lipnet.vocab_size}")
        
        # Count parameters
        total_params = sum(p.numel() for p in lipnet.model.parameters())
        print(f"Total parameters: {total_params:,}")
    else:
        print("Model failed to load")
    
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print()
    print("To use the web interface:")
    print("1. Create a 'templates' folder")
    print("2. Save index.html to templates/index.html")
    print("3. Access http://localhost:5000")
    print()
    print("API endpoints:")
    print("- POST /api/process-video")
    print("- GET /api/status") 
    print("- GET /api/model-info")
    
    app.run(debug=True, host='0.0.0.0', port=5000)