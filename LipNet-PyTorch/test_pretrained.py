import torch
from model import LipNet

# Check available pretrained weights
import os
pretrain_files = [f for f in os.listdir("pretrain") if f.endswith(".pt")]
print("📁 Available pretrained weights:")
for f in pretrain_files:
    print(f"  - {f}")

if pretrain_files:
    # Load the better model (overlap - lower WER)
    weight_path = "pretrain/LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt"
    print(f"\n🔄 Loading: {weight_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = LipNet()
    model = model.to(device)
    
    checkpoint = torch.load(weight_path, map_location=device)
    print(f"✅ Checkpoint loaded successfully!")
    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Model weights loaded from model_state_dict!")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded directly!")
    
    # Test model with dummy input (CORRECT DIMENSIONS!)
    model.eval()
    batch_size = 1
    channels = 3        # RGB channels  
    sequence_length = 75
    height, width = 64, 128  # CORRECT dimensions found!
    
    # Correct input format: [batch, channels, sequence, height, width]
    dummy_input = torch.randn(batch_size, channels, sequence_length, height, width).to(device)
    print(f"🧪 Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Forward pass successful!")
        print(f"📤 Output shape: {output.shape}")
        
        # Check output dimensions (should be [batch, sequence, vocab_size])
        print(f"📊 Output details:")
        print(f"   - Batch size: {output.shape[0]}")
        print(f"   - Sequence length: {output.shape[1]}")
        print(f"   - Vocabulary size: {output.shape[2]} (27 letters + 1 blank = 28)")
        
        # Test CTC decoding
        print(f"\n🔤 CTC Output Analysis:")
        print(f"   - Output represents probabilities over 28 classes")
        print(f"   - Classes: A-Z (26) + space (1) + CTC blank (1) = 28")
        print(f"   - Each timestep predicts character probabilities")
    
    print("\n🎯 BASELINE ESTABLISHED!")
    print(f"📊 Pre-trained performance:")
    print(f"   - Word Error Rate (WER): 4.64%")
    print(f"   - Character Error Rate (CER): 1.97%")
    print(f"🚀 Ready for fine-tuning experiments!")
        
else:
    print("❌ No .pt files found in pretrain folder")