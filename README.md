# Deep Learning-Based Lip-Reading System

**Final Year Project - Bachelor of Computer Science (AI)**  

## **Project Status: MILESTONE ACHIEVED - Stable Transformer Model**

**Current Model**: Stable Transformer trained on GRID Corpus  
**Performance**: 17.42% WER on test set (600 samples)  
**Architecture**: 4-layer transformer with 3D CNN backbone  
**Status**: Completed as investigation milestone  

---

## **Major Achievements**

### **Model Development Timeline**
| Phase | Dataset | Samples | WER | Status |
|-------|---------|---------|-----|--------|
| Initial | GRID-only | 20 | 50.8% | Overfitting issues |
| Scaled | GRID-only | 100 | 71.4% | Pattern memorization |
| **Stable** | **GRID (3 speakers)** | **3000** | **17.42%** | **Milestone achieved** |

### **Technical Accomplishments**
- **Stable Architecture**: 4-layer transformer with consistent performance
- **Successful Training**: 17.42% WER demonstrates effective learning
- **Complete Pipeline**: End-to-end system from video to text
- **Web Deployment**: Functional Flask backend with UI

---

## **System Architecture**

### **Data Processing Pipeline**
```
GRID Corpus Videos (.mpg) →
Frame Extraction (75 frames) →
MediaPipe Lip Detection →
Normalization (64x128) →
[1, 75, 128, 64] Tensors →
Train: 2400 / Test: 600
```

### **Stable Transformer Architecture**
```
3D CNN Backbone (32→64→128 channels) →
Adaptive Pooling (75, 4, 2) →
Feature Projection (256D) →
Positional Encoding →
4-Layer Transformer (8 heads) →
CTC Output (28 classes)
```

### **Model Specifications**
- **Parameters**: 4,897,564 (4.9M)
- **Architecture**: Stable Transformer
- **d_model**: 256
- **Layers**: 4 transformer encoder layers
- **Attention Heads**: 8
- **Training Time**: ~2 hours (50 epochs)

---

## **Performance Results**

### **Test Set Evaluation (600 samples)**
| Metric | Value | Description |
|--------|-------|-------------|
| Overall WER | 17.42% | Word Error Rate |
| Character Error Rate | 17.42% | Character-level errors |
| Word Accuracy | 79.84% | Correctly predicted words |
| Sentence Accuracy | 31.33% | Perfect predictions (188/600) |
| Mean Confidence | 97.36% | Model confidence score |

### **Per-Speaker Performance**
- **Speaker s23**: 13.78% WER (best)
- **Speaker s1**: 18.64% WER
- **Speaker s10**: 19.85% WER (challenging)

### **Common Confusions**
- **Words**: 'AT' ↔ 'IN', 'AT' ↔ 'AN'
- **Characters**: 'T' ↔ 'N', 'A' ↔ 'I'

---

## **Project Structure**
```
LipNet-FYP/
├── src/
│   ├── stable_transformer.py       # Model architecture & training
│   ├── test_model.py              # Testing & evaluation
│   ├── visualizations.py          # Report visualizations
│   ├── flask_backend.py           # Web API backend
│   └── templates/
│       └── index.html             # Web interface
├── data/
│   └── processed_grid_only/       # GRID dataset (not uploaded)
│       ├── train/                 # 2400 samples
│       └── test/                  # 600 samples
├── models/
│   └── best_stable_transformer.pth # Trained model
├── results/                       # (not uploaded)
│   ├── test_results.csv           # All predictions
│   ├── test_summary.txt           # Performance summary
│   └── confusion_analysis.txt     # Error analysis
└── README.md                      # This file
```

---

## **Web Interface**

### **Features**
- **Live Camera**: Real-time lip reading (webcam)
- **Video Upload**: Process recorded videos
- **Confidence Display**: Visual confidence meter
- **REST API**: Integration endpoints

### **Running the System**
```bash
# Install dependencies
pip install torch torchvision flask flask-cors opencv-python mediapipe

# Start the backend
cd src
python flask_backend.py

# Access the interface
# http://localhost:5000
```

---

## **Technical Analysis**

### **Strengths**
- **Good GRID Performance**: 17.42% WER shows effective learning
- **Stable Training**: No overfitting, consistent convergence
- **High Confidence**: Model is confident in predictions
- **Complete System**: End-to-end pipeline implemented

### **Limitations**
- **Dataset Specific**: Only works well on GRID vocabulary
- **Limited Generalization**: Struggles with non-GRID content
- **Speaker Dependent**: Performance varies by speaker
- **Controlled Environment**: Requires good lighting/angles

---

## **Lessons Learned**

1. **Dataset Matters**: GRID alone insufficient for real-world use
2. **Architecture Success**: Transformer outperforms LSTM baselines
3. **CTC Works Well**: Effective for sequence alignment
4. **Need Diversity**: Multiple datasets crucial for generalization

---

## **Conclusion**

This stable transformer model represents a significant milestone in the project, demonstrating:
- Successful implementation of transformer-based lip reading
- Good performance on controlled dataset (17.42% WER)
- Complete end-to-end system with web interface
- Clear understanding of limitations and next steps

While not suitable for production use due to GRID-specific training, this model validates the technical approach and provides a solid foundation for future multi-dataset training.

---

## **Acknowledgments**

- **GRID Corpus**: Cooke et al. (2006)
- **PyTorch Community**: Deep learning framework

---

*This project milestone demonstrates the feasibility of transformer-based lip reading while highlighting the importance of diverse training data for real-world applications.*