# Attribution and Acknowledgments

## Original Work
This project builds upon the LipNet implementation by:
- **Repository**: https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch
- **Authors**: VIPL Audio-Visual Speech Understanding Research Group
- **Paper**: Assael, Y. M., et al. (2016). LipNet: End-to-End Sentence-level Lipreading

## What We Used From Original Work
- model.py → src/models/lipnet_base.py: Core LipNet architecture
- dataset.py → src/models/dataset_base.py: Data loading utilities  
- Pre-trained weights: Baseline models for comparison and fine-tuning
- GRID corpus training methodology: Reference implementation

## Our Original Contributions (Jimmy Yeow Kai Jim - FYP 2024/2025)
- **Enhanced Preprocessing**: src/improved_preprocessing.py
- **Baseline Evaluation Framework**: src/baseline_evaluation.py
- **Domain Adaptation Techniques**: [To be implemented in Phase 2]
- **Real-world Performance Analysis**: Identification and documentation of domain gap
- **Speaker Adaptation Methods**: [To be implemented in Phase 2]
- **Real-time Optimization**: [To be implemented in Phase 3]

## Files Modified from Original
- src/models/lipnet_base.py: Added FYP-specific modifications (clearly marked)
- All other files in src/ are original FYP contributions

## Academic Integrity Statement
This project is submitted as original work for Final Year Project requirements.
All use of existing code is properly attributed and falls within fair use for 
academic research and improvement purposes.

## License Compliance
We acknowledge and comply with all original license terms.
