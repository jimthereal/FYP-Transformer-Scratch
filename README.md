# LipNet-FYP: Deep Learning Lip-Reading System

**Final Year Project - Developing Lip-Reading System with Deep Learning to Enhance Communication for Hearing-Impaired Individuals**

## Project Overview

This project enhances existing lip-reading technology to bridge the gap between laboratory performance and real-world application, specifically targeting improved communication accessibility for hearing-impaired individuals.

### Our Enhancements (FYP Contributions)
- **Problem Identification**: Documented significant domain gap in real-world performance
- **Enhanced Preprocessing**: Advanced mouth region extraction with face landmarks
- **Evaluation Framework**: Comprehensive baseline testing and analysis
- **Performance Analysis**: Real-world vs laboratory condition comparison
- **[Phase 2]** Domain adaptation techniques
- **[Phase 3]** Real-time optimization and user interface

## Key Findings - Phase 1

### Technical Validation
- Successfully loaded and tested pre-trained models
- Implemented improved preprocessing pipeline
- Established baseline evaluation framework

### Critical Discovery: Domain Gap
- **GRID Corpus Performance**: 4.64% WER (excellent)
- **Real-world Performance**: Poor (gibberish output)
- **Root Cause**: Domain mismatch between training and deployment conditions

## Technical References

- **Original LipNet**: Assael, Y. M., et al. (2016). LipNet: End-to-End Sentence-level Lipreading
- **PyTorch Implementation**: VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch
- **GRID Corpus**: Cooke, M., et al. (2006). An audio-visual corpus for speech perception and automatic speech recognition
