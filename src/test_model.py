"""
Test the trained stable transformer model and generate comprehensive results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model architecture from stable_transformer.py
from stable_transformer import StableLipNet, GRIDDataset, create_char_mappings, ctc_decode, calculate_wer

class ModelTester:
    def __init__(self, model_path='best_stable_transformer.pth', test_path='data/processed_grid_only/test'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.test_path = test_path
        
        # Load model and mappings
        self.load_model()
        
        # Create output directory
        self.output_dir = Path('test_results')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Model Tester initialized")
        print(f"Device: {self.device}")
        print(f"Test data: {self.test_path}")
        print(f"Output directory: {self.output_dir}")
    
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        # Load with weights_only=False for compatibility
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = checkpoint['vocab_size']
        self.blank_idx = self.vocab_size - 1
        
        # Create model
        self.model = StableLipNet(vocab_size=self.vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Best WER: {checkpoint['best_wer']:.4f}")
        print(f"Epoch: {checkpoint['epoch']}")
    
    def test_model(self):
        """Run comprehensive testing"""
        print("\nRunning comprehensive model testing...")
        
        # Load test dataset
        test_dataset = GRIDDataset(self.test_path)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, shuffle=False, num_workers=0
        )
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        speaker_results = defaultdict(list)
        
        # Process all test samples
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                videos = batch['video'].to(self.device)
                sentences = batch['sentence']
                speakers = batch['speaker_id']
                
                # Forward pass
                output = self.model(videos)
                output_probs = torch.softmax(output, dim=2)
                
                # Decode each sample
                for i in range(len(sentences)):
                    # Decode prediction
                    decoded = ctc_decode(output_probs[i], self.blank_idx)
                    pred_text = ''.join([self.idx_to_char.get(idx, '?') for idx in decoded])
                    
                    # Calculate confidence (average of max probabilities)
                    max_probs = torch.max(output_probs[i], dim=1)[0]
                    confidence = max_probs.mean().item()
                    
                    # Store results
                    all_predictions.append(pred_text)
                    all_targets.append(sentences[i])
                    all_confidences.append(confidence)
                    
                    # Store by speaker
                    wer = calculate_wer(pred_text, sentences[i])
                    speaker_results[speakers[i]].append(wer)
        
        # Calculate overall metrics
        results = self.calculate_metrics(all_predictions, all_targets, all_confidences, speaker_results)
        
        # Save detailed results
        self.save_detailed_results(all_predictions, all_targets, all_confidences, results)
        
        return results
    
    def calculate_metrics(self, predictions, targets, confidences, speaker_results):
        """Calculate comprehensive metrics"""
        print("\nCalculating metrics...")
        
        # Overall WER
        wers = [calculate_wer(p, t) for p, t in zip(predictions, targets)]
        overall_wer = np.mean(wers)
        
        # Character Error Rate (CER)
        cers = []
        for pred, target in zip(predictions, targets):
            pred_chars = list(pred)
            target_chars = list(target)
            cer = calculate_wer(''.join(pred_chars), ''.join(target_chars))
            cers.append(cer)
        overall_cer = np.mean(cers)
        
        # Per-speaker WER
        speaker_wers = {}
        for speaker, wers in speaker_results.items():
            speaker_wers[speaker] = np.mean(wers)
        
        # Confidence statistics
        conf_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        # Word-level accuracy
        word_correct = 0
        word_total = 0
        for pred, target in zip(predictions, targets):
            pred_words = pred.split()
            target_words = target.split()
            for pw, tw in zip(pred_words, target_words):
                word_total += 1
                if pw == tw:
                    word_correct += 1
        word_accuracy = word_correct / word_total if word_total > 0 else 0
        
        # Sentence-level accuracy
        sentence_correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        sentence_accuracy = sentence_correct / len(predictions)
        
        results = {
            'overall_wer': overall_wer,
            'overall_cer': overall_cer,
            'speaker_wers': speaker_wers,
            'confidence_stats': conf_stats,
            'word_accuracy': word_accuracy,
            'sentence_accuracy': sentence_accuracy,
            'total_samples': len(predictions),
            'perfect_predictions': sentence_correct
        }
        
        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
        print(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
        print(f"Word Accuracy: {word_accuracy:.4f} ({word_accuracy*100:.2f}%)")
        print(f"Sentence Accuracy: {sentence_accuracy:.4f} ({sentence_accuracy*100:.2f}%)")
        print(f"Perfect Predictions: {sentence_correct}/{len(predictions)}")
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {conf_stats['mean']:.4f}")
        print(f"  Std: {conf_stats['std']:.4f}")
        print(f"  Range: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
        print(f"\nPer-Speaker WER:")
        for speaker, wer in sorted(speaker_wers.items()):
            print(f"  {speaker}: {wer:.4f} ({wer*100:.2f}%)")
        
        return results
    
    def save_detailed_results(self, predictions, targets, confidences, results):
        """Save detailed results to file"""
        print("\nSaving detailed results...")
        
        # Create DataFrame with all results
        df = pd.DataFrame({
            'target': targets,
            'prediction': predictions,
            'confidence': confidences,
            'wer': [calculate_wer(p, t) for p, t in zip(predictions, targets)],
            'correct': [p == t for p, t in zip(predictions, targets)]
        })
        
        # Save to CSV
        csv_path = self.output_dir / 'test_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
        # Save summary
        summary_path = self.output_dir / 'test_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("STABLE TRANSFORMER LIPNET - TEST RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test samples: {results['total_samples']}\n")
            f.write(f"\nPERFORMANCE METRICS:\n")
            f.write(f"Overall WER: {results['overall_wer']:.4f} ({results['overall_wer']*100:.2f}%)\n")
            f.write(f"Overall CER: {results['overall_cer']:.4f} ({results['overall_cer']*100:.2f}%)\n")
            f.write(f"Word Accuracy: {results['word_accuracy']:.4f} ({results['word_accuracy']*100:.2f}%)\n")
            f.write(f"Sentence Accuracy: {results['sentence_accuracy']:.4f} ({results['sentence_accuracy']*100:.2f}%)\n")
            f.write(f"\nSPEAKER ANALYSIS:\n")
            for speaker, wer in sorted(results['speaker_wers'].items()):
                f.write(f"  {speaker}: WER={wer:.4f}\n")
            f.write(f"\nBEST PREDICTIONS:\n")
            # Find best predictions
            best_indices = df.nsmallest(10, 'wer').index
            for idx in best_indices:
                f.write(f"  Target: '{df.loc[idx, 'target']}'\n")
                f.write(f"  Pred:   '{df.loc[idx, 'prediction']}'\n")
                f.write(f"  WER: {df.loc[idx, 'wer']:.3f}, Conf: {df.loc[idx, 'confidence']:.3f}\n\n")
            
            f.write(f"\nWORST PREDICTIONS:\n")
            # Find worst predictions
            worst_indices = df.nlargest(10, 'wer').index
            for idx in worst_indices:
                f.write(f"  Target: '{df.loc[idx, 'target']}'\n")
                f.write(f"  Pred:   '{df.loc[idx, 'prediction']}'\n")
                f.write(f"  WER: {df.loc[idx, 'wer']:.3f}, Conf: {df.loc[idx, 'confidence']:.3f}\n\n")
        
        print(f"Saved summary to {summary_path}")
        
        return df
    
    def generate_confusion_analysis(self, predictions, targets):
        """Analyze common confusions"""
        print("\nAnalyzing common confusions...")
        
        word_confusions = defaultdict(Counter)
        char_confusions = defaultdict(Counter)
        
        for pred, target in zip(predictions, targets):
            # Word-level confusions
            pred_words = pred.split()
            target_words = target.split()
            
            for pw, tw in zip(pred_words, target_words):
                if pw != tw:
                    word_confusions[tw][pw] += 1
            
            # Character-level confusions
            for pc, tc in zip(pred, target):
                if pc != tc:
                    char_confusions[tc][pc] += 1
        
        # Save confusion analysis
        confusion_path = self.output_dir / 'confusion_analysis.txt'
        with open(confusion_path, 'w') as f:
            f.write("CONFUSION ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            f.write("TOP WORD CONFUSIONS:\n")
            # Get top confusions
            all_confusions = []
            for target_word, conf_dict in word_confusions.items():
                for pred_word, count in conf_dict.items():
                    all_confusions.append((count, target_word, pred_word))
            
            all_confusions.sort(reverse=True)
            for count, target, pred in all_confusions[:20]:
                f.write(f"  '{target}' -> '{pred}': {count} times\n")
            
            f.write("\n\nTOP CHARACTER CONFUSIONS:\n")
            # Get top character confusions
            char_conf_list = []
            for target_char, conf_dict in char_confusions.items():
                for pred_char, count in conf_dict.items():
                    char_conf_list.append((count, target_char, pred_char))
            
            char_conf_list.sort(reverse=True)
            for count, target, pred in char_conf_list[:20]:
                f.write(f"  '{target}' -> '{pred}': {count} times\n")
        
        print(f"Saved confusion analysis to {confusion_path}")

def main():
    """Main testing function"""
    print("TESTING STABLE TRANSFORMER LIPNET")
    print("="*60)
    
    # Create tester
    tester = ModelTester()
    
    # Run comprehensive testing
    results = tester.test_model()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = LipNetVisualizer(output_dir='test_results')
    
    # Load test results
    df = pd.read_csv('test_results/test_results.csv')
    
    # Generate plots
    visualizer.plot_wer_distribution(df)
    visualizer.plot_confidence_analysis(df)
    visualizer.plot_speaker_performance(results['speaker_wers'])
    visualizer.plot_error_analysis(df)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print(f"Results saved in: test_results/")
    print("\nKey findings:")
    print(f"- Best WER: {results['overall_wer']:.4f} ({results['overall_wer']*100:.2f}%)")
    print(f"- {results['perfect_predictions']} perfect predictions out of {results['total_samples']}")
    print(f"- Word accuracy: {results['word_accuracy']*100:.1f}%")
    
    # Analyze confusions
    predictions = df['prediction'].tolist()
    targets = df['target'].tolist()
    tester.generate_confusion_analysis(predictions, targets)

class LipNetVisualizer:
    """Visualization utilities for test results"""
    def __init__(self, output_dir='test_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'error': '#dc3545'
        }
    
    def plot_wer_distribution(self, df):
        """Plot WER distribution"""
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.hist(df['wer'], bins=30, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        plt.axvline(df['wer'].mean(), color=self.colors['error'], linestyle='--', linewidth=2, label=f'Mean: {df["wer"].mean():.3f}')
        
        plt.xlabel('Word Error Rate', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('WER Distribution Across Test Set', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'wer_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_analysis(self, df):
        """Plot confidence vs accuracy analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confidence distribution
        ax1.hist(df['confidence'], bins=30, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Confidence vs WER scatter
        scatter = ax2.scatter(df['confidence'], df['wer'], alpha=0.5, c=df['wer'], cmap='RdYlGn_r')
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Word Error Rate', fontsize=12)
        ax2.set_title('Confidence vs WER Correlation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('WER', fontsize=10)
        
        # Add correlation coefficient
        corr = df['confidence'].corr(df['wer'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_speaker_performance(self, speaker_wers):
        """Plot per-speaker performance"""
        plt.figure(figsize=(10, 6))
        
        speakers = list(speaker_wers.keys())
        wers = list(speaker_wers.values())
        
        bars = plt.bar(speakers, wers, color=[self.colors['primary'], self.colors['secondary'], self.colors['success']])
        
        # Add value labels
        for bar, wer in zip(bars, wers):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{wer:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Speaker', fontsize=12)
        plt.ylabel('Word Error Rate', fontsize=12)
        plt.title('Performance by Speaker', fontsize=14, fontweight='bold')
        plt.ylim(0, max(wers) * 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'speaker_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, df):
        """Plot error analysis by sentence length"""
        # Calculate sentence lengths
        df['target_length'] = df['target'].str.len()
        df['target_words'] = df['target'].str.split().str.len()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # WER vs sentence length
        length_bins = pd.cut(df['target_length'], bins=10)
        wer_by_length = df.groupby(length_bins)['wer'].mean()
        
        wer_by_length.plot(kind='bar', ax=ax1, color=self.colors['primary'], alpha=0.7)
        ax1.set_xlabel('Sentence Length (characters)', fontsize=12)
        ax1.set_ylabel('Average WER', fontsize=12)
        ax1.set_title('WER vs Sentence Length', fontsize=14, fontweight='bold')
        ax1.set_xticklabels([f'{int(i.left)}-{int(i.right)}' for i in wer_by_length.index], rotation=45)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # WER vs number of words
        word_bins = pd.cut(df['target_words'], bins=range(3, 9))
        wer_by_words = df.groupby(word_bins)['wer'].mean()
        
        wer_by_words.plot(kind='bar', ax=ax2, color=self.colors['secondary'], alpha=0.7)
        ax2.set_xlabel('Number of Words', fontsize=12)
        ax2.set_ylabel('Average WER', fontsize=12)
        ax2.set_title('WER vs Number of Words', fontsize=14, fontweight='bold')
        ax2.set_xticklabels([f'{int(i.left)}-{int(i.right)}' for i in wer_by_words.index], rotation=0)
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()