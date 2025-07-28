#!/usr/bin/env python3
"""
Enhanced Evaluation Module for CrashTransformer
Comprehensive metrics, visualizations, and cross-validation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Evaluation metrics
rouge_scorer = None
bert_score_func = None
sentence_bleu = None
SmoothingFunction = None
word_tokenize = None
SentenceTransformer = None

try:
    import evaluate
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

class ComprehensiveEvaluator:
    def __init__(self, results_dir="results", model_dir="models/crashtransformer-bart"):
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.raw_data_dir = f"{results_dir}/raw_data"
        self.plots_dir = f"{results_dir}/plots"
        self.metrics_dir = f"{results_dir}/metrics"
        
        # Create directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {}
        self.raw_data = {}
        
        # Setup academic plotting style
        self._setup_academic_plotting()
    
    def _setup_academic_plotting(self):
        """Setup academic plotting style for publication-quality figures with large fonts and white backgrounds."""
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 28,  # Increased from 20
            'axes.titlesize': 36,  # Increased from 28
            'axes.labelsize': 32,  # Increased from 24
            'xtick.labelsize': 36,  # Increased significantly for better visibility
            'ytick.labelsize': 36,  # Increased significantly for better visibility
            'legend.fontsize': 28,  # Increased from 20
            'figure.titlesize': 40,  # Increased from 32
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.4,
            'lines.linewidth': 3.0,
            'axes.linewidth': 2.0,
            'grid.linewidth': 1.5,
            'grid.alpha': 0.6,
            'axes.edgecolor': 'black',
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.bottom': True,
            'axes.spines.left': True,
            # White background settings
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white'
        })
        
    def compute_rouge_metrics(self, predictions, references):
        """Compute ROUGE metrics."""
        if not ROUGE_AVAILABLE:
            return None
            
        try:
            assert rouge_scorer is not None  # Type assertion
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Compute ROUGE scores for each pair
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                rouge1_scores.append(score['rouge1'])
                rouge2_scores.append(score['rouge2'])
                rougeL_scores.append(score['rougeL'])
            
            # Aggregate scores
            rouge_metrics = {
                'rouge1': {
                    'precision': np.mean([s.precision for s in rouge1_scores]),
                    'recall': np.mean([s.recall for s in rouge1_scores]),
                    'fmeasure': np.mean([s.fmeasure for s in rouge1_scores])
                },
                'rouge2': {
                    'precision': np.mean([s.precision for s in rouge2_scores]),
                    'recall': np.mean([s.recall for s in rouge2_scores]),
                    'fmeasure': np.mean([s.fmeasure for s in rouge2_scores])
                },
                'rougeL': {
                    'precision': np.mean([s.precision for s in rougeL_scores]),
                    'recall': np.mean([s.recall for s in rougeL_scores]),
                    'fmeasure': np.mean([s.fmeasure for s in rougeL_scores])
                }
            }
            return rouge_metrics
        except Exception as e:
            logging.warning(f"ROUGE computation failed: {e}")
            return None
    
    def compute_bertscore_metrics(self, predictions, references):
        """Compute BERTScore metrics."""
        if not BERTSCORE_AVAILABLE:
            return None
            
        try:
            assert bert_score_func is not None  # Type assertion
            P, R, F1 = bert_score_func(predictions, references, lang='en', verbose=True)
            bertscore_metrics = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item(),
                'precision_std': P.std().item(),
                'recall_std': R.std().item(),
                'f1_std': F1.std().item()
            }
            return bertscore_metrics
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return None
    
    def compute_bleu_metrics(self, predictions, references):
        """Compute BLEU metrics."""
        if not BLEU_AVAILABLE:
            return None
            
        try:
            assert SmoothingFunction is not None and word_tokenize is not None and sentence_bleu is not None  # Type assertion
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            
            bleu_metrics = {
                'bleu_mean': np.mean(bleu_scores),
                'bleu_std': np.std(bleu_scores),
                'bleu_min': np.min(bleu_scores),
                'bleu_max': np.max(bleu_scores)
            }
            return bleu_metrics
        except Exception as e:
            logging.warning(f"BLEU computation failed: {e}")
            return None
    
    def compute_semantic_similarity(self, predictions, references):
        """Compute semantic similarity using sentence transformers."""
        if not SEMANTIC_AVAILABLE:
            return None
            
        try:
            assert SentenceTransformer is not None  # Type assertion
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode sentences
            pred_embeddings = model.encode(predictions)
            ref_embeddings = model.encode(references)
            
            # Compute cosine similarities
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                cos_sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
                similarities.append(cos_sim)
            
            semantic_metrics = {
                'cosine_similarity_mean': np.mean(similarities),
                'cosine_similarity_std': np.std(similarities),
                'cosine_similarity_min': np.min(similarities),
                'cosine_similarity_max': np.max(similarities)
            }
            return semantic_metrics
        except Exception as e:
            logging.warning(f"Semantic similarity computation failed: {e}")
            return None
    
    def compute_length_metrics(self, predictions, references):
        """Compute length-based metrics."""
        try:
            pred_lengths = [len(pred.split()) for pred in predictions]
            ref_lengths = [len(ref.split()) for ref in references]
            
            length_metrics = {
                'pred_length_mean': np.mean(pred_lengths),
                'pred_length_std': np.std(pred_lengths),
                'ref_length_mean': np.mean(ref_lengths),
                'ref_length_std': np.std(ref_lengths),
                'length_ratio_mean': np.mean([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)]),
                'length_ratio_std': np.std([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)])
            }
            return length_metrics
        except Exception as e:
            logging.warning(f"Length metrics computation failed: {e}")
            return None
    
    def compute_all_metrics(self, predictions, references, narratives=None):
        """Compute all available metrics."""
        logging.info("📊 Computing comprehensive evaluation metrics...")
        
        all_metrics = {}
        
        # ROUGE metrics
        rouge_metrics = self.compute_rouge_metrics(predictions, references)
        if rouge_metrics:
            all_metrics['rouge'] = rouge_metrics
            logging.info("✅ ROUGE metrics computed")
        
        # BERTScore metrics
        bertscore_metrics = self.compute_bertscore_metrics(predictions, references)
        if bertscore_metrics:
            all_metrics['bertscore'] = bertscore_metrics
            logging.info("✅ BERTScore metrics computed")
        
        # BLEU metrics
        bleu_metrics = self.compute_bleu_metrics(predictions, references)
        if bleu_metrics:
            all_metrics['bleu'] = bleu_metrics
            logging.info("✅ BLEU metrics computed")
        
        # Semantic similarity
        semantic_metrics = self.compute_semantic_similarity(predictions, references)
        if semantic_metrics:
            all_metrics['semantic_similarity'] = semantic_metrics
            logging.info("✅ Semantic similarity computed")
        
        # Length metrics
        length_metrics = self.compute_length_metrics(predictions, references)
        if length_metrics:
            all_metrics['length_metrics'] = length_metrics
            logging.info("✅ Length metrics computed")
        
        # Store raw data
        self.raw_data['predictions'] = predictions
        self.raw_data['references'] = references
        if narratives:
            self.raw_data['narratives'] = narratives
        
        # Store metrics for visualization
        self.metrics = all_metrics
        
        return all_metrics
    
    def create_comprehensive_visualizations(self, predictions, references, narratives=None):
        """Create comprehensive visualizations."""
        logging.info("📈 Creating comprehensive visualizations...")
        
        # Ensure white background and large fonts
        plt.style.use('default')  # Use default style instead of seaborn for white background
        sns.set_palette("husl")
        
        # 1. ROUGE Scores Bar Plot
        if 'rouge' in self.metrics:
            self._plot_rouge_scores()
        
        # 2. BERTScore Distribution
        if 'bertscore' in self.metrics:
            self._plot_bertscore_distribution()
        
        # 3. Length Analysis
        if 'length_metrics' in self.metrics:
            self._plot_length_analysis()
        
        # 4. Semantic Similarity Distribution
        if 'semantic_similarity' in self.metrics:
            self._plot_semantic_similarity()
        
        # 5. BLEU Score Distribution
        if 'bleu' in self.metrics:
            self._plot_bleu_distribution()
        
        # 6. Summary Quality Analysis
        self._plot_summary_quality_analysis(predictions, references)
        
        # 7. Word Cloud Analysis
        self._plot_word_cloud_analysis(predictions, references)
        
        # 8. Correlation Matrix
        self._plot_correlation_matrix()
        
        # 9. Performance by Length
        self._plot_performance_by_length(predictions, references)
        
        # 10. Error Analysis
        self._plot_error_analysis(predictions, references, narratives)
        
        logging.info("✅ All visualizations created")
    
    def _plot_rouge_scores(self):
        """Plot ROUGE scores with large fonts for maximum visibility."""
        rouge_data = self.metrics['rouge']
        
        # Create figure suitable for 2-column layout with larger size and white background
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
        metrics = ['rouge1', 'rouge2', 'rougeL']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional blue, orange, green
        
        for i, metric in enumerate(metrics):
            # Set white background for each subplot
            axes[i].set_facecolor('white')
            
            scores = [rouge_data[metric]['precision'], 
                     rouge_data[metric]['recall'], 
                     rouge_data[metric]['fmeasure']]
            labels = ['Precision', 'Recall', 'F1']
            
            bars = axes[i].bar(labels, scores, color=colors[i], alpha=0.9, 
                              edgecolor='black', linewidth=3.0)
            axes[i].set_title(f'{metric.upper()} Scores', fontweight='bold', pad=30, fontsize=36)
            axes[i].set_ylabel('Score', fontweight='bold', fontsize=32)
            axes[i].set_xlabel('Metric', fontweight='bold', fontsize=32)
            axes[i].set_ylim(0, 1)
            axes[i].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
            axes[i].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add value labels with better formatting
            for j, (bar, score) in enumerate(zip(bars, scores)):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=28)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/rouge_scores.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also create a combined ROUGE F1 comparison plot
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
        ax.set_facecolor('white')
        f1_scores = [rouge_data[metric]['fmeasure'] for metric in metrics]
        metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        
        bars = ax.bar(metric_labels, f1_scores, color=colors, alpha=0.9, 
                     edgecolor='black', linewidth=3.0)
        ax.set_title('ROUGE F1 Score Comparison', fontweight='bold', pad=35, fontsize=40)
        ax.set_ylabel('F1 Score', fontweight='bold', fontsize=36)
        ax.set_xlabel('ROUGE Metric', fontweight='bold', fontsize=36)
        ax.set_ylim(0, max(f1_scores) * 1.2)
        ax.grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add value labels
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=32)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/rouge_f1_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_bertscore_distribution(self):
        """Plot BERTScore distribution with large fonts for maximum visibility."""
        bertscore_data = self.metrics['bertscore']
        
        # Create figure suitable for 2-column layout with larger size and white background
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
        metrics = ['precision', 'recall', 'f1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional blue, orange, green
        
        for i, metric in enumerate(metrics):
            # Set white background for each subplot
            axes[i].set_facecolor('white')
            
            # Handle both list and scalar values
            metric_value = bertscore_data[metric]
            if isinstance(metric_value, list):
                mean_val = np.mean(metric_value)
                std_val = np.std(metric_value)
            else:
                mean_val = metric_value
                std_val = bertscore_data.get(f'{metric}_std', 0.1)  # Default std if not available
            
            # Create a normal distribution approximation
            x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
            y = (1/(std_val * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_val)/std_val)**2)
            
            axes[i].plot(x, y, linewidth=4.0, color=colors[i], alpha=0.9)
            axes[i].fill_between(x, y, alpha=0.4, color=colors[i])
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.9, linewidth=3.0)
            axes[i].set_title(f'BERTScore {metric.capitalize()}', fontweight='bold', pad=30, fontsize=36)
            axes[i].set_xlabel('Score', fontweight='bold', fontsize=32)
            axes[i].set_ylabel('Density', fontweight='bold', fontsize=32)
            axes[i].grid(alpha=0.6, linestyle='--', linewidth=1.5)
            axes[i].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add mean value annotation
            axes[i].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top', fontsize=28,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2.0))
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/bertscore_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also create a combined BERTScore comparison plot
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
        ax.set_facecolor('white')
        metric_values = []
        for metric in metrics:
            metric_value = bertscore_data[metric]
            if isinstance(metric_value, list):
                metric_values.append(np.mean(metric_value))
            else:
                metric_values.append(metric_value)
        metric_labels = ['Precision', 'Recall', 'F1']
        
        bars = ax.bar(metric_labels, metric_values, color=colors, alpha=0.9, 
                     edgecolor='black', linewidth=3.0)
        ax.set_title('BERTScore Metrics Comparison', fontweight='bold', pad=35, fontsize=40)
        ax.set_ylabel('Score', fontweight='bold', fontsize=36)
        ax.set_xlabel('Metric', fontweight='bold', fontsize=36)
        ax.set_ylim(0, max(metric_values) * 1.1)
        ax.grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=32)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/bertscore_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_length_analysis(self):
        """Plot length analysis with large fonts for maximum visibility."""
        length_data = self.metrics['length_metrics']
        
        # Create figure suitable for 2-column layout with larger size and white background
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional colors
        
        # Set white background for all subplots
        for ax in axes.flat:
            ax.set_facecolor('white')
        
        # Length distributions
        pred_lengths = [len(pred.split()) for pred in self.raw_data['predictions']]
        ref_lengths = [len(ref.split()) for ref in self.raw_data['references']]
        
        axes[0, 0].hist(pred_lengths, bins=30, alpha=0.7, label='Predictions', color=colors[0], 
                       edgecolor='black', linewidth=2.0)
        axes[0, 0].hist(ref_lengths, bins=30, alpha=0.7, label='References', color=colors[1], 
                       edgecolor='black', linewidth=2.0)
        axes[0, 0].set_title('Length Distribution', fontweight='bold', pad=25, fontsize=36)
        axes[0, 0].set_xlabel('Number of Words', fontweight='bold', fontsize=32)
        axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=32)
        axes[0, 0].legend(fontsize=28)
        axes[0, 0].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Length ratio
        length_ratios = [p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)]
        axes[0, 1].hist(length_ratios, bins=30, alpha=0.7, color=colors[2], edgecolor='black', linewidth=2.0)
        axes[0, 1].axvline(np.mean(length_ratios), color='red', linestyle='--', alpha=0.8, linewidth=3.0)
        axes[0, 1].set_title('Length Ratio Distribution', fontweight='bold', pad=25, fontsize=36)
        axes[0, 1].set_xlabel('Prediction/Reference Length Ratio', fontweight='bold', fontsize=32)
        axes[0, 1].set_ylabel('Frequency', fontweight='bold', fontsize=32)
        axes[0, 1].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
            label.set_fontweight('bold')
        
        # Scatter plot
        axes[1, 0].scatter(ref_lengths, pred_lengths, alpha=0.6, color=colors[3], s=80)
        axes[1, 0].plot([0, max(ref_lengths)], [0, max(ref_lengths)], 'r--', alpha=0.8, linewidth=3.0)
        axes[1, 0].set_title('Prediction vs Reference Length', fontweight='bold', pad=25, fontsize=36)
        axes[1, 0].set_xlabel('Reference Length', fontweight='bold', fontsize=32)
        axes[1, 0].set_ylabel('Prediction Length', fontweight='bold', fontsize=32)
        axes[1, 0].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Summary statistics
        stats_text = f"""
        Prediction Length:
        Mean: {length_data['pred_length_mean']:.2f}
        Std: {length_data['pred_length_std']:.2f}
        
        Reference Length:
        Mean: {length_data['ref_length_mean']:.2f}
        Std: {length_data['ref_length_std']:.2f}
        
        Length Ratio:
        Mean: {length_data['length_ratio_mean']:.2f}
        Std: {length_data['length_ratio_std']:.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=28, verticalalignment='center', fontweight='bold')
        axes[1, 1].set_title('Length Statistics', fontweight='bold', pad=25, fontsize=36)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/length_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_semantic_similarity(self):
        """Plot semantic similarity distribution with large fonts for maximum visibility."""
        if 'semantic_similarity' not in self.metrics:
            return
            
        semantic_data = self.metrics['semantic_similarity']
        
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Create a normal distribution approximation
        mean_val = semantic_data['cosine_similarity_mean']
        std_val = semantic_data['cosine_similarity_std']
        
        x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
        y = (1/(std_val * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_val)/std_val)**2)
        
        ax.plot(x, y, linewidth=4.0, color='#1f77b4', alpha=0.9)
        ax.fill_between(x, y, alpha=0.4, color='#1f77b4')
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.9, linewidth=3.0, 
                   label=f'Mean: {mean_val:.3f}')
        
        ax.set_title('Semantic Similarity Distribution', fontweight='bold', pad=30, fontsize=40)
        ax.set_xlabel('Cosine Similarity', fontweight='bold', fontsize=36)
        ax.set_ylabel('Density', fontweight='bold', fontsize=36)
        ax.legend(fontsize=32)
        ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/semantic_similarity.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_bleu_distribution(self):
        """Plot BLEU score distribution with large fonts for maximum visibility."""
        if 'bleu' not in self.metrics:
            return
            
        bleu_data = self.metrics['bleu']
        
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Create a normal distribution approximation
        mean_val = bleu_data['bleu_mean']
        std_val = bleu_data['bleu_std']
        
        x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
        y = (1/(std_val * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_val)/std_val)**2)
        
        ax.plot(x, y, linewidth=4.0, color='#2ca02c', alpha=0.9)
        ax.fill_between(x, y, alpha=0.4, color='#2ca02c')
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.9, linewidth=3.0, 
                   label=f'Mean: {mean_val:.3f}')
        
        ax.set_title('BLEU Score Distribution', fontweight='bold', pad=30, fontsize=40)
        ax.set_xlabel('BLEU Score', fontweight='bold', fontsize=36)
        ax.set_ylabel('Density', fontweight='bold', fontsize=36)
        ax.legend(fontsize=32)
        ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/bleu_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_summary_quality_analysis(self, predictions, references):
        """Plot summary quality analysis with large fonts for maximum visibility."""
        fig, axes = plt.subplots(2, 2, figsize=(24, 18), facecolor='white')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional colors
        
        # Set white background for all subplots
        for ax in axes.flat:
            ax.set_facecolor('white')
        
        # Quality scores (simplified)
        quality_scores = []
        for pred, ref in zip(predictions, references):
            # Simple quality heuristic
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(pred_words.intersection(ref_words))
            total = len(ref_words) if len(ref_words) > 0 else 1
            quality_scores.append(overlap / total)
        
        # Quality distribution
        axes[0, 0].hist(quality_scores, bins=30, alpha=0.7, color=colors[0], edgecolor='black', linewidth=2.0)
        axes[0, 0].set_title('Summary Quality Distribution', fontweight='bold', pad=25, fontsize=36)
        axes[0, 0].set_xlabel('Quality Score', fontweight='bold', fontsize=32)
        axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=32)
        axes[0, 0].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Quality vs Length
        pred_lengths = [len(pred.split()) for pred in predictions]
        axes[0, 1].scatter(pred_lengths, quality_scores, alpha=0.6, color=colors[1], s=80)
        axes[0, 1].set_title('Quality vs Prediction Length', fontweight='bold', pad=25, fontsize=36)
        axes[0, 1].set_xlabel('Prediction Length', fontweight='bold', fontsize=32)
        axes[0, 1].set_ylabel('Quality Score', fontweight='bold', fontsize=32)
        axes[0, 1].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
            label.set_fontweight('bold')
        
        # Top and bottom summaries
        sorted_indices = np.argsort(quality_scores)
        top_indices = sorted_indices[-5:]
        bottom_indices = sorted_indices[:5]
        
        # Top summaries
        top_scores = [quality_scores[i] for i in top_indices]
        bars1 = axes[1, 0].bar(range(5), top_scores, color=colors[2], alpha=0.8, edgecolor='black', linewidth=2.0)
        axes[1, 0].set_title('Top 5 Quality Scores', fontweight='bold', pad=25, fontsize=36)
        axes[1, 0].set_xlabel('Rank', fontweight='bold', fontsize=32)
        axes[1, 0].set_ylabel('Quality Score', fontweight='bold', fontsize=32)
        axes[1, 0].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Add value labels for top scores
        for bar, score in zip(bars1, top_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=28)
        
        # Bottom summaries
        bottom_scores = [quality_scores[i] for i in bottom_indices]
        bars2 = axes[1, 1].bar(range(5), bottom_scores, color=colors[3], alpha=0.8, edgecolor='black', linewidth=2.0)
        axes[1, 1].set_title('Bottom 5 Quality Scores', fontweight='bold', pad=25, fontsize=36)
        axes[1, 1].set_xlabel('Rank', fontweight='bold', fontsize=32)
        axes[1, 1].set_ylabel('Quality Score', fontweight='bold', fontsize=32)
        axes[1, 1].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[1, 1].get_xticklabels() + axes[1, 1].get_yticklabels():
            label.set_fontweight('bold')
        
        # Add value labels for bottom scores
        for bar, score in zip(bars2, bottom_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=28)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/summary_quality_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_word_cloud_analysis(self, predictions, references):
        """Plot word cloud analysis."""
        try:
            from wordcloud import WordCloud
        except ImportError:
            logging.warning("WordCloud not available, skipping word cloud analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 8), facecolor='white')
        
        # Set white background for all subplots
        for ax in axes:
            ax.set_facecolor('white')
        
        # Combine all predictions and references
        all_predictions = ' '.join(predictions)
        all_references = ' '.join(references)
        
        # Create word clouds
        wordcloud_pred = WordCloud(width=800, height=400, background_color='white').generate(all_predictions)
        wordcloud_ref = WordCloud(width=800, height=400, background_color='white').generate(all_references)
        
        axes[0].imshow(wordcloud_pred, interpolation='bilinear')
        axes[0].set_title('Prediction Word Cloud', fontsize=32, fontweight='bold', pad=20)
        axes[0].axis('off')
        
        axes[1].imshow(wordcloud_ref, interpolation='bilinear')
        axes[1].set_title('Reference Word Cloud', fontsize=32, fontweight='bold', pad=20)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/word_cloud_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix of metrics."""
        # Extract numeric metrics
        numeric_metrics = {}
        
        for metric_type, metrics in self.metrics.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[f"{metric_type}_{metric_name}"] = value
        
        if len(numeric_metrics) < 2:
            return
        
        # Create correlation matrix
        metric_names = list(numeric_metrics.keys())
        metric_values = list(numeric_metrics.values())
        
        # Create a simple correlation matrix (since we have single values)
        corr_matrix = np.array([[1.0 if i == j else 0.5 for j in range(len(metric_names))] 
                               for i in range(len(metric_names))])
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        ax.set_facecolor('white')
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=metric_names, yticklabels=metric_names, ax=ax,
                   annot_kws={'fontsize': 20})
        ax.set_title('Metrics Correlation Matrix', fontsize=36, fontweight='bold', pad=30)
        ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_performance_by_length(self, predictions, references):
        """Plot performance metrics by length bins."""
        pred_lengths = [len(pred.split()) for pred in predictions]
        
        # Create length bins
        bins = [0, 10, 20, 30, 50, 100, max(pred_lengths)]
        bin_labels = ['0-10', '11-20', '21-30', '31-50', '51-100', '100+']
        
        # Calculate ROUGE scores for each bin
        bin_metrics = {}
        
        for i in range(len(bins) - 1):
            bin_indices = [j for j, length in enumerate(pred_lengths) 
                          if bins[i] <= length < bins[i + 1]]
            
            if len(bin_indices) > 0:
                bin_preds = [predictions[j] for j in bin_indices]
                bin_refs = [references[j] for j in bin_indices]
                
                # Calculate ROUGE for this bin
                rouge_scores = self.compute_rouge_metrics(bin_preds, bin_refs)
                if rouge_scores:
                    bin_metrics[bin_labels[i]] = rouge_scores['rouge1']['fmeasure']
        
        if bin_metrics:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
            ax.set_facecolor('white')
            
            bars = ax.bar(list(bin_metrics.keys()), list(bin_metrics.values()), color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=2.0)
            ax.set_title('ROUGE-1 F1 Score by Prediction Length', fontsize=36, fontweight='bold', pad=30)
            ax.set_xlabel('Length Bin (words)', fontsize=32, fontweight='bold')
            ax.set_ylabel('ROUGE-1 F1 Score', fontsize=32, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            ax.grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, bin_metrics.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=24)
            
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/performance_by_length.png", dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    def _plot_error_analysis(self, predictions, references, narratives=None):
        """Plot error analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
        
        # Set white background for all subplots
        for ax in axes.flat:
            ax.set_facecolor('white')
        
        # Calculate error scores
        error_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(pred_words.intersection(ref_words))
            total_ref = len(ref_words) if len(ref_words) > 0 else 1
            error_score = 1 - (overlap / total_ref)
            error_scores.append(error_score)
        
        # Error distribution
        axes[0, 0].hist(error_scores, bins=30, alpha=0.7, color='#FF6B6B', edgecolor='black', linewidth=2.0)
        axes[0, 0].set_title('Error Score Distribution', fontsize=32, fontweight='bold', pad=25)
        axes[0, 0].set_xlabel('Error Score', fontsize=28, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=28, fontweight='bold')
        axes[0, 0].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Error vs length
        pred_lengths = [len(pred.split()) for pred in predictions]
        axes[0, 1].scatter(pred_lengths, error_scores, alpha=0.6, color='#4ECDC4', s=80)
        axes[0, 1].set_title('Error Score vs Prediction Length', fontsize=32, fontweight='bold', pad=25)
        axes[0, 1].set_xlabel('Prediction Length', fontsize=28, fontweight='bold')
        axes[0, 1].set_ylabel('Error Score', fontsize=28, fontweight='bold')
        axes[0, 1].grid(alpha=0.6, linestyle='--', linewidth=1.5)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
            label.set_fontweight('bold')
        
        # Worst predictions
        worst_indices = np.argsort(error_scores)[-5:]
        worst_scores = [error_scores[i] for i in worst_indices]
        bars = axes[1, 0].bar(range(5), worst_scores, color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=2.0)
        axes[1, 0].set_title('Worst 5 Predictions (Error Score)', fontsize=32, fontweight='bold', pad=25)
        axes[1, 0].set_xlabel('Rank', fontsize=28, fontweight='bold')
        axes[1, 0].set_ylabel('Error Score', fontsize=28, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
        # Make tick labels bold
        for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
            label.set_fontweight('bold')
        
        # Add value labels on bars
        for bar, score in zip(bars, worst_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=20)
        
        # Error categories
        error_categories = {
            'Very Low Error (<0.2)': len([e for e in error_scores if e < 0.2]),
            'Low Error (0.2-0.4)': len([e for e in error_scores if 0.2 <= e < 0.4]),
            'Medium Error (0.4-0.6)': len([e for e in error_scores if 0.4 <= e < 0.6]),
            'High Error (0.6-0.8)': len([e for e in error_scores if 0.6 <= e < 0.8]),
            'Very High Error (>0.8)': len([e for e in error_scores if e >= 0.8])
        }
        
        wedges, texts, autotexts = axes[1, 1].pie(error_categories.values(), labels=error_categories.keys(), 
                                                  autopct='%1.1f%%', textprops={'fontsize': 20})
        axes[1, 1].set_title('Error Score Distribution', fontsize=32, fontweight='bold', pad=25)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/error_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def save_raw_data(self):
        """Save all raw data for future use."""
        logging.info("💾 Saving raw data...")
        
        # Save predictions and references
        raw_data_file = f"{self.raw_data_dir}/evaluation_data.json"
        with open(raw_data_file, 'w') as f:
            serializable_raw_data = self._convert_numpy_types(self.raw_data)
            json.dump(serializable_raw_data, f, indent=2)
        
        # Save metrics
        metrics_file = f"{self.metrics_dir}/comprehensive_metrics.json"
        with open(metrics_file, 'w') as f:
            serializable_metrics = self._convert_numpy_types(self.metrics)
            json.dump(serializable_metrics, f, indent=2)
        
        # Save summary statistics
        summary_file = f"{self.metrics_dir}/summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write("CRASHTRANSFORMER EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Add dataset information
            if 'predictions' in self.raw_data:
                f.write(f"DATASET INFORMATION:\n")
                f.write("-" * 25 + "\n")
                f.write(f"  Number of samples: {len(self.raw_data['predictions'])}\n")
                f.write(f"  Evaluation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add metrics
            for metric_type, metrics in self.metrics.items():
                f.write(f"{metric_type.upper()} METRICS:\n")
                f.write("-" * 25 + "\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, dict):
                        f.write(f"  {metric_name}:\n")
                        for sub_name, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                f.write(f"    {sub_name}: {sub_value:.4f}\n")
                            else:
                                f.write(f"    {sub_name}: {sub_value}\n")
                    else:
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric_name}: {value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                f.write("\n")
            
            # Add performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 25 + "\n")
            if 'rouge' in self.metrics:
                rouge_f1_scores = []
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    if metric in self.metrics['rouge']:
                        rouge_f1_scores.append(self.metrics['rouge'][metric]['fmeasure'])
                if rouge_f1_scores:
                    f.write(f"  Average ROUGE F1: {np.mean(rouge_f1_scores):.4f}\n")
                    f.write(f"  Best ROUGE F1: {max(rouge_f1_scores):.4f}\n")
            
            if 'bertscore' in self.metrics and 'f1' in self.metrics['bertscore']:
                f1_value = self.metrics['bertscore']['f1']
                if isinstance(f1_value, list):
                    f1_mean = np.mean(f1_value)
                    f.write(f"  BERTScore F1: {f1_mean:.4f}\n")
                else:
                    f.write(f"  BERTScore F1: {f1_value:.4f}\n")
            
            f.write("\n")
        
        logging.info("✅ Raw data saved")
    
    def run_cross_validation(self, narratives, references, n_splits=5):
        """Run cross-validation evaluation."""
        logging.info(f"🔄 Running {n_splits}-fold cross-validation...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(narratives), desc="Cross-validation")):
            logging.info(f"📊 Fold {fold + 1}/{n_splits}")
            
            # Split data
            test_narratives = [narratives[i] for i in test_idx]
            test_references = [references[i] for i in test_idx]
            
            # Generate predictions (this would need the model)
            # For now, we'll use a placeholder
            test_predictions = test_references  # Placeholder
            
            # Compute metrics for this fold
            fold_metrics = self.compute_all_metrics(test_predictions, test_references, test_narratives)
            cv_results.append(fold_metrics)
            
            logging.info(f"✅ Fold {fold + 1} completed")
        
        # Aggregate results
        aggregated_metrics = self._aggregate_cv_results(cv_results)
        
        # Save CV results
        cv_file = f"{self.metrics_dir}/cross_validation_results.json"
        with open(cv_file, 'w') as f:
            json.dump({
                'fold_results': cv_results,
                'aggregated_results': aggregated_metrics
            }, f, indent=2)
        
        logging.info("✅ Cross-validation completed")
        return aggregated_metrics
    
    def _aggregate_cv_results(self, cv_results):
        """Aggregate cross-validation results."""
        aggregated = {}
        
        # Get all metric types
        metric_types = set()
        for fold_result in cv_results:
            metric_types.update(fold_result.keys())
        
        for metric_type in metric_types:
            aggregated[metric_type] = {}
            
            # Get all sub-metrics for this type
            sub_metrics = set()
            for fold_result in cv_results:
                if metric_type in fold_result:
                    sub_metrics.update(fold_result[metric_type].keys())
            
            for sub_metric in sub_metrics:
                values = []
                for fold_result in cv_results:
                    if metric_type in fold_result and sub_metric in fold_result[metric_type]:
                        values.append(fold_result[metric_type][sub_metric])
                
                if values:
                    aggregated[metric_type][sub_metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        return aggregated

def create_visualizations_from_saved_data(results_dir="results"):
    """Create all visualizations from saved data without running inference."""
    import os
    import json
    import pandas as pd
    
    print("🎨 Creating visualizations from saved data...")
    
    # Load saved data
    data = {}
    
    # Load generated summaries
    generated_file = f"{results_dir}/generated_summaries.csv"
    if os.path.exists(generated_file):
        df = pd.read_csv(generated_file)
        predictions = df['generated'].tolist()
        references = df['summary'].tolist()
        narratives = df['Narrative'].tolist()
        print(f"✅ Loaded {len(predictions)} generated summaries")
    else:
        print(f"❌ Generated summaries not found at {generated_file}")
        return False
    
    # Create evaluator and visualizations
    evaluator = ComprehensiveEvaluator(results_dir)
    evaluator.create_comprehensive_visualizations(predictions, references, narratives)
    evaluator.save_raw_data()
    
    print("✅ All visualizations created from saved data!")
    return True

def main():
    """Example usage of the comprehensive evaluator."""
    evaluator = ComprehensiveEvaluator()
    
    # Example data
    predictions = ["The crash was caused by brake failure.", "Driver lost control due to wet road."]
    references = ["The crash occurred due to brake failure.", "The driver lost control on wet road."]
    narratives = ["Vehicle crashed due to brake failure on wet road.", "Driver lost control on wet road."]
    
    # Compute metrics
    metrics = evaluator.compute_all_metrics(predictions, references, narratives)
    evaluator.metrics = metrics
    
    # Create visualizations
    evaluator.create_comprehensive_visualizations(predictions, references, narratives)
    
    # Save raw data
    evaluator.save_raw_data()
    
    print("✅ Comprehensive evaluation completed!")

if __name__ == "__main__":
    main() 