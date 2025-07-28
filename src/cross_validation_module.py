#!/usr/bin/env python3
"""
Cross-Validation Module for CrashTransformer
Implements k-fold cross-validation with comprehensive evaluation
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import Dataset
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class CrossValidator:
    def __init__(self, model_checkpoint="facebook/bart-base", results_dir="results"):
        self.model_checkpoint = model_checkpoint
        self.results_dir = results_dir
        self.cv_dir = f"{results_dir}/cross_validation"
        os.makedirs(self.cv_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Setup academic plotting style
        self._setup_academic_plotting()
    
    def _setup_academic_plotting(self):
        """Setup academic plotting style for publication-quality figures with large fonts and white backgrounds."""
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 28,
            'axes.titlesize': 36,
            'axes.labelsize': 32,
            'xtick.labelsize': 36,  # Increased significantly for better visibility
            'ytick.labelsize': 36,  # Increased significantly for better visibility
            'legend.fontsize': 28,
            'figure.titlesize': 40,
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
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for cross-validation."""
        logging.info("🔧 Setting up model and tokenizer for cross-validation...")
        self.tokenizer = BartTokenizer.from_pretrained(self.model_checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_checkpoint)
        logging.info("✅ Model and tokenizer setup completed")
    
    def tokenize_data(self, narratives, summaries):
        """Tokenize the data for training."""
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.setup_model_and_tokenizer()
            
        def tokenize_fn(batch):
            assert self.tokenizer is not None  # Type assertion
            model_inputs = self.tokenizer(batch["narrative"], max_length=512, truncation=True, padding="max_length")
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(batch["summary"], max_length=64, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Create dataset
        data = {"narrative": narratives, "summary": summaries}
        dataset = Dataset.from_dict(data)
        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        
        return tokenized_dataset
    
    def train_fold(self, train_dataset, val_dataset, fold_num, epochs=2):
        """Train model for one fold."""
        logging.info(f"🏋️ Training fold {fold_num}...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.cv_dir}/fold_{fold_num}",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{self.cv_dir}/logs",
            logging_steps=10,
            report_to="none"
        )
        
        # Initialize model for this fold
        fold_model = BartForConditionalGeneration.from_pretrained(self.model_checkpoint)
        
        # Setup trainer
        from transformers.data.data_collator import DataCollatorForSeq2Seq
        assert self.tokenizer is not None  # Type assertion
        trainer = Trainer(
            model=fold_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=fold_model)
        )
        
        # Train
        trainer.train()
        
        return trainer, fold_model
    
    def evaluate_fold(self, model, test_narratives, test_summaries):
        """Evaluate one fold."""
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.setup_model_and_tokenizer()
            
        model.eval()
        predictions = []
        
        for narrative in tqdm(test_narratives, desc="Generating predictions"):
            assert self.tokenizer is not None  # Type assertion
            inputs = self.tokenizer(narrative, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    max_length=64, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            predictions.append(summary)
        
        return predictions
    
    def compute_fold_metrics(self, predictions, references):
        """Compute metrics for one fold."""
        from .enhanced_evaluation import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator(self.results_dir)
        metrics = evaluator.compute_all_metrics(predictions, references)
        
        return metrics
    
    def run_cross_validation(self, narratives, summaries, n_splits=5, epochs=2):
        """Run k-fold cross-validation."""
        logging.info(f"🔄 Starting {n_splits}-fold cross-validation...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Initialize cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store results
        cv_results = {
            'fold_results': [],
            'aggregated_metrics': {},
            'training_history': [],
            'predictions': [],
            'references': []
        }
        
        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(narratives)):
            logging.info(f"📊 Fold {fold + 1}/{n_splits}")
            
            # Split data
            train_narratives = [narratives[i] for i in train_idx]
            train_summaries = [summaries[i] for i in train_idx]
            val_narratives = [narratives[i] for i in val_idx]
            val_summaries = [summaries[i] for i in val_idx]
            
            # Tokenize data
            train_dataset = self.tokenize_data(train_narratives, train_summaries)
            val_dataset = self.tokenize_data(val_narratives, val_summaries)
            
            # Train model
            trainer, fold_model = self.train_fold(train_dataset, val_dataset, fold + 1, epochs)
            
            # Evaluate model
            predictions = self.evaluate_fold(fold_model, val_narratives, val_summaries)
            
            # Compute metrics
            fold_metrics = self.compute_fold_metrics(predictions, val_summaries)
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'metrics': fold_metrics,
                'predictions': predictions,
                'references': val_summaries,
                'narratives': val_narratives
            }
            
            cv_results['fold_results'].append(fold_result)
            cv_results['predictions'].extend(predictions)
            cv_results['references'].extend(val_summaries)
            
            # Log fold results
            logging.info(f"✅ Fold {fold + 1} completed")
            if 'rouge' in fold_metrics:
                rouge_f1 = fold_metrics['rouge']['rouge1']['fmeasure']
                logging.info(f"   ROUGE-1 F1: {rouge_f1:.4f}")
        
        # Aggregate results
        cv_results['aggregated_metrics'] = self._aggregate_cv_metrics(cv_results['fold_results'])
        
        # Save results
        self._save_cv_results(cv_results)
        
        # Create visualizations
        self._create_cv_visualizations(cv_results)
        
        logging.info("✅ Cross-validation completed")
        return cv_results
    
    def _aggregate_cv_metrics(self, fold_results):
        """Aggregate metrics across all folds."""
        aggregated = {}
        
        # Get all metric types
        metric_types = set()
        for fold_result in fold_results:
            metric_types.update(fold_result['metrics'].keys())
        
        for metric_type in metric_types:
            aggregated[metric_type] = {}
            
            # Get all sub-metrics for this type
            sub_metrics = set()
            for fold_result in fold_results:
                if metric_type in fold_result['metrics']:
                    sub_metrics.update(fold_result['metrics'][metric_type].keys())
            
            for sub_metric in sub_metrics:
                values = []
                for fold_result in fold_results:
                    if metric_type in fold_result['metrics'] and sub_metric in fold_result['metrics'][metric_type]:
                        value = fold_result['metrics'][metric_type][sub_metric]
                        if isinstance(value, dict):
                            # Handle nested metrics (like ROUGE precision, recall, fmeasure)
                            for nested_metric, nested_value in value.items():
                                if isinstance(nested_value, (int, float)):
                                    values.append(nested_value)
                        elif isinstance(value, (int, float)):
                            values.append(value)
                
                if values:
                    aggregated[metric_type][sub_metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }
        
        return aggregated
    
    def _save_cv_results(self, cv_results):
        """Save cross-validation results."""
        logging.info("💾 Saving cross-validation results...")
        
        # Save main results
        results_file = f"{self.cv_dir}/cv_results.json"
        with open(results_file, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        # Save aggregated metrics
        metrics_file = f"{self.cv_dir}/aggregated_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(cv_results['aggregated_metrics'], f, indent=2)
        
        # Save summary
        summary_file = f"{self.cv_dir}/cv_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("CRASHTRANSFORMER CROSS-VALIDATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of folds: {len(cv_results['fold_results'])}\n")
            f.write(f"Total samples: {len(cv_results['predictions'])}\n\n")
            
            f.write("AGGREGATED METRICS:\n")
            f.write("-" * 20 + "\n")
            
            for metric_type, metrics in cv_results['aggregated_metrics'].items():
                f.write(f"\n{metric_type.upper()}:\n")
                for metric_name, values in metrics.items():
                    if isinstance(values, dict) and 'mean' in values:
                        f.write(f"  {metric_name}:\n")
                        f.write(f"    Mean: {values['mean']:.4f}\n")
                        f.write(f"    Std: {values['std']:.4f}\n")
                        f.write(f"    Min: {values['min']:.4f}\n")
                        f.write(f"    Max: {values['max']:.4f}\n")
        
        logging.info("✅ Cross-validation results saved")
    
    def _create_cv_visualizations(self, cv_results):
        """Create cross-validation visualizations with academic formatting."""
        logging.info("📈 Creating cross-validation visualizations...")
        
        # Use default style for white background
        plt.style.use('default')
        
        # 1. Fold Performance Comparison
        self._plot_fold_performance(cv_results)
        
        # 2. Metric Distribution Across Folds
        self._plot_metric_distribution(cv_results)
        
        # 3. Learning Curves
        self._plot_learning_curves(cv_results)
        
        # 4. Prediction Quality by Fold
        self._plot_prediction_quality_by_fold(cv_results)
        
        logging.info("✅ Cross-validation visualizations created")
    
    def _plot_fold_performance(self, cv_results):
        """Plot performance comparison across folds with large fonts for maximum visibility."""
        # Create figure suitable for 2-column layout with larger size and white background
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional colors
        
        # Set white background for all subplots
        for ax in axes.flat:
            ax.set_facecolor('white')
        
        # ROUGE-1 F1 scores
        rouge1_f1_scores = []
        for fold_result in cv_results['fold_results']:
            if 'rouge' in fold_result['metrics']:
                rouge1_f1_scores.append(fold_result['metrics']['rouge']['rouge1']['fmeasure'])
        
        if rouge1_f1_scores:
            bars = axes[0, 0].bar(range(1, len(rouge1_f1_scores) + 1), rouge1_f1_scores, 
                                 color=colors[0], alpha=0.8, edgecolor='black', linewidth=2.0)
            axes[0, 0].set_title('ROUGE-1 F1 Scores by Fold', fontweight='bold', pad=25, fontsize=36)
            axes[0, 0].set_xlabel('Fold', fontweight='bold', fontsize=32)
            axes[0, 0].set_ylabel('ROUGE-1 F1 Score', fontweight='bold', fontsize=32)
            axes[0, 0].set_xticks(range(1, len(rouge1_f1_scores) + 1))
            axes[0, 0].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
            axes[0, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add value labels
            for bar, score in zip(bars, rouge1_f1_scores):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', 
                               fontweight='bold', fontsize=28)
        
        # ROUGE-2 F1 scores
        rouge2_f1_scores = []
        for fold_result in cv_results['fold_results']:
            if 'rouge' in fold_result['metrics']:
                rouge2_f1_scores.append(fold_result['metrics']['rouge']['rouge2']['fmeasure'])
        
        if rouge2_f1_scores:
            bars = axes[0, 1].bar(range(1, len(rouge2_f1_scores) + 1), rouge2_f1_scores, 
                                 color=colors[1], alpha=0.8, edgecolor='black', linewidth=2.0)
            axes[0, 1].set_title('ROUGE-2 F1 Scores by Fold', fontweight='bold', pad=25, fontsize=36)
            axes[0, 1].set_xlabel('Fold', fontweight='bold', fontsize=32)
            axes[0, 1].set_ylabel('ROUGE-2 F1 Score', fontweight='bold', fontsize=32)
            axes[0, 1].set_xticks(range(1, len(rouge2_f1_scores) + 1))
            axes[0, 1].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
            axes[0, 1].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add value labels
            for bar, score in zip(bars, rouge2_f1_scores):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', 
                               fontweight='bold', fontsize=28)
        
        # ROUGE-L F1 scores
        rougeL_f1_scores = []
        for fold_result in cv_results['fold_results']:
            if 'rouge' in fold_result['metrics']:
                rougeL_f1_scores.append(fold_result['metrics']['rouge']['rougeL']['fmeasure'])
        
        if rougeL_f1_scores:
            bars = axes[1, 0].bar(range(1, len(rougeL_f1_scores) + 1), rougeL_f1_scores, 
                                 color=colors[2], alpha=0.8, edgecolor='black', linewidth=2.0)
            axes[1, 0].set_title('ROUGE-L F1 Scores by Fold', fontweight='bold', pad=25, fontsize=28)
            axes[1, 0].set_xlabel('Fold', fontweight='bold', fontsize=24)
            axes[1, 0].set_ylabel('ROUGE-L F1 Score', fontweight='bold', fontsize=24)
            axes[1, 0].set_xticks(range(1, len(rougeL_f1_scores) + 1))
            axes[1, 0].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
            axes[1, 0].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add value labels
            for bar, score in zip(bars, rougeL_f1_scores):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', 
                               fontweight='bold', fontsize=28)
        
        # Overall performance summary
        summary_text = f"""
        Cross-Validation Summary:
        
        ROUGE-1 F1:
        Mean: {np.mean(rouge1_f1_scores):.4f}
        Std: {np.std(rouge1_f1_scores):.4f}
        
        ROUGE-2 F1:
        Mean: {np.mean(rouge2_f1_scores):.4f}
        Std: {np.std(rouge2_f1_scores):.4f}
        
        ROUGE-L F1:
        Mean: {np.mean(rougeL_f1_scores):.4f}
        Std: {np.std(rougeL_f1_scores):.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=28, verticalalignment='center', fontweight='bold')
        axes[1, 1].set_title('Performance Summary', fontsize=36, fontweight='bold', pad=25)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.cv_dir}/fold_performance.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_metric_distribution(self, cv_results):
        """Plot metric distribution across folds with academic formatting."""
        # Create figure suitable for 2-column layout with white background
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Set white background for all subplots
        for ax in axes.flat:
            ax.set_facecolor('white')
        
        # ROUGE metrics distribution
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        rouge_data = {metric: [] for metric in rouge_metrics}
        
        for fold_result in cv_results['fold_results']:
            if 'rouge' in fold_result['metrics']:
                for metric in rouge_metrics:
                    rouge_data[metric].append(fold_result['metrics']['rouge'][metric]['fmeasure'])
        
        # Plot distributions
        for i, metric in enumerate(rouge_metrics):
            row, col = i // 2, i % 2
            axes[row, col].hist(rouge_data[metric], bins=10, alpha=0.7, color=colors[i], 
                               edgecolor='black', linewidth=2.0)
            axes[row, col].axvline(np.mean(rouge_data[metric]), color='red', linestyle='--', 
                                  alpha=0.8, linewidth=3.0)
            axes[row, col].set_title(f'{metric.upper()} F1 Distribution', fontweight='bold', pad=25, fontsize=36)
            axes[row, col].set_xlabel('F1 Score', fontweight='bold', fontsize=32)
            axes[row, col].set_ylabel('Frequency', fontweight='bold', fontsize=32)
            axes[row, col].grid(alpha=0.6, linestyle='--', linewidth=1.5)
            axes[row, col].tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
            # Make tick labels bold
            for label in axes[row, col].get_xticklabels() + axes[row, col].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add statistics annotation
            mean_val = np.mean(rouge_data[metric])
            std_val = np.std(rouge_data[metric])
            axes[row, col].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                               transform=axes[row, col].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2.0),
                               fontsize=24, fontweight='bold')
        
        # Summary statistics
        summary_text = f"""
        Metric Distributions:
        
        ROUGE-1:
        Mean: {np.mean(rouge_data['rouge1']):.4f}
        Std: {np.std(rouge_data['rouge1']):.4f}
        
        ROUGE-2:
        Mean: {np.mean(rouge_data['rouge2']):.4f}
        Std: {np.std(rouge_data['rouge2']):.4f}
        
        ROUGE-L:
        Mean: {np.mean(rouge_data['rougeL']):.4f}
        Std: {np.std(rouge_data['rougeL']):.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=28, verticalalignment='center', fontweight='bold')
        axes[1, 1].set_title('Distribution Summary', fontweight='bold', pad=25, fontsize=36)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.cv_dir}/metric_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_learning_curves(self, cv_results):
        """Plot learning curves with academic formatting."""
        # This would require access to training history
        # For now, create a placeholder plot with academic formatting
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')
        ax.text(0.5, 0.5, 'Learning curves would be plotted here\nif training history is available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=32, fontweight='bold')
        ax.set_title('Learning Curves', fontweight='bold', pad=30, fontsize=40)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.cv_dir}/learning_curves.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plt.close()
    
    def _plot_prediction_quality_by_fold(self, cv_results):
        """Plot prediction quality analysis by fold with academic formatting."""
        # Create figure suitable for 2-column layout with white background
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Set white background for all subplots
        for ax in axes.flat:
            ax.set_facecolor('white')
        
        # Quality scores by fold
        quality_scores_by_fold = []
        for fold_result in cv_results['fold_results']:
            fold_quality = []
            for pred, ref in zip(fold_result['predictions'], fold_result['references']):
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                overlap = len(pred_words.intersection(ref_words))
                total = len(ref_words) if len(ref_words) > 0 else 1
                quality = overlap / total
                fold_quality.append(quality)
            quality_scores_by_fold.append(np.mean(fold_quality))
        
        # Plot quality by fold
        if quality_scores_by_fold:
            bars = axes[0, 0].bar(range(1, len(quality_scores_by_fold) + 1), quality_scores_by_fold, 
                                 color=colors[0], alpha=0.8, edgecolor='black', linewidth=2.0)
            axes[0, 0].set_title('Average Quality Score by Fold', fontweight='bold', pad=25, fontsize=36)
            axes[0, 0].set_xlabel('Fold', fontweight='bold', fontsize=32)
            axes[0, 0].set_ylabel('Quality Score', fontweight='bold', fontsize=32)
            axes[0, 0].set_xticks(range(1, len(quality_scores_by_fold) + 1))
            axes[0, 0].grid(axis='y', alpha=0.6, linestyle='--', linewidth=1.5)
            axes[0, 0].tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
            
            # Add value labels
            for bar, score in zip(bars, quality_scores_by_fold):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', 
                               fontweight='bold', fontsize=28)
        
        # Quality distribution across all folds
        all_quality_scores = []
        for fold_result in cv_results['fold_results']:
            for pred, ref in zip(fold_result['predictions'], fold_result['references']):
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                overlap = len(pred_words.intersection(ref_words))
                total = len(ref_words) if len(ref_words) > 0 else 1
                quality = overlap / total
                all_quality_scores.append(quality)
        
        if all_quality_scores:
            axes[0, 1].hist(all_quality_scores, bins=30, alpha=0.7, color=colors[1], 
                           edgecolor='black', linewidth=2.0)
            axes[0, 1].axvline(np.mean(all_quality_scores), color='red', linestyle='--', 
                              alpha=0.8, linewidth=3.0)
            axes[0, 1].set_title('Quality Score Distribution (All Folds)', fontweight='bold', pad=25, fontsize=36)
            axes[0, 1].set_xlabel('Quality Score', fontweight='bold', fontsize=32)
            axes[0, 1].set_ylabel('Frequency', fontweight='bold', fontsize=32)
            axes[0, 1].grid(alpha=0.6, linestyle='--', linewidth=1.5)
            axes[0, 1].tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
            
            # Add mean annotation
            mean_val = np.mean(all_quality_scores)
            axes[0, 1].text(0.02, 0.98, f'Mean: {mean_val:.3f}', 
                           transform=axes[0, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2.0),
                           fontsize=24, fontweight='bold')
        
        # Quality vs length
        all_lengths = []
        for fold_result in cv_results['fold_results']:
            for pred in fold_result['predictions']:
                all_lengths.append(len(pred.split()))
        
        if all_lengths and all_quality_scores:
            axes[1, 0].scatter(all_lengths, all_quality_scores, alpha=0.6, color=colors[2], s=80)
            axes[1, 0].set_title('Quality vs Prediction Length', fontweight='bold', pad=25, fontsize=36)
            axes[1, 0].set_xlabel('Prediction Length', fontweight='bold', fontsize=32)
            axes[1, 0].set_ylabel('Quality Score', fontweight='bold', fontsize=32)
            axes[1, 0].grid(alpha=0.6, linestyle='--', linewidth=1.5)
            axes[1, 0].tick_params(axis='both', which='major', labelsize=28, width=2.0, length=8)
        
        # Summary statistics
        summary_text = f"""
        Quality Analysis:
        
        Overall Quality:
        Mean: {np.mean(all_quality_scores):.4f}
        Std: {np.std(all_quality_scores):.4f}
        
        Quality by Fold:
        Best: {np.max(quality_scores_by_fold):.4f}
        Worst: {np.min(quality_scores_by_fold):.4f}
        
        Prediction Length:
        Mean: {np.mean(all_lengths):.2f}
        Std: {np.std(all_lengths):.2f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=28, verticalalignment='center', fontweight='bold')
        axes[1, 1].set_title('Quality Summary', fontweight='bold', pad=25, fontsize=36)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.cv_dir}/prediction_quality.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def main():
    """Example usage of cross-validation."""
    # Load your data
    # narratives = [...]  # Your crash narratives
    # summaries = [...]   # Your causal summaries
    
    # Initialize cross-validator
    cv = CrossValidator()
    
    # Run cross-validation
    # results = cv.run_cross_validation(narratives, summaries, n_splits=5, epochs=2)
    
    print("Cross-validation module ready!")

if __name__ == "__main__":
    main() 