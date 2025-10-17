#!/usr/bin/env python3
"""
Create training visualizations from saved training data
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

def create_loss_plot(training_dir: str, output_dir: str = None):
    """Create loss curve visualization"""
    
    if output_dir is None:
        output_dir = os.path.join(training_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training history
    history_file = os.path.join(training_dir, "logs", "training_history.json")
    if not os.path.exists(history_file):
        print(f"❌ Training history not found at {history_file}")
        return False
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("❌ No training history found")
        return False
    
    # Extract data
    epochs = []
    train_losses = []
    steps = []
    
    for entry in history:
        if 'epoch' in entry:
            epochs.append(entry['epoch'])
        if 'train_loss' in entry:
            train_losses.append(entry['train_loss'])
        if 'step' in entry:
            steps.append(entry['step'])
    
    # Create loss plot
    plt.figure(figsize=(10, 6))
    
    if len(train_losses) > 1:
        plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Single point - create a bar chart
        plt.bar(['Training Loss'], train_losses, color='blue', alpha=0.7)
        plt.ylabel('Loss')
        plt.title('Training Loss (Single Epoch)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Loss curve saved to: {loss_plot_path}")
    return True

def create_metrics_plot(training_dir: str, output_dir: str = None):
    """Create metrics visualization"""
    
    if output_dir is None:
        output_dir = os.path.join(training_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation results
    eval_file = os.path.join(training_dir, "evaluation_results.json")
    if not os.path.exists(eval_file):
        print(f"❌ Evaluation results not found at {eval_file}")
        return False
    
    with open(eval_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract ROUGE metrics
    rouge_metrics = {}
    for key, value in metrics.items():
        if key.startswith('rouge_'):
            metric_name = key.replace('rouge_', '').replace('_', ' ').title()
            rouge_metrics[metric_name] = value
    
    if not rouge_metrics:
        print("❌ No ROUGE metrics found")
        return False
    
    # Create metrics bar chart
    plt.figure(figsize=(12, 8))
    
    metrics_names = list(rouge_metrics.keys())
    metrics_values = list(rouge_metrics.values())
    
    bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.xlabel('ROUGE Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    metrics_plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics plot saved to: {metrics_plot_path}")
    return True

def create_training_summary(training_dir: str, output_dir: str = None):
    """Create training summary report"""
    
    if output_dir is None:
        output_dir = os.path.join(training_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    history_file = os.path.join(training_dir, "logs", "training_history.json")
    eval_file = os.path.join(training_dir, "evaluation_results.json")
    
    summary = {
        "training_info": {},
        "final_metrics": {},
        "files_available": []
    }
    
    # Training info
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if history:
            last_entry = history[-1]
            summary["training_info"] = {
                "total_epochs": last_entry.get('epoch', 'N/A'),
                "total_steps": last_entry.get('step', 'N/A'),
                "final_train_loss": last_entry.get('train_loss', 'N/A'),
                "training_time": last_entry.get('train_runtime', 'N/A'),
                "samples_per_second": last_entry.get('train_samples_per_second', 'N/A')
            }
    
    # Final metrics
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            metrics = json.load(f)
        summary["final_metrics"] = metrics
    
    # Available files
    for root, dirs, files in os.walk(training_dir):
        for file in files:
            if file.endswith(('.json', '.csv', '.png', '.log')):
                rel_path = os.path.relpath(os.path.join(root, file), training_dir)
                summary["files_available"].append(rel_path)
    
    # Save summary
    summary_file = os.path.join(output_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ Training summary saved to: {summary_file}")
    return True

def main():
    """Main function to create all visualizations"""
    
    if len(sys.argv) < 2:
        print("Usage: python create_visualizations.py <training_directory>")
        print("Example: python create_visualizations.py artifacts/test_training")
        return 1
    
    training_dir = sys.argv[1]
    
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return 1
    
    print(f"📊 Creating visualizations for: {training_dir}")
    
    # Create visualizations
    success_count = 0
    
    if create_loss_plot(training_dir):
        success_count += 1
    
    if create_metrics_plot(training_dir):
        success_count += 1
    
    if create_training_summary(training_dir):
        success_count += 1
    
    print(f"\n✅ Created {success_count}/3 visualizations")
    print(f"📁 Visualizations saved to: {os.path.join(training_dir, 'visualizations')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
