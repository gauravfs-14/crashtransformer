#!/usr/bin/env python3
"""
Quick Visualization Generator
Multiple ways to create visualizations without running inference
"""

import os
import sys
import json
import pandas as pd

def method1_standalone_script():
    """Method 1: Use the standalone visualization script."""
    print("🎯 Method 1: Standalone Visualization Script")
    print("Run: python create_visualizations.py")
    print("This creates all visualizations from existing saved data")
    print()

def method2_comprehensive_evaluation():
    """Method 2: Use the comprehensive evaluation runner."""
    print("🎯 Method 2: Comprehensive Evaluation Runner")
    print("Run: python run_comprehensive_evaluation.py")
    print("This loads existing results and creates visualizations")
    print()

def method3_direct_function_call():
    """Method 3: Direct function call from enhanced evaluation."""
    print("🎯 Method 3: Direct Function Call")
    print("Run: python -c \"from enhanced_evaluation import create_visualizations_from_saved_data; create_visualizations_from_saved_data()\"")
    print("This directly calls the visualization function")
    print()

def method4_inference_script():
    """Method 4: Use the inference script."""
    print("🎯 Method 4: Inference Script")
    print("Run: python inference_script.py --evaluate")
    print("This loads the model and creates basic visualizations")
    print()

def check_available_data():
    """Check what data is available for visualization."""
    print("📊 Checking Available Data:")
    print("-" * 30)
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ Results directory not found")
        return False
    
    # Check for generated summaries
    generated_file = f"{results_dir}/generated_summaries.csv"
    if os.path.exists(generated_file):
        df = pd.read_csv(generated_file)
        print(f"✅ Generated summaries: {len(df)} samples")
    else:
        print("❌ Generated summaries not found")
    
    # Check for causal summaries
    causal_file = f"{results_dir}/causal_summaries.csv"
    if os.path.exists(causal_file):
        df = pd.read_csv(causal_file)
        print(f"✅ Causal summaries: {len(df)} samples")
    else:
        print("❌ Causal summaries not found")
    
    # Check for metrics
    metrics_file = f"{results_dir}/metrics.json"
    if os.path.exists(metrics_file):
        print("✅ Metrics file found")
    else:
        print("❌ Metrics file not found")
    
    # Check for raw data
    raw_data_file = f"{results_dir}/raw_data/evaluation_data.json"
    if os.path.exists(raw_data_file):
        print("✅ Raw evaluation data found")
    else:
        print("❌ Raw evaluation data not found")
    
    # Check for cross-validation results
    cv_dir = f"{results_dir}/cross_validation"
    if os.path.exists(cv_dir):
        cv_files = [f for f in os.listdir(cv_dir) if f.endswith('.json')]
        if cv_files:
            print(f"✅ Cross-validation results: {len(cv_files)} files")
        else:
            print("❌ No cross-validation results found")
    else:
        print("❌ Cross-validation directory not found")
    
    print()

def show_visualization_types():
    """Show what types of visualizations can be created."""
    print("📈 Available Visualization Types:")
    print("-" * 35)
    
    print("🎨 Basic Visualizations:")
    print("  - ROUGE Scores (bar plot)")
    print("  - Training Time")
    print("  - Basic metrics comparison")
    print()
    
    print("📊 Enhanced Visualizations:")
    print("  - ROUGE Score Comparisons")
    print("  - BERTScore Distributions")
    print("  - Length Analysis (histograms, scatter plots)")
    print("  - Semantic Similarity Distribution")
    print("  - BLEU Score Distribution")
    print("  - Summary Quality Analysis")
    print("  - Word Cloud Analysis")
    print("  - Correlation Matrix")
    print("  - Performance by Length")
    print("  - Error Analysis")
    print()
    
    print("🔄 Cross-Validation Visualizations:")
    print("  - Fold Performance Comparison")
    print("  - Metric Distribution Across Folds")
    print("  - Learning Curves")
    print("  - Prediction Quality by Fold")
    print()

def show_usage_examples():
    """Show usage examples."""
    print("🚀 Usage Examples:")
    print("-" * 20)
    
    print("1. Create all visualizations:")
    print("   python create_visualizations.py")
    print()
    
    print("2. Create only basic visualizations:")
    print("   python create_visualizations.py --visualization-type basic")
    print()
    
    print("3. Create only enhanced visualizations:")
    print("   python create_visualizations.py --visualization-type enhanced")
    print()
    
    print("4. Create only cross-validation visualizations:")
    print("   python create_visualizations.py --visualization-type cv")
    print()
    
    print("5. Use comprehensive evaluation (loads existing results):")
    print("   python run_comprehensive_evaluation.py")
    print()
    
    print("6. Use inference script (loads model and creates basic plots):")
    print("   python inference_script.py --evaluate")
    print()

def main():
    """Main function."""
    print("🎨 CrashTransformer Visualization Generator")
    print("=" * 50)
    print()
    
    print("✅ All visualizations can be created WITHOUT running full inference!")
    print("The system stores raw data and provides multiple ways to generate plots.")
    print()
    
    # Check available data
    check_available_data()
    
    # Show visualization types
    show_visualization_types()
    
    # Show methods
    print("🛠️ Methods to Create Visualizations:")
    print("-" * 35)
    method1_standalone_script()
    method2_comprehensive_evaluation()
    method3_direct_function_call()
    method4_inference_script()
    
    # Show usage examples
    show_usage_examples()
    
    print("💡 Tips:")
    print("- All methods work with existing saved data")
    print("- No need to rerun training or inference")
    print("- Raw data is preserved for reproducibility")
    print("- Visualizations are saved in results/plots/")
    print("- Summary reports are generated automatically")
    print()
    
    print("🎉 Ready to create visualizations!")

if __name__ == "__main__":
    main() 