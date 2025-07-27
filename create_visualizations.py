#!/usr/bin/env python3
"""
Visualization Generator for CrashTransformer
Creates all visualizations from existing saved data without running inference
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/visualizations_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualization Generator")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory containing saved results")
    parser.add_argument("--visualization-type", choices=["all", "basic", "enhanced", "cv"], 
                       default="all", help="Type of visualizations to create")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of all visualizations")
    return parser.parse_args()

def load_existing_data(results_dir):
    """Load existing data from saved files."""
    data = {}
    
    # Load generated summaries
    generated_file = f"{results_dir}/generated_summaries.csv"
    if os.path.exists(generated_file):
        df = pd.read_csv(generated_file)
        data['predictions'] = df['generated'].tolist()
        data['references'] = df['summary'].tolist()
        data['narratives'] = df['Narrative'].tolist()
        logging.info(f"✅ Loaded {len(data['predictions'])} generated summaries")
    else:
        logging.warning(f"⚠️  Generated summaries not found at {generated_file}")
        return None
    
    # Load causal summaries
    causal_file = f"{results_dir}/causal_summaries.csv"
    if os.path.exists(causal_file):
        causal_df = pd.read_csv(causal_file)
        data['causal_summaries'] = causal_df
        logging.info(f"✅ Loaded {len(causal_df)} causal summaries")
    
    # Load metrics
    metrics_file = f"{results_dir}/metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
        logging.info("✅ Loaded existing metrics")
    
    # Load raw evaluation data
    raw_data_file = f"{results_dir}/raw_data/evaluation_data.json"
    if os.path.exists(raw_data_file):
        with open(raw_data_file, 'r') as f:
            data['raw_evaluation'] = json.load(f)
        logging.info("✅ Loaded raw evaluation data")
    
    return data

def create_basic_visualizations(data, results_dir):
    """Create basic ROUGE score visualizations."""
    logging.info("📊 Creating basic visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create plots directory
        plots_dir = f"{results_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # ROUGE Scores Plot
        if 'metrics' in data and 'rouge' in data['metrics']:
            rouge_data = data['metrics']['rouge']
            
            plt.figure(figsize=(10, 6))
            metrics = ['rouge1', 'rouge2', 'rougeL']
            scores = []
            labels = []
            
            for metric in metrics:
                if metric in rouge_data:
                    if isinstance(rouge_data[metric], dict) and 'fmeasure' in rouge_data[metric]:
                        scores.append(rouge_data[metric]['fmeasure'])
                        labels.append(metric.upper())
                    elif isinstance(rouge_data[metric], (int, float)):
                        scores.append(rouge_data[metric])
                        labels.append(metric.upper())
            
            if scores:
                plt.bar(labels, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                plt.title("ROUGE Scores", fontsize=16, fontweight="bold")
                plt.ylabel("F1 Score", fontsize=14)
                plt.xlabel("Metric", fontsize=14)
                plt.ylim(0, 1)
                plt.grid(axis="y", alpha=0.3)
                
                # Add value labels
                for i, score in enumerate(scores):
                    plt.text(i, score + 0.01, f'{score:.3f}', 
                           ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"{plots_dir}/rouge_scores_basic.png", dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("✅ Created ROUGE scores plot")
        
        # Training Time Plot
        if 'metrics' in data and 'training_time' in data['metrics']:
            training_time = data['metrics']['training_time']
            
            plt.figure(figsize=(8, 6))
            time_units = ['seconds', 'minutes', 'hours']
            time_values = [training_time.get(unit, 0) for unit in time_units]
            
            plt.bar(time_units, time_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            plt.title("Training Time", fontsize=16, fontweight="bold")
            plt.ylabel("Time", fontsize=14)
            plt.xlabel("Unit", fontsize=14)
            
            # Add value labels
            for i, value in enumerate(time_values):
                plt.text(i, value + max(time_values) * 0.01, f'{value:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/training_time.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("✅ Created training time plot")
        
        logging.info("✅ Basic visualizations completed")
        
    except ImportError as e:
        logging.error(f"❌ Matplotlib/Seaborn not available: {e}")
    except Exception as e:
        logging.error(f"❌ Basic visualization failed: {e}")

def create_enhanced_visualizations(data, results_dir):
    """Create enhanced visualizations using the comprehensive evaluator."""
    logging.info("📈 Creating enhanced visualizations...")
    
    try:
        from enhanced_evaluation import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator(results_dir)
        
        # Use existing data if available
        if 'predictions' in data and 'references' in data:
            predictions = data['predictions']
            references = data['references']
            narratives = data.get('narratives', [])
            
            # Create all visualizations
            evaluator.create_comprehensive_visualizations(predictions, references, narratives)
            evaluator.save_raw_data()
            
            logging.info("✅ Enhanced visualizations completed")
        else:
            logging.warning("⚠️  No prediction data available for enhanced visualizations")
            
    except ImportError as e:
        logging.error(f"❌ Enhanced evaluation module not available: {e}")
        logging.info("💡 Install required packages: pip install nltk sentence-transformers wordcloud")
    except Exception as e:
        logging.error(f"❌ Enhanced visualization failed: {e}")

def create_cross_validation_visualizations(data, results_dir):
    """Create cross-validation visualizations."""
    logging.info("🔄 Creating cross-validation visualizations...")
    
    try:
        from cross_validation_module import CrossValidator
        
        cv_dir = f"{results_dir}/cross_validation"
        cv_results_file = f"{cv_dir}/cv_results.json"
        
        if os.path.exists(cv_results_file):
            with open(cv_results_file, 'r') as f:
                cv_results = json.load(f)
            
            cv = CrossValidator(results_dir=results_dir)
            cv._create_cv_visualizations(cv_results)
            
            logging.info("✅ Cross-validation visualizations completed")
        else:
            logging.warning("⚠️  Cross-validation results not found")
            
    except ImportError as e:
        logging.error(f"❌ Cross-validation module not available: {e}")
    except Exception as e:
        logging.error(f"❌ Cross-validation visualization failed: {e}")

def create_summary_report(data, results_dir):
    """Create a summary report of all available data."""
    logging.info("📋 Creating summary report...")
    
    report_file = f"{results_dir}/visualization_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("CRASHTRANSFORMER VISUALIZATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data summary
        f.write("DATA SUMMARY:\n")
        f.write("-" * 15 + "\n")
        
        if 'predictions' in data:
            f.write(f"Predictions: {len(data['predictions'])} samples\n")
        if 'references' in data:
            f.write(f"References: {len(data['references'])} samples\n")
        if 'narratives' in data:
            f.write(f"Narratives: {len(data['narratives'])} samples\n")
        if 'causal_summaries' in data:
            f.write(f"Causal Summaries: {len(data['causal_summaries'])} samples\n")
        
        f.write("\n")
        
        # Metrics summary
        if 'metrics' in data:
            f.write("METRICS SUMMARY:\n")
            f.write("-" * 15 + "\n")
            
            for metric_type, metrics in data['metrics'].items():
                f.write(f"\n{metric_type.upper()}:\n")
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict):
                            f.write(f"  {metric_name}:\n")
                            for sub_name, sub_value in value.items():
                                f.write(f"    {sub_name}: {sub_value}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                else:
                    f.write(f"  {metrics}\n")
        
        f.write("\n")
        
        # Available visualizations
        plots_dir = f"{results_dir}/plots"
        if os.path.exists(plots_dir):
            f.write("AVAILABLE VISUALIZATIONS:\n")
            f.write("-" * 25 + "\n")
            
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            for plot_file in sorted(plot_files):
                f.write(f"  - {plot_file}\n")
        
        f.write("\n")
        
        # Cross-validation summary
        cv_dir = f"{results_dir}/cross_validation"
        if os.path.exists(cv_dir):
            f.write("CROSS-VALIDATION RESULTS:\n")
            f.write("-" * 25 + "\n")
            
            cv_files = [f for f in os.listdir(cv_dir) if f.endswith(('.json', '.txt', '.png'))]
            for cv_file in sorted(cv_files):
                f.write(f"  - {cv_file}\n")
    
    logging.info("✅ Summary report created")

def main():
    """Main function."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🎨 Starting Visualization Generation")
    logger.info(f"📁 Results directory: {args.results_dir}")
    logger.info(f"🎯 Visualization type: {args.visualization_type}")
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        logger.error(f"❌ Results directory not found: {args.results_dir}")
        return False
    
    # Load existing data
    data = load_existing_data(args.results_dir)
    if data is None:
        logger.error("❌ No data found to create visualizations")
        return False
    
    # Create plots directory
    plots_dir = f"{args.results_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create visualizations based on type
    if args.visualization_type in ["all", "basic"]:
        create_basic_visualizations(data, args.results_dir)
    
    if args.visualization_type in ["all", "enhanced"]:
        create_enhanced_visualizations(data, args.results_dir)
    
    if args.visualization_type in ["all", "cv"]:
        create_cross_validation_visualizations(data, args.results_dir)
    
    # Create summary report
    create_summary_report(data, args.results_dir)
    
    logger.info("🎉 Visualization generation completed successfully!")
    logger.info(f"📁 Visualizations saved to: {plots_dir}")
    logger.info(f"📋 Summary report: {args.results_dir}/visualization_summary.txt")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 