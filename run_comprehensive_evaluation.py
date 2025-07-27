#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner for CrashTransformer
Run this script to perform comprehensive evaluation on saved results
"""

import os
import sys
import argparse
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/comprehensive_evaluation_{timestamp}.log"
    
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
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Runner")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory containing saved results")
    parser.add_argument("--model-dir", default="models/crashtransformer-bart",
                       help="Directory containing trained model")
    parser.add_argument("--cross-validation", action="store_true",
                       help="Run cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--cv-epochs", type=int, default=2,
                       help="Number of epochs for cross-validation")
    parser.add_argument("--data-file", default="AV1_5_2024_UnCrPP 2.xlsx",
                       help="Path to data file")
    parser.add_argument("--worksheet", default="Narr_CrLev",
                       help="Worksheet name in Excel file")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting Comprehensive Evaluation")
    logger.info(f"📁 Results directory: {args.results_dir}")
    logger.info(f"🤖 Model directory: {args.model_dir}")
    
    # Check if enhanced evaluation is available
    try:
        from enhanced_evaluation import ComprehensiveEvaluator
        ENHANCED_EVAL_AVAILABLE = True
        logger.info("✅ Enhanced evaluation module loaded")
    except ImportError as e:
        logger.error(f"❌ Enhanced evaluation not available: {e}")
        logger.info("💡 Install required packages: pip install nltk sentence-transformers wordcloud")
        return False
    
    # Check if cross-validation is available
    try:
        from cross_validation_module import CrossValidator
        CV_AVAILABLE = True
        logger.info("✅ Cross-validation module loaded")
    except ImportError as e:
        logger.warning(f"⚠️  Cross-validation not available: {e}")
        CV_AVAILABLE = False
    
    # Load data
    logger.info("📄 Loading data...")
    try:
        import pandas as pd
        df = pd.read_excel(args.data_file, sheet_name=args.worksheet)
        logger.info(f"✅ Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return False
    
    # Check for existing results
    causal_summaries_path = f"{args.results_dir}/causal_summaries.csv"
    if not os.path.exists(causal_summaries_path):
        logger.error(f"❌ Causal summaries not found at {causal_summaries_path}")
        logger.info("💡 Run the main pipeline first to generate causal summaries")
        return False
    
    # Load causal summaries
    logger.info("📁 Loading causal summaries...")
    try:
        causal_df = pd.read_csv(causal_summaries_path)
        logger.info(f"✅ Loaded {len(causal_df)} causal summaries")
    except Exception as e:
        logger.error(f"❌ Failed to load causal summaries: {e}")
        return False
    
    # Check for existing model predictions
    generated_summaries_path = f"{args.results_dir}/generated_summaries.csv"
    if os.path.exists(generated_summaries_path):
        logger.info("📁 Found existing generated summaries, loading...")
        try:
            generated_df = pd.read_csv(generated_summaries_path)
            predictions = generated_df["generated"].tolist()
            references = generated_df["summary"].tolist()
            narratives = generated_df["Narrative"].tolist()
            logger.info(f"✅ Loaded {len(predictions)} generated summaries")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load generated summaries: {e}")
            logger.info("🔄 Will generate new predictions...")
            predictions = None
    else:
        logger.info("📁 No existing generated summaries found")
        predictions = None
    
    # Generate predictions if needed
    if predictions is None:
        logger.info("🤖 Loading model for prediction generation...")
        try:
            from transformers import BartTokenizer, BartForConditionalGeneration
            import torch
            
            tokenizer = BartTokenizer.from_pretrained(args.model_dir)
            model = BartForConditionalGeneration.from_pretrained(args.model_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            logger.info("✅ Model loaded successfully")
            
            # Generate predictions
            logger.info("🔍 Generating predictions...")
            predictions = []
            references = []
            narratives = []
            
            # Use a subset for evaluation
            eval_samples = min(100, len(causal_df))
            sample_df = causal_df.head(eval_samples)
            
            for idx, row in sample_df.iterrows():
                narrative = row["Narrative"]
                reference = row["summary"]
                
                # Tokenize
                inputs = tokenizer(narrative, return_tensors="pt", truncation=True, max_length=512).to(device)
                
                # Generate
                with torch.no_grad():
                    summary_ids = model.generate(
                        inputs["input_ids"], 
                        max_length=64, 
                        num_beams=4, 
                        early_stopping=True
                    )
                
                prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                predictions.append(prediction)
                references.append(reference)
                narratives.append(narrative)
            
            logger.info(f"✅ Generated {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate predictions: {e}")
            return False
    
    # Run comprehensive evaluation
    logger.info("📊 Running comprehensive evaluation...")
    try:
        evaluator = ComprehensiveEvaluator(args.results_dir)
        all_metrics = evaluator.compute_all_metrics(predictions, references, narratives)
        
        # Create visualizations
        logger.info("📈 Creating comprehensive visualizations...")
        evaluator.create_comprehensive_visualizations(predictions, references, narratives)
        evaluator.save_raw_data()
        
        logger.info("✅ Comprehensive evaluation completed")
        
    except Exception as e:
        logger.error(f"❌ Comprehensive evaluation failed: {e}")
        return False
    
    # Run cross-validation if requested
    if args.cross_validation and CV_AVAILABLE:
        logger.info("🔄 Running cross-validation...")
        try:
            cv = CrossValidator(results_dir=args.results_dir)
            cv_results = cv.run_cross_validation(
                narratives=causal_df["Narrative"].tolist(),
                summaries=causal_df["summary"].tolist(),
                n_splits=args.cv_folds,
                epochs=args.cv_epochs
            )
            logger.info("✅ Cross-validation completed")
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            return False
    elif args.cross_validation and not CV_AVAILABLE:
        logger.warning("⚠️  Cross-validation requested but module not available")
    
    logger.info("🎉 Comprehensive evaluation completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 