#!/usr/bin/env python3
"""
Training script for fine-tuning crash summarization models
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.fine_tuning import fine_tune_crash_model, CrashSummarizationFineTuner, FineTuningConfig
from misc.logger import get_logger

def main():
    parser = argparse.ArgumentParser(description="Fine-tune crash summarization models")
    
    # Data arguments
    parser.add_argument("--training_data", type=str, required=True,
                       help="Path to training data (CSV or JSONL)")
    parser.add_argument("--data_format", type=str, default="csv", choices=["csv", "jsonl"],
                       help="Format of training data")
    parser.add_argument("--validation_split", type=float, default=0.1,
                       help="Fraction of data to use for validation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
                       help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_models",
                       help="Directory to save fine-tuned model")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=128,
                       help="Maximum target sequence length")
    
    # Advanced training arguments
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for training logs")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(log_dir=args.log_dir, log_level=args.log_level)
    logger.info("Starting crash summarization model fine-tuning")
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create fine-tuning configuration
        config = FineTuningConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            early_stopping_patience=args.early_stopping_patience
        )
        
        # Initialize fine-tuner
        fine_tuner = CrashSummarizationFineTuner(config, logger)
        
        # Prepare training data
        logger.info("Preparing training data...")
        training_data = fine_tuner.prepare_training_data(args.training_data, args.data_format)
        
        if len(training_data.narratives) == 0:
            logger.error("No training data found. Check your data format and file path.")
            return 1
        
        logger.info(f"Prepared {len(training_data.narratives)} training examples")
        
        # Train the model
        logger.info("Starting model training...")
        model_path = fine_tuner.train_model(training_data, validation_split=args.validation_split)
        
        logger.info(f"✅ Fine-tuning completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        
        # Evaluate the model
        logger.info("Evaluating fine-tuned model...")
        metrics = fine_tuner.evaluate_model(training_data, model_path)
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save evaluation results
        eval_results_path = os.path.join(args.output_dir, "evaluation_results.json")
        import json
        import numpy as np
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        metrics_serializable = convert_numpy(metrics)
        with open(eval_results_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {eval_results_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
