#!/usr/bin/env python3
"""
Script to prepare training data from pipeline outputs for fine-tuning
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.fine_tuning import CrashSummarizationFineTuner, FineTuningConfig
from misc.logger import get_logger

def prepare_training_data_from_pipeline_outputs(
    graphs_file: str,
    summaries_file: str,
    output_file: str,
    data_format: str = "csv",
    logger=None
):
    """Prepare training data from pipeline outputs"""
    
    if logger:
        logger.info(f"Preparing training data from pipeline outputs")
        logger.info(f"Graphs file: {graphs_file}")
        logger.info(f"Summaries file: {summaries_file}")
        logger.info(f"Output file: {output_file}")
    
    # Initialize fine-tuner for data preparation
    config = FineTuningConfig()
    fine_tuner = CrashSummarizationFineTuner(config, logger)
    
    # Create training dataset from pipeline outputs
    training_data = fine_tuner.create_training_dataset_from_pipeline_outputs(
        graphs_file, summaries_file
    )
    
    if len(training_data.narratives) == 0:
        if logger:
            logger.error("No training data found in pipeline outputs")
        return False
    
    # Save training data
    if data_format == "csv":
        df = pd.DataFrame({
            'Narrative': training_data.narratives,
            'Summary': training_data.summaries,
            'plan_lines': [str(plan) for plan in training_data.plans],
            'Crash_ID': [meta.get('crash_id', '') for meta in training_data.metadata],
            'Crash_Severity': [meta.get('severity', '') for meta in training_data.metadata],
            'City': [meta.get('city', '') for meta in training_data.metadata]
        })
        df.to_csv(output_file, index=False)
        
    elif data_format == "jsonl":
        with open(output_file, 'w') as f:
            for i in range(len(training_data.narratives)):
                record = {
                    'narrative': training_data.narratives[i],
                    'summary': training_data.summaries[i],
                    'plan_lines': training_data.plans[i],
                    'metadata': training_data.metadata[i]
                }
                f.write(json.dumps(record) + '\n')
    
    if logger:
        logger.info(f"✅ Training data prepared successfully!")
        logger.info(f"Total examples: {len(training_data.narratives)}")
        logger.info(f"Saved to: {output_file}")
    
    return True

def create_synthetic_training_data(
    output_file: str,
    num_examples: int = 1000,
    data_format: str = "csv",
    logger=None
):
    """Create synthetic training data for demonstration"""
    
    if logger:
        logger.info(f"Creating synthetic training data with {num_examples} examples")
    
    # Sample crash narratives and summaries
    sample_narratives = [
        "Unit 1 was traveling eastbound on Main Street when Unit 2 failed to yield at the intersection and collided with Unit 1.",
        "Unit 2 was stopped at a red light when Unit 1 failed to control speed and struck Unit 2 from behind.",
        "Unit 1 was making a left turn when Unit 2, traveling in the opposite direction, failed to stop at the red light and collided with Unit 1.",
        "Unit 2 was parked on the side of the road when Unit 1, distracted by a phone call, veered off the road and struck Unit 2.",
        "Unit 1 was changing lanes when Unit 2, in the adjacent lane, failed to check blind spots and sideswiped Unit 1."
    ]
    
    sample_summaries = [
        "Unit 2 failed to yield at intersection, causing collision with Unit 1.",
        "Unit 1 failed to control speed and rear-ended stationary Unit 2.",
        "Unit 2 ran red light and collided with Unit 1 making left turn.",
        "Unit 1 was distracted and veered off road, striking parked Unit 2.",
        "Unit 2 failed to check blind spots while Unit 1 was changing lanes, causing sideswipe."
    ]
    
    sample_plans = [
        ["Unit 2 failed to yield -> collision with Unit 1"],
        ["Unit 1 failed to control speed -> rear-end collision with Unit 2"],
        ["Unit 2 ran red light -> collision with Unit 1"],
        ["Unit 1 distracted -> veered off road -> struck Unit 2"],
        ["Unit 2 failed to check blind spots -> sideswipe with Unit 1"]
    ]
    
    narratives = []
    summaries = []
    plans = []
    crash_ids = []
    severities = []
    cities = []
    
    for i in range(num_examples):
        # Randomly select from samples
        import random
        idx = i % len(sample_narratives)
        
        narratives.append(sample_narratives[idx])
        summaries.append(sample_summaries[idx])
        plans.append(sample_plans[idx])
        crash_ids.append(f"SYNTH_{i+1:06d}")
        severities.append(random.choice(["Not Injured", "Injured", "Fatal"]))
        cities.append(random.choice(["Dallas", "Houston", "Austin", "San Antonio", "Fort Worth"]))
    
    # Save training data
    if data_format == "csv":
        df = pd.DataFrame({
            'Narrative': narratives,
            'Summary': summaries,
            'plan_lines': [str(plan) for plan in plans],
            'Crash_ID': crash_ids,
            'Crash_Severity': severities,
            'City': cities
        })
        df.to_csv(output_file, index=False)
        
    elif data_format == "jsonl":
        with open(output_file, 'w') as f:
            for i in range(len(narratives)):
                record = {
                    'narrative': narratives[i],
                    'summary': summaries[i],
                    'plan_lines': plans[i],
                    'metadata': {
                        'crash_id': crash_ids[i],
                        'severity': severities[i],
                        'city': cities[i]
                    }
                }
                f.write(json.dumps(record) + '\n')
    
    if logger:
        logger.info(f"✅ Synthetic training data created successfully!")
        logger.info(f"Total examples: {len(narratives)}")
        logger.info(f"Saved to: {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    
    # Data source options
    parser.add_argument("--source", type=str, choices=["pipeline", "synthetic"], default="synthetic",
                       help="Source of training data")
    parser.add_argument("--graphs_file", type=str, default=None,
                       help="Path to crash graphs JSONL file (for pipeline source)")
    parser.add_argument("--summaries_file", type=str, default=None,
                       help="Path to crash summaries JSONL file (for pipeline source)")
    parser.add_argument("--num_examples", type=int, default=1000,
                       help="Number of synthetic examples to create")
    
    # Output options
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path for training data")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "jsonl"],
                       help="Output format for training data")
    
    # Logging options
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for logs")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(log_dir=args.log_dir, log_level=args.log_level)
    
    try:
        if args.source == "pipeline":
            if not args.graphs_file or not args.summaries_file:
                logger.error("Graphs file and summaries file are required for pipeline source")
                return 1
            
            if not os.path.exists(args.graphs_file):
                logger.error(f"Graphs file not found: {args.graphs_file}")
                return 1
            
            if not os.path.exists(args.summaries_file):
                logger.error(f"Summaries file not found: {args.summaries_file}")
                return 1
            
            success = prepare_training_data_from_pipeline_outputs(
                args.graphs_file,
                args.summaries_file,
                args.output,
                args.format,
                logger
            )
            
        elif args.source == "synthetic":
            success = create_synthetic_training_data(
                args.output,
                args.num_examples,
                args.format,
                logger
            )
        
        if success:
            logger.info("✅ Training data preparation completed successfully!")
            return 0
        else:
            logger.error("❌ Training data preparation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Training data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
