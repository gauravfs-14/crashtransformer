#!/usr/bin/env python3
"""
CrashTransformer Inference Script
Loads saved model and results without rerunning the full pipeline.
"""

import os
import json
import pandas as pd
import torch
import logging
from datetime import datetime
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrashTransformerInference:
    def __init__(self, model_dir="models/crashtransformer-bart", results_dir="results"):
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the trained model and tokenizer."""
        try:
            logger.info(f"🤖 Loading model from {self.model_dir}...")
            self.model = BartForConditionalGeneration.from_pretrained(self.model_dir)
            self.tokenizer = BartTokenizer.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def load_saved_results(self):
        """Load saved causal summaries and metrics."""
        results = {}
        
        # Load causal summaries
        causal_summaries_path = f"{self.results_dir}/causal_summaries.csv"
        if os.path.exists(causal_summaries_path):
            logger.info(f"📁 Loading causal summaries from {causal_summaries_path}")
            results['causal_summaries'] = pd.read_csv(causal_summaries_path)
            logger.info(f"✅ Loaded {len(results['causal_summaries'])} causal summaries")
        else:
            logger.warning("⚠️  Causal summaries not found")
            results['causal_summaries'] = None
        
        # Load metrics
        metrics_path = f"{self.results_dir}/metrics.json"
        if os.path.exists(metrics_path):
            logger.info(f"📊 Loading metrics from {metrics_path}")
            with open(metrics_path, 'r') as f:
                results['metrics'] = json.load(f)
            logger.info("✅ Metrics loaded successfully")
        else:
            logger.warning("⚠️  Metrics not found")
            results['metrics'] = None
        
        # Load generated summaries
        generated_summaries_path = f"{self.results_dir}/generated_summaries.csv"
        if os.path.exists(generated_summaries_path):
            logger.info(f"📝 Loading generated summaries from {generated_summaries_path}")
            results['generated_summaries'] = pd.read_csv(generated_summaries_path)
            logger.info(f"✅ Loaded {len(results['generated_summaries'])} generated summaries")
        else:
            logger.warning("⚠️  Generated summaries not found")
            results['generated_summaries'] = None
        
        return results
    
    def generate_summary(self, text):
        """Generate a summary for a single text."""
        if self.model is None or self.tokenizer is None:
            logger.error("❌ Model not loaded. Call load_model() first.")
            return None
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"], 
                    max_length=64, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"❌ Error generating summary: {e}")
            return None
    
    def generate_summaries_batch(self, texts, batch_size=8):
        """Generate summaries for a batch of texts."""
        if self.model is None or self.tokenizer is None:
            logger.error("❌ Model not loaded. Call load_model() first.")
            return None
        
        summaries = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating summaries"):
            batch_texts = texts[i:i + batch_size]
            batch_summaries = []
            
            for text in batch_texts:
                summary = self.generate_summary(text)
                batch_summaries.append(summary)
            
            summaries.extend(batch_summaries)
        
        return summaries
    
    def evaluate_model(self, test_data=None):
        """Evaluate the model on test data."""
        if test_data is None:
            # Use saved generated summaries if available
            results = self.load_saved_results()
            if results['generated_summaries'] is not None:
                test_data = results['generated_summaries']
            else:
                logger.error("❌ No test data available for evaluation")
                return None
        
        logger.info("📊 Evaluating model...")
        
        # Calculate basic metrics
        if 'generated' in test_data.columns and 'summary' in test_data.columns:
            total_samples = len(test_data)
            successful_generations = test_data['generated'].notna().sum()
            
            logger.info(f"📈 Evaluation Results:")
            logger.info(f"   Total samples: {total_samples}")
            logger.info(f"   Successful generations: {successful_generations}")
            logger.info(f"   Success rate: {successful_generations/total_samples*100:.2f}%")
            
            # Show some examples
            logger.info(f"📝 Sample generations:")
            for i, row in test_data.head(3).iterrows():
                logger.info(f"   Original: {row['Narrative'][:100]}...")
                logger.info(f"   Generated: {row['generated']}")
                logger.info(f"   Reference: {row['summary']}")
                logger.info("   ---")
        
        return test_data
    
    def print_metrics(self):
        """Print saved metrics."""
        results = self.load_saved_results()
        
        if results['metrics']:
            metrics = results['metrics']
            logger.info("📊 Saved Metrics:")
            
            if 'training_time' in metrics:
                training_time = metrics['training_time']
                logger.info(f"⏱️  Training Time: {training_time['minutes']:.2f} minutes")
            
            if 'rouge' in metrics:
                rouge = metrics['rouge']
                logger.info(f"📈 ROUGE Scores:")
                for key, value in rouge.items():
                    if isinstance(value, dict) and 'fmeasure' in value:
                        logger.info(f"   {key}: {value['fmeasure']:.4f}")
                    else:
                        logger.info(f"   {key}: {value:.4f}")
            
            if 'bertscore' in metrics:
                bertscore = metrics['bertscore']
                logger.info(f"🤖 BERTScore:")
                for key, value in bertscore.items():
                    logger.info(f"   {key}: {value:.4f}")
        else:
            logger.warning("⚠️  No metrics found")

def main():
    parser = argparse.ArgumentParser(description="CrashTransformer Inference")
    parser.add_argument("--model-dir", default="models/crashtransformer-bart", 
                       help="Directory containing the trained model")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory containing saved results")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Evaluate the model on saved test data")
    parser.add_argument("--metrics", action="store_true", 
                       help="Print saved metrics")
    parser.add_argument("--text", type=str, 
                       help="Generate summary for a single text")
    parser.add_argument("--file", type=str, 
                       help="Generate summaries for texts in a CSV file")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = CrashTransformerInference(args.model_dir, args.results_dir)
    
    # Load model
    if not inference.load_model():
        logger.error("❌ Failed to load model. Exiting.")
        return
    
    # Print metrics if requested
    if args.metrics:
        inference.print_metrics()
    
    # Evaluate model if requested
    if args.evaluate:
        inference.evaluate_model()
    
    # Generate summary for single text
    if args.text:
        logger.info(f"📝 Generating summary for: {args.text[:100]}...")
        summary = inference.generate_summary(args.text)
        logger.info(f"✅ Generated summary: {summary}")
    
    # Generate summaries for file
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"❌ File not found: {args.file}")
            return
        
        logger.info(f"📁 Loading texts from {args.file}")
        df = pd.read_csv(args.file)
        
        if 'Narrative' not in df.columns:
            logger.error("❌ No 'Narrative' column found in file")
            return
        
        logger.info(f"🔄 Generating summaries for {len(df)} texts...")
        summaries = inference.generate_summaries_batch(df['Narrative'].tolist())
        
        df['generated_summary'] = summaries
        output_file = f"generated_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Generated summaries saved to {output_file}")
    
    # If no specific action requested, show available data
    if not any([args.evaluate, args.metrics, args.text, args.file]):
        logger.info("ℹ️  No specific action requested. Available options:")
        logger.info("   --evaluate: Evaluate model on saved test data")
        logger.info("   --metrics: Print saved metrics")
        logger.info("   --text 'your text': Generate summary for single text")
        logger.info("   --file path/to/file.csv: Generate summaries for CSV file")

if __name__ == "__main__":
    main() 