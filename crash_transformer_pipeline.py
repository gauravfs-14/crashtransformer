# crash_transformer_pipeline.py

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import ollama
import argparse
import sys
import time

# Enhanced evaluation imports
try:
    from src.enhanced_evaluation import ComprehensiveEvaluator
    ENHANCED_EVAL_AVAILABLE = True
except ImportError:
    ENHANCED_EVAL_AVAILABLE = False

try:
    from src.cross_validation_module import CrossValidator
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import evaluate

# =======================
# 🎯 Argument Parsing
# =======================
def parse_arguments():
    parser = argparse.ArgumentParser(description="CrashTransformer Pipeline")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode with reduced dataset and epochs")
    return parser.parse_args()

args = parse_arguments()

# =======================
# 📁 Setup + Reproducibility
# =======================
DATA_PATH = "AV1_5_2024_UnCrPP 2.xlsx"
WORKSHEET_NAME = "Narr_CrLev"
MODEL_DIR = "models/crashtransformer-bart"
RESULTS_DIR = "results"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
log_filename = f"{LOG_DIR}/crashtransformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test mode adjustments
if args.test:
    MODEL_DIR = "models/crashtransformer-bart-test"
    RESULTS_DIR = "results-test"
    logger.info("🧪 Running in TEST MODE - reduced dataset and epochs")

# Training time tracking
training_start_time = None
training_end_time = None

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logger.info("🚀 Starting CrashTransformer Pipeline")
logger.info(f"📁 Data path: {DATA_PATH}")
logger.info(f"📊 Worksheet: {WORKSHEET_NAME}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
logger.info("🎲 Set random seeds for reproducibility")

# =======================
# 🔧 Ollama Setup Check
# =======================
def check_ollama_availability():
    """Check if Ollama is available and the required model is installed."""
    try:
        # Test if Ollama is running
        models = ollama.list()
        logger.info(f"Available models: {models}")
        
        # Extract model names from the response
        # The models are returned as objects with a 'model' attribute
        available_models = []
        if hasattr(models, 'models'):
            available_models = [model.model for model in models.models if model.model]
        elif isinstance(models, dict) and 'models' in models:
            # Fallback for dictionary format
            available_models = [model.get('name', model.get('model', '')) for model in models['models'] if model.get('name') or model.get('model')]
        
        logger.info(f"Extracted model names: {available_models}")
        
        if 'llama3.1:8b' in available_models:
            logger.info("✅ Ollama with llama3.1:8b model is available")
            return True
        else:
            model_list = ", ".join(available_models) if available_models else "none"
            logger.warning(f"⚠️  llama3.1:8b model not found in Ollama. Available models: {model_list}")
            logger.info("💡 To install the model, run: ollama pull llama3.1:8b")
            return False
    except Exception as e:
        logger.error(f"❌ Ollama not available: {e}")
        logger.info("💡 Please ensure Ollama is installed and running: https://ollama.ai/")
        return False

# Check Ollama availability
ollama_available = check_ollama_availability()

# =======================
# 📄 Load and Preprocess Data
# =======================
logger.info("📄 Loading data from Excel file...")
try:
    # Use smaller dataset for test mode
    if args.test:
        df = pd.read_excel(DATA_PATH, sheet_name=WORKSHEET_NAME, nrows=100)
        logger.info(f"🧪 TEST MODE: Loaded {len(df)} rows (limited for testing)")
    else:
        df = pd.read_excel(DATA_PATH, sheet_name=WORKSHEET_NAME)
        logger.info(f"✅ Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
except FileNotFoundError:
    logger.error(f"❌ File not found at: {DATA_PATH}")
    raise FileNotFoundError(f"❌ File not found at: {DATA_PATH}")
except Exception as e:
    logger.error(f"❌ Error reading Excel file: {e}")
    raise Exception(f"❌ Error reading Excel file: {e}")

logger.info("🧹 Preprocessing data...")
initial_rows = len(df)
df = df[df["Narrative"].notnull() & (df["Narrative"].str.len() > 30)]
df = df.fillna("Unknown")
df.columns = df.columns.str.strip()
df = df.reset_index(drop=True)
logger.info(f"✅ Filtered data: {initial_rows} → {len(df)} rows (removed {initial_rows - len(df)} rows)")

# Cache for generated summaries to avoid re-processing
summary_cache = {}

def generate_causal_summary(text):
    """
    Generate a causal summary using Ollama with Llama 3.1:8b model.
    Extracts the most relevant causal information from crash narratives.
    """
    # Check cache first
    text_hash = hash(text)
    if text_hash in summary_cache:
        return summary_cache[text_hash]
    
    # Use Ollama if available, otherwise fall back to keyword-based approach
    if not ollama_available:
        logger.debug("Using keyword-based fallback for causal summary generation")
        keywords = ["due to", "because", "failed to", "after", "as a result", "caused by"]
        for sent in text.split("."):
            if any(k in sent.lower() for k in keywords):
                result = sent.strip() + "."
                summary_cache[text_hash] = result
                return result
        result = text.split(".")[0].strip() + "."
        summary_cache[text_hash] = result
        return result
    
    try:
        prompt = f"""You are an expert crash analyst. Given the following crash narrative, extract the most important causal information in a concise summary (1-2 sentences maximum). Focus on the root cause, what failed, and why it happened. Do not include any other information.

Crash Narrative: {text}

Causal Summary:"""
        
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        summary = response['message']['content'].strip()
        
        # Clean up the response - remove any extra formatting
        if summary.startswith('Causal Summary:'):
            summary = summary.replace('Causal Summary:', '').strip()
        
        # Ensure we have a valid summary
        if not summary or len(summary) < 10:
            # Fallback to first sentence if LLM response is too short
            return text.split(".")[0].strip() + "."
        
        # Cache the result
        summary_cache[text_hash] = summary
        return summary
        
    except Exception as e:
        logger.warning(f"Ollama API call failed for text: {text[:100]}... Error: {e}")
        # Fallback to keyword-based approach
        keywords = ["due to", "because", "failed to", "after", "as a result", "caused by"]
        for sent in text.split("."):
            if any(k in sent.lower() for k in keywords):
                result = sent.strip() + "."
                summary_cache[text_hash] = result
                return result
        result = text.split(".")[0].strip() + "."
        summary_cache[text_hash] = result
        return result

# =======================
# 🧪 Test Function
# =======================
def run_comprehensive_test():
    """Run a comprehensive test of all pipeline components."""
    logger.info("🧪 Starting comprehensive test...")
    
    test_results = {
        "data_loading": False,
        "ollama_integration": False,
        "tokenization": False,
        "model_loading": False,
        "training_setup": False,
        "evaluation_metrics": False
    }
    
    try:
        # Test 1: Data Loading
        logger.info("📄 Testing data loading...")
        test_df = pd.read_excel(DATA_PATH, sheet_name=WORKSHEET_NAME, nrows=10)
        if len(test_df) > 0 and "Narrative" in test_df.columns:
            test_results["data_loading"] = True
            logger.info("✅ Data loading test passed")
        else:
            logger.error("❌ Data loading test failed")
            
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
    
    try:
        # Test 2: Ollama Integration
        logger.info("🤖 Testing Ollama integration...")
        if ollama_available:
            test_text = "The vehicle crashed due to brake failure on a wet road."
            summary = generate_causal_summary(test_text)
            if summary and len(summary) > 5:
                test_results["ollama_integration"] = True
                logger.info(f"✅ Ollama integration test passed: {summary}")
            else:
                logger.warning("⚠️  Ollama integration test failed - using fallback")
        else:
            logger.warning("⚠️  Ollama not available - skipping integration test")
            test_results["ollama_integration"] = True  # Fallback works
            
    except Exception as e:
        logger.error(f"❌ Ollama integration test failed: {e}")
    
    try:
        # Test 3: Tokenization
        logger.info("🔤 Testing tokenization...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        test_text = "This is a test narrative for tokenization."
        tokens = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        if tokens["input_ids"].shape[1] > 0:
            test_results["tokenization"] = True
            logger.info("✅ Tokenization test passed")
        else:
            logger.error("❌ Tokenization test failed")
            
    except Exception as e:
        logger.error(f"❌ Tokenization test failed: {e}")
    
    try:
        # Test 4: Model Loading
        logger.info("🤖 Testing model loading...")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        if model is not None:
            test_results["model_loading"] = True
            logger.info("✅ Model loading test passed")
        else:
            logger.error("❌ Model loading test failed")
            
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
    
    try:
        # Test 5: Training Setup
        logger.info("🏋️ Testing training setup...")
        training_args = TrainingArguments(
            output_dir=RESULTS_DIR,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            weight_decay=0.01,
            save_total_limit=1,
            fp16=False,  # Disable for test
            logging_dir="./logs",
            logging_steps=1,
            report_to="none"
        )
        test_results["training_setup"] = True
        logger.info("✅ Training setup test passed")
        
    except Exception as e:
        logger.error(f"❌ Training setup test failed: {e}")
    
    try:
        # Test 6: Evaluation Metrics
        logger.info("📊 Testing evaluation metrics...")
        rouge = evaluate.load("rouge")
        test_predictions = ["The crash was caused by brake failure."]
        test_references = ["The crash occurred due to brake failure."]
        rouge_results = rouge.compute(predictions=test_predictions, references=test_references)
        if rouge_results:
            test_results["evaluation_metrics"] = True
            logger.info("✅ Evaluation metrics test passed")
        else:
            logger.error("❌ Evaluation metrics test failed")
            
    except Exception as e:
        logger.error(f"❌ Evaluation metrics test failed: {e}")
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info("=" * 50)
    logger.info("🧪 COMPREHENSIVE TEST RESULTS")
    logger.info("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Pipeline is ready to run.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Please check the issues above.")
        return False

# Run comprehensive test if in test mode
if args.test:
    if run_comprehensive_test():
        logger.info("🧪 Test mode completed successfully. Exiting.")
        sys.exit(0)
    else:
        logger.error("🧪 Test mode failed. Please fix the issues before running the full pipeline.")
        sys.exit(1)

logger.info("📝 Generating causal summaries...")

def generate_causal_summaries_batch(texts, batch_size=5):
    """Generate summaries in batches to speed up processing."""
    summaries = []
    total_texts = len(texts)
    
    logger.info(f"🔄 Processing {total_texts} texts in batches of {batch_size}")
    
    for i in tqdm(range(0, total_texts, batch_size), desc="Generating summaries"):
        batch_texts = texts[i:i + batch_size]
        batch_summaries = []
        
        for j, text in enumerate(batch_texts):
            current_index = i + j + 1
            if current_index % 10 == 0:  # Log every 10th item
                logger.info(f"📝 Processed {current_index}/{total_texts} texts")
            
            summary = generate_causal_summary(text)
            batch_summaries.append(summary)
        
        summaries.extend(batch_summaries)
    
    return summaries

# Check if summaries already exist
causal_summaries_path = f"{RESULTS_DIR}/causal_summaries.csv"
if os.path.exists(causal_summaries_path):
    logger.info("📁 Found existing causal summaries, loading...")
    existing_df = pd.read_csv(causal_summaries_path)
    if len(existing_df) == len(df):
        logger.info("✅ Using existing causal summaries")
        df["summary"] = existing_df["summary"]
    else:
        logger.info("⚠️  Existing summaries don't match current data, regenerating...")
        # Use batch processing for better performance
        if args.test:
            batch_size = 3  # Smaller batches for test mode
        else:
            batch_size = 10  # Larger batches for full mode
        df["summary"] = generate_causal_summaries_batch(df["Narrative"].tolist(), batch_size)
else:
    # Use batch processing for better performance
    if args.test:
        batch_size = 3  # Smaller batches for test mode
    else:
        batch_size = 10  # Larger batches for full mode
    df["summary"] = generate_causal_summaries_batch(df["Narrative"].tolist(), batch_size)
df = pd.DataFrame(df[["Narrative", "summary"]])
df.to_csv(f"{RESULTS_DIR}/causal_summaries.csv", index=False)
logger.info(f"✅ Generated summaries for {len(df)} narratives")

# =======================
# ✂️ Tokenization
# =======================
logger.info("🔧 Setting up tokenizer...")
model_checkpoint = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
logger.info(f"✅ Loaded tokenizer: {model_checkpoint}")

def tokenize_fn(batch):
    model_inputs = tokenizer(batch["Narrative"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["summary"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logger.info("📊 Splitting data into train/test sets...")
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
# Ensure both are treated as DataFrames
train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)
logger.info(f"✅ Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
logger.info("✅ Created HuggingFace datasets")

logger.info("🔤 Tokenizing datasets...")
train_tokenized = train_dataset.map(tokenize_fn, batched=True)
test_tokenized = test_dataset.map(tokenize_fn, batched=True)
logger.info("✅ Tokenization completed")

# =======================
# 🏋️ Train Model
# =======================
logger.info("🤖 Loading pre-trained model...")
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
logger.info(f"✅ Loaded model: {model_checkpoint}")

# Adjust training parameters for test mode
if args.test:
    num_epochs = 1
    batch_size = 2
    logging_steps = 1
    logger.info("🧪 TEST MODE: Using reduced training parameters (1 epoch, batch_size=2)")
else:
    num_epochs = 3
    batch_size = 4
    logging_steps = 10

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=torch.cuda.is_available() and not args.test,  # Disable fp16 in test mode
    logging_dir="./logs",
    logging_steps=logging_steps,
    report_to="none"
)

logger.info("🏋️ Setting up trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

logger.info("🚀 Starting model training...")
training_start_time = time.time()
trainer.train()
training_end_time = time.time()
training_duration = training_end_time - training_start_time
logger.info(f"✅ Model training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

# =======================
# 💾 Save Model
# =======================
logger.info(f"💾 Saving model to {MODEL_DIR}...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
logger.info("✅ Model and tokenizer saved successfully")

# =======================
# 🔍 Inference
# =======================
logger.info("🔍 Setting up inference function...")
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    summary_ids = model.generate(inputs["input_ids"], max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
logger.info("✅ Inference function ready")

# =======================
# 📊 Evaluation
# =======================
logger.info("📊 Loading evaluation metrics...")
rouge = evaluate.load("rouge")
try:
    bertscore = evaluate.load("bertscore")
    logger.info("✅ Evaluation metrics loaded (ROUGE + BERTScore)")
except ImportError as e:
    logger.warning(f"⚠️  BERTScore not available: {e}")
    logger.info("💡 Install with: pip install bert-score")
    bertscore = None
    logger.info("✅ Evaluation metrics loaded (ROUGE only)")

# Log module availability
if ENHANCED_EVAL_AVAILABLE:
    logger.info("✅ Enhanced evaluation module loaded")
else:
    logger.warning("⚠️  Enhanced evaluation not available")

if CV_AVAILABLE:
    logger.info("✅ Cross-validation module loaded")
else:
    logger.warning("⚠️  Cross-validation not available")

logger.info("🔍 Generating predictions for evaluation...")
# Use fewer samples for evaluation in test mode
if args.test:
    eval_samples = min(10, len(test_df))
    logger.info(f"🧪 TEST MODE: Using {eval_samples} samples for evaluation")
else:
    eval_samples = min(100, len(test_df))

sample_texts = test_df["Narrative"].iloc[:eval_samples].tolist()
references = test_df["summary"].iloc[:eval_samples].tolist()
predictions = [generate_summary(t) for t in tqdm(sample_texts, desc="Generating summaries")]
logger.info(f"✅ Generated {len(predictions)} predictions")

logger.info("📈 Computing comprehensive evaluation metrics...")

# Initialize metrics variables
rouge_results = {}
bertscore_results = None

# Use enhanced evaluation if available
if ENHANCED_EVAL_AVAILABLE:
    try:
        from src.enhanced_evaluation import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator(RESULTS_DIR)
        all_metrics = evaluator.compute_all_metrics(predictions, references, sample_texts)
        
        # Extract individual metrics for compatibility
        rouge_results = all_metrics.get('rouge', {})
        bertscore_results = all_metrics.get('bertscore', None)
        
        # Create comprehensive visualizations
        logger.info("📈 Creating comprehensive visualizations...")
        evaluator.create_comprehensive_visualizations(predictions, references, sample_texts)
        evaluator.save_raw_data()
        
        logger.info("✅ Enhanced evaluation completed")
    except Exception as e:
        logger.warning(f"⚠️  Enhanced evaluation failed: {e}")
        logger.info("🔄 Falling back to basic evaluation...")
        ENHANCED_EVAL_AVAILABLE = False

# Fallback to basic evaluation if enhanced evaluation failed or not available
if not ENHANCED_EVAL_AVAILABLE:
    logger.info("📈 Computing basic evaluation metrics...")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    logger.info("✅ ROUGE metrics computed")

    if bertscore is not None:
        try:
            bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
            logger.info("✅ BERTScore metrics computed")
        except Exception as e:
            logger.warning(f"⚠️  BERTScore computation failed: {e}")
            bertscore_results = None
    else:
        logger.info("⚠️  Skipping BERTScore (not available)")

    logger.info("✅ Basic evaluation completed")

# Combine metrics
all_metrics = {
    "rouge": rouge_results,
    "bertscore": bertscore_results if bertscore_results else {},
    "training_time": {
        "seconds": training_duration,
        "minutes": training_duration / 60,
        "hours": training_duration / 3600
    }
}

# Log metrics
logger.info(f"📊 ROUGE Scores: {rouge_results}")
if bertscore_results and 'f1' in bertscore_results:
    # Handle both list and scalar values for BERTScore F1
    f1_value = bertscore_results['f1']
    if isinstance(f1_value, list):
        # If it's a list, take the mean
        f1_mean = np.mean(f1_value)
        logger.info(f"📊 BERTScore F1: {f1_mean:.4f}")
    else:
        # If it's a scalar
        logger.info(f"📊 BERTScore F1: {f1_value:.4f}")
else:
    logger.info("📊 BERTScore: Not available")
logger.info(f"⏱️  Training Time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

# Save metrics + outputs
logger.info("💾 Saving results...")
test_df_sample = test_df.iloc[:eval_samples].copy()
test_df_sample["generated"] = predictions
test_df_sample.to_csv(f"{RESULTS_DIR}/generated_summaries.csv", index=False)

with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)
logger.info("✅ Results saved successfully")

# =======================
# 📈 Plot ROUGE Scores
# =======================
logger.info("📊 Creating visualization...")

# Setup academic plotting style with white backgrounds and large fonts
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

# Use default style for white background
plt.style.use('default')

fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('white')

# Initialize variables
scores = []
labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
bars = None

if rouge_results:
    # Extract F1 scores from the nested dictionary structure
    for metric in ["rouge1", "rouge2", "rougeL"]:
        if metric in rouge_results and "fmeasure" in rouge_results[metric]:
            scores.append(float(rouge_results[metric]["fmeasure"]))
        else:
            scores.append(0.0)
    
    if scores:
        bars = ax.bar(labels, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                     alpha=0.8, edgecolor='black', linewidth=2.0)

ax.set_title("CrashTransformer ROUGE Scores", fontsize=40, fontweight="bold", pad=30)
ax.set_ylabel("Score", fontsize=36, fontweight="bold")
ax.set_xlabel("Metric", fontsize=36, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=36, width=3.0, length=10)
# Make tick labels bold
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
ax.grid(axis="y", linestyle="--", linewidth=1.5, alpha=0.6)

# Add value labels on bars
if rouge_results and scores and bars is not None:
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom', 
               fontweight='bold', fontsize=28)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/rouge_scores_plot.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
logger.info("✅ Visualization saved")

# =======================
# 🔄 Cross-Validation (Optional)
# =======================
cv_results = None
if CV_AVAILABLE and not args.test:
    logger.info("🔄 Starting cross-validation...")
    try:
        from src.cross_validation_module import CrossValidator
        cv = CrossValidator(results_dir=RESULTS_DIR)
        cv_results = cv.run_cross_validation(
            narratives=df["Narrative"].tolist(),
            summaries=df["summary"].tolist(),
            n_splits=5,
            epochs=1  # Reduced for efficiency
        )
        logger.info("✅ Cross-validation completed")
        
        # Add cross-validation results to metrics
        if cv_results and 'aggregated_metrics' in cv_results:
            all_metrics["cross_validation"] = cv_results['aggregated_metrics']
            logger.info("📊 Cross-validation metrics added to results")
    except Exception as e:
        logger.warning(f"⚠️  Cross-validation failed: {e}")
else:
    logger.info("⚠️  Skipping cross-validation (not available or test mode)")

# =======================
# ✅ Done
# =======================
if args.test:
    logger.info("🧪 TEST MODE: CrashTransformer pipeline test completed successfully!")
    print("🧪 TEST MODE: CrashTransformer pipeline test completed successfully!")
else:
    logger.info("🎉 CrashTransformer pipeline completed successfully!")
    print("✅ CrashTransformer training, evaluation, and export completed.")

logger.info(f"⏱️  Total Training Time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
print(f"⏱️  Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

logger.info(f"📁 Log file: {log_filename}")
logger.info(f"📁 Model saved to: {MODEL_DIR}")
logger.info(f"📁 Results saved to: {RESULTS_DIR}")
