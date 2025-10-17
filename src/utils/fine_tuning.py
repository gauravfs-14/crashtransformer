# fine_tuning.py

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    model_name: str = "facebook/bart-base"
    output_dir: str = "fine_tuned_models"
    max_length: int = 512
    max_target_length: int = 128
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 0  # Disable multiprocessing on macOS
    remove_unused_columns: bool = False

@dataclass
class TrainingData:
    """Container for training data"""
    narratives: List[str] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    plans: List[List[str]] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)

class CrashSummarizationFineTuner:
    """Fine-tuner for crash summarization models"""
    
    def __init__(self, config: FineTuningConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.logger.info(f"Initialized fine-tuner with model: {config.model_name}")
        self.logger.info(f"Using device: {self.device}")
    
    def prepare_training_data(self, data_source: str, data_format: str = "csv") -> TrainingData:
        """Prepare training data from various sources"""
        training_data = TrainingData()
        
        if data_format == "csv":
            df = pd.read_csv(data_source)
            
            # Extract narratives and summaries
            for _, row in df.iterrows():
                narrative = str(row.get('Narrative', ''))
                summary = str(row.get('Summary', ''))
                plan_lines = eval(row.get('plan_lines', '[]')) if 'plan_lines' in row else []
                
                if narrative and summary:
                    training_data.narratives.append(narrative)
                    training_data.summaries.append(summary)
                    training_data.plans.append(plan_lines)
                    training_data.metadata.append({
                        'crash_id': row.get('Crash_ID', ''),
                        'severity': row.get('Crash_Severity', ''),
                        'city': row.get('City', '')
                    })
        
        elif data_format == "jsonl":
            with open(data_source, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    narrative = data.get('narrative', '')
                    summary = data.get('summary', '')
                    plan_lines = data.get('plan_lines', [])
                    
                    if narrative and summary:
                        training_data.narratives.append(narrative)
                        training_data.summaries.append(summary)
                        training_data.plans.append(plan_lines)
                        training_data.metadata.append(data.get('metadata', {}))
        
        self.logger.info(f"Prepared {len(training_data.narratives)} training examples")
        return training_data
    
    def create_training_prompts(self, training_data: TrainingData) -> Tuple[List[str], List[str]]:
        """Create training prompts with causal plans"""
        inputs = []
        targets = []
        
        for narrative, summary, plan_lines in zip(
            training_data.narratives, 
            training_data.summaries, 
            training_data.plans
        ):
            # Create plan-conditioned prompt
            if plan_lines:
                plan_text = "\n".join([f"{i+1}) {line}" for i, line in enumerate(plan_lines)])
                prompt = (
                    "You are a crash analyst. Write a concise causal summary in 1 to 3 sentences. "
                    "Cover the listed edges and keep claims faithful to the narrative.\n"
                    "Plan:\n"
                    f"{plan_text}\n"
                    "Narrative:\n"
                    f"{narrative.strip()}\n"
                    "Summary:"
                )
            else:
                # Fallback prompt without plan
                prompt = (
                    "You are a crash analyst. Write a concise causal summary in 1 to 3 sentences "
                    "based on the crash narrative.\n"
                    "Narrative:\n"
                    f"{narrative.strip()}\n"
                    "Summary:"
                )
            
            inputs.append(prompt)
            targets.append(summary)
        
        return inputs, targets
    
    def create_dataset(self, inputs: List[str], targets: List[str]) -> Dataset:
        """Create HuggingFace dataset for training"""
        
        def tokenize_function(examples):
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                examples["target"],
                max_length=self.config.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Create dataset
        dataset_dict = {
            "input": inputs,
            "target": targets
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_model(self, training_data: TrainingData, validation_split: float = 0.1) -> str:
        """Train the model with the provided data"""
        
        # Disable multiprocessing on macOS to avoid _share_filename_ error
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Create training prompts
        inputs, targets = self.create_training_prompts(training_data)
        
        # Split data
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=validation_split, random_state=42
        )
        
        # Create datasets
        train_dataset = self.create_dataset(train_inputs, train_targets)
        val_dataset = self.create_dataset(val_inputs, val_targets)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            report_to=None,  # Disable TensorBoard for now
            logging_strategy="steps",
            save_total_limit=3,  # Keep more checkpoints
            prediction_loss_only=False,  # Save predictions for analysis
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        # Train the model
        self.logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save training history and predictions
        self.save_training_history(trainer, val_inputs, val_targets)
        
        # Save the final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        self.logger.info(f"Fine-tuning completed. Model saved to: {final_model_path}")
        return final_model_path
    
    def save_training_history(self, trainer, val_inputs: List[str], val_targets: List[str]):
        """Save training history, predictions, and raw data for visualization"""
        import json
        import pandas as pd
        import numpy as np
        
        # Save training logs
        logs_dir = os.path.join(self.config.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Get training history from trainer state
        if hasattr(trainer.state, 'log_history'):
            training_history = trainer.state.log_history
            
            # Save training history as JSON
            history_file = os.path.join(logs_dir, "training_history.json")
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2, default=str)
            
            # Extract loss data for plotting
            train_losses = []
            eval_losses = []
            steps = []
            
            for log in training_history:
                if 'loss' in log:
                    train_losses.append(log['loss'])
                    steps.append(log.get('step', 0))
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
            
            # Save loss data for visualization
            loss_data = {
                'steps': steps,
                'train_loss': train_losses,
                'eval_loss': eval_losses
            }
            
            loss_file = os.path.join(logs_dir, "loss_data.json")
            with open(loss_file, 'w') as f:
                json.dump(loss_data, f, indent=2)
            
            # Save as CSV for easy plotting
            if train_losses:
                loss_df = pd.DataFrame({
                    'step': steps[:len(train_losses)],
                    'train_loss': train_losses,
                    'eval_loss': eval_losses[:len(train_losses)] if eval_losses else [None] * len(train_losses)
                })
                loss_df.to_csv(os.path.join(logs_dir, "loss_curve.csv"), index=False)
        
        # Generate and save predictions on validation set
        if val_inputs and val_targets:
            predictions_file = os.path.join(logs_dir, "validation_predictions.json")
            
            # Get predictions from the trainer
            try:
                eval_results = trainer.evaluate()
                self.logger.info(f"Validation results: {eval_results}")
                
                # Save validation results
                with open(predictions_file, 'w') as f:
                    json.dump(eval_results, f, indent=2, default=str)
                    
            except Exception as e:
                self.logger.warning(f"Could not generate validation predictions: {e}")
        
        self.logger.info(f"Training history saved to: {logs_dir}")
    
    def evaluate_model(self, test_data: TrainingData, model_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the fine-tuned model"""
        
        if model_path:
            # Load the fine-tuned model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model = self.model
            tokenizer = self.tokenizer
        
        model.eval()
        
        # Prepare test data
        test_inputs, test_targets = self.create_training_prompts(test_data)
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for input_text, target in zip(test_inputs, test_targets):
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    max_length=self.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate prediction
                outputs = model.generate(
                    **inputs,
                    max_length=self.config.max_target_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode prediction
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(prediction)
                references.append(target)
        
        # Calculate metrics
        from .advanced_metrics import calculate_metrics
        metrics = calculate_metrics(predictions, references)
        
        return metrics
    
    def create_training_dataset_from_pipeline_outputs(self, 
                                                     graphs_file: str, 
                                                     summaries_file: str) -> TrainingData:
        """Create training dataset from pipeline outputs"""
        training_data = TrainingData()
        
        # Load graphs
        with open(graphs_file, 'r') as f:
            graphs = [json.loads(line) for line in f]
        
        # Load summaries
        with open(summaries_file, 'r') as f:
            summaries = [json.loads(line) for line in f]
        
        # Create mapping
        summaries_dict = {s['Crash_ID']: s for s in summaries}
        
        for graph in graphs:
            crash_id = graph.get('crash', {}).get('crash_id')
            # Convert to int for comparison with summaries_dict keys
            crash_id_int = int(crash_id) if crash_id else None
            if crash_id_int in summaries_dict:
                narrative = graph.get('crash', {}).get('raw_narrative', '')
                summary_data = summaries_dict[crash_id_int]
                # Use LLM summary from graphs file (where we added it)
                summary = graph.get('llm_summary', '')
                plan_lines = summary_data.get('plan_lines', [])
                
                if narrative and summary:
                    training_data.narratives.append(narrative)
                    training_data.summaries.append(summary)
                    training_data.plans.append(plan_lines)
                    training_data.metadata.append({
                        'crash_id': crash_id,
                        'severity': graph.get('crash', {}).get('crash_severity', ''),
                        'city': graph.get('crash', {}).get('city', '')
                    })
        
        self.logger.info(f"Created training dataset with {len(training_data.narratives)} examples")
        return training_data

def fine_tune_crash_model(
    training_data_path: str,
    data_format: str = "csv",
    model_name: str = "facebook/bart-base",
    output_dir: str = "fine_tuned_models",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    logger=None
) -> str:
    """
    Fine-tune a model for crash summarization
    
    Args:
        training_data_path: Path to training data (CSV or JSONL)
        data_format: Format of training data ('csv' or 'jsonl')
        model_name: Base model to fine-tune
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        logger: Logger instance
        
    Returns:
        Path to the fine-tuned model
    """
    
    # Create configuration
    config = FineTuningConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Initialize fine-tuner
    fine_tuner = CrashSummarizationFineTuner(config, logger)
    
    # Prepare training data
    training_data = fine_tuner.prepare_training_data(training_data_path, data_format)
    
    if len(training_data.narratives) == 0:
        raise ValueError("No training data found. Check your data format and file path.")
    
    # Train the model
    model_path = fine_tuner.train_model(training_data)
    
    return model_path

if __name__ == "__main__":
    # Example usage
    model_path = fine_tune_crash_model(
        training_data_path="training_data.csv",
        data_format="csv",
        model_name="facebook/bart-base",
        output_dir="fine_tuned_models",
        num_epochs=3
    )
    print(f"Fine-tuned model saved to: {model_path}")
