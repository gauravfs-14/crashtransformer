# 🎯 Fine-Tuning Guide for CrashTransformer

This guide explains how to fine-tune transformer models specifically for crash summarization tasks.

## 🚀 Quick Start

### 1. Prepare Training Data

```bash
# Create synthetic training data (for testing)
python crashtransformer.py prepare-data --source synthetic --output training_data.csv --num_examples 1000

# Or prepare from pipeline outputs
python crashtransformer.py prepare-data --source pipeline --graphs_file artifacts/crash_graphs.jsonl --summaries_file artifacts/crash_summaries.jsonl --output training_data.csv
```

### 2. Train the Model

```bash
# Basic training
python crashtransformer.py train --training_data training_data.csv --num_epochs 3 --batch_size 8

# Advanced training with custom parameters
python crashtransformer.py train --training_data training_data.csv --model_name facebook/bart-base --num_epochs 5 --batch_size 16 --learning_rate 3e-5 --output_dir my_fine_tuned_model
```

### 3. Use Fine-Tuned Model

```bash
# Run pipeline with fine-tuned model
python crashtransformer.py run --csv crashes.csv --fine_tuned_model my_fine_tuned_model/final_model
```

## 📊 Training Data Format

### CSV Format

```csv
Narrative,Summary,plan_lines,Crash_ID,Crash_Severity,City
"Unit 1 failed to control speed and struck Unit 2 from behind.","Unit 1 failed to control speed and rear-ended Unit 2.","['1) failed to control speed -> rear-end collision']","19955047","Not Injured","Dallas"
```

### JSONL Format

```jsonl
{"narrative": "Unit 1 failed to control speed and struck Unit 2 from behind.", "summary": "Unit 1 failed to control speed and rear-ended Unit 2.", "plan_lines": ["1) failed to control speed -> rear-end collision"], "metadata": {"crash_id": "19955047", "severity": "Not Injured", "city": "Dallas"}}
```

## 🔧 Training Configuration

### Model Parameters

- **`--model_name`**: Base model to fine-tune (default: `facebook/bart-base`)
- **`--num_epochs`**: Number of training epochs (default: 3)
- **`--batch_size`**: Training batch size (default: 8)
- **`--learning_rate`**: Learning rate (default: 5e-5)

### Advanced Parameters

- **`--max_length`**: Maximum input sequence length (default: 512)
- **`--max_target_length`**: Maximum target sequence length (default: 128)
- **`--warmup_steps`**: Number of warmup steps (default: 500)
- **`--weight_decay`**: Weight decay for regularization (default: 0.01)
- **`--fp16`**: Use mixed precision training
- **`--early_stopping_patience`**: Early stopping patience (default: 3)

### Data Parameters

- **`--validation_split`**: Fraction of data for validation (default: 0.1)
- **`--data_format`**: Input data format (`csv` or `jsonl`)

## 📈 Training Process

### 1. Data Preparation

- **Synthetic Data**: Creates realistic crash scenarios for testing
- **Pipeline Data**: Uses outputs from previous pipeline runs
- **Custom Data**: Prepare your own training data in the required format

### 2. Model Training

- **Plan-Conditioned Training**: Uses causal plans to guide summarization
- **Validation Split**: Automatically splits data for training/validation
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Mixed Precision**: Uses FP16 for faster training on compatible hardware

### 3. Model Evaluation

- **Automatic Evaluation**: Tests model on validation data
- **Comprehensive Metrics**: ROUGE, BLEU, BERTScore, semantic similarity
- **Performance Tracking**: Training loss, validation loss, learning rate

## 🎯 Supported Models

### Pre-trained Models

- **BART**: `facebook/bart-base`, `facebook/bart-large`
- **T5**: `t5-base`, `t5-small`, `t5-large`
- **Pegasus**: `google/pegasus-xsum`
- **Any HuggingFace seq2seq model**

### Fine-tuning Benefits

- **Domain-Specific**: Trained on crash narratives
- **Plan-Aware**: Understands causal relationships
- **Quality-Focused**: Optimized for crash summarization metrics
- **Efficient**: Faster inference than general models

## 📊 Training Metrics

### During Training

- **Training Loss**: Model's performance on training data
- **Validation Loss**: Model's performance on validation data
- **Learning Rate**: Current learning rate (with warmup)
- **Gradient Norm**: Gradient clipping information

### After Training

- **ROUGE Scores**: 9 different ROUGE metrics
- **BLEU Scores**: 4 different BLEU metrics
- **BERTScore**: Semantic similarity metrics
- **Length Analysis**: Summary length statistics

## 🚀 Advanced Usage

### Custom Training Data

```python
# Create custom training data
import pandas as pd

data = {
    'Narrative': ['Your crash narratives here...'],
    'Summary': ['Your summaries here...'],
    'plan_lines': ['Your causal plans here...'],
    'Crash_ID': ['Unique IDs'],
    'Crash_Severity': ['Severity levels'],
    'City': ['City names']
}

df = pd.DataFrame(data)
df.to_csv('custom_training_data.csv', index=False)
```

### Multi-GPU Training

```bash
# Use multiple GPUs (if available)
python crashtransformer.py train --training_data data.csv --batch_size 32 --fp16
```

### Hyperparameter Tuning

```bash
# Experiment with different learning rates
python crashtransformer.py train --training_data data.csv --learning_rate 1e-5
python crashtransformer.py train --training_data data.csv --learning_rate 3e-5
python crashtransformer.py train --training_data data.csv --learning_rate 5e-5
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `--batch_size`
   - Use `--fp16` for mixed precision
   - Reduce `--max_length`

2. **Poor Performance**
   - Increase `--num_epochs`
   - Adjust `--learning_rate`
   - Check training data quality

3. **Overfitting**
   - Increase `--early_stopping_patience`
   - Add more training data
   - Reduce `--num_epochs`

### Monitoring Training

```bash
# Check training logs
tail -f logs/crashtransformer-*.log

# Monitor GPU usage
nvidia-smi -l 1
```

## 📁 Output Structure

```
fine_tuned_models/
├── final_model/           # Final trained model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── logs/                 # Training logs
│   └── events.out.tfevents.*
├── evaluation_results.json  # Evaluation metrics
└── training_args.bin     # Training configuration
```

## 🎯 Best Practices

### Data Quality

- **Clean Narratives**: Remove irrelevant information
- **Consistent Summaries**: Use consistent style and format
- **Balanced Data**: Include various crash types and severities
- **Sufficient Volume**: At least 1000 examples for good results

### Training Strategy

- **Start Small**: Begin with fewer epochs and smaller batch sizes
- **Monitor Metrics**: Watch for overfitting and underfitting
- **Validate Regularly**: Use validation split to check performance
- **Save Checkpoints**: Keep intermediate models for comparison

### Model Selection

- **BART**: Good for general summarization
- **T5**: Excellent for text-to-text tasks
- **Pegasus**: Optimized for abstractive summarization
- **Custom Models**: Fine-tune domain-specific models

## 🚀 Production Deployment

### Model Optimization

```bash
# Quantize model for faster inference
python -c "
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('fine_tuned_models/final_model')
model.save_pretrained('optimized_model')
"
```

### Batch Processing

```bash
# Process large datasets with fine-tuned model
python crashtransformer.py run --csv large_dataset.csv --fine_tuned_model fine_tuned_models/final_model --batch_size 32
```

## 📚 Additional Resources

- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers/)
- **Training Arguments**: [Reference](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)
- **Model Hub**: [Pre-trained Models](https://huggingface.co/models)

---

**🎯 Happy Fine-Tuning!** Your crash summarization model will be much more effective with domain-specific training! 🚗💥🤖
