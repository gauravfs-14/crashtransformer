# CrashTransformer: Enhanced Crash Narrative Summarization Pipeline

A comprehensive pipeline for generating causal summaries from crash narratives using advanced NLP techniques, including Ollama integration, comprehensive evaluation metrics, and cross-validation.

## 🚀 Features

### Core Functionality
- **Ollama Integration**: Uses Llama 3.1:8b model for causal summary generation
- **BART Fine-tuning**: Trains a BART model for crash narrative summarization
- **Comprehensive Evaluation**: Multiple evaluation metrics and visualizations
- **Cross-Validation**: K-fold cross-validation for robust model evaluation
- **Raw Data Storage**: All results saved for reproducibility and further analysis

### Evaluation Metrics
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
- **BERTScore**: Semantic similarity using BERT embeddings
- **BLEU Score**: N-gram overlap metrics
- **Semantic Similarity**: Cosine similarity using sentence transformers
- **Length Analysis**: Prediction vs reference length analysis
- **Quality Metrics**: Custom quality scoring and error analysis

### Visualizations
- **ROUGE Score Comparisons**: Bar plots and distributions
- **BERTScore Distributions**: Statistical analysis of semantic similarity
- **Length Analysis**: Histograms, scatter plots, and ratio analysis
- **Word Cloud Analysis**: Most frequent words in predictions vs references
- **Correlation Matrix**: Relationships between different metrics
- **Performance by Length**: How model performance varies with text length
- **Error Analysis**: Detailed analysis of prediction errors
- **Cross-Validation Plots**: Fold performance comparisons and distributions

## 📁 Project Structure

```
crashtransformer/
├── crash_transformer_pipeline.py    # Main pipeline
├── src/                             # Source modules
│   ├── __init__.py                 # Package initialization
│   ├── enhanced_evaluation.py      # Comprehensive evaluation module
│   └── cross_validation_module.py  # Cross-validation module
├── inference_script.py              # Inference and analysis script
├── run_comprehensive_evaluation.py  # Standalone evaluation runner
├── test_pipeline.py                 # Test script
├── pyproject.toml                   # Dependencies
├── results/                         # Output directory
│   ├── raw_data/                    # Raw evaluation data
│   ├── plots/                       # Generated visualizations
│   ├── metrics/                     # Evaluation metrics
│   └── cross_validation/            # Cross-validation results
├── models/                          # Trained models
└── logs/                           # Log files
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd crashtransformer
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Install Ollama** (for causal summary generation):
   ```bash
   # Follow instructions at https://ollama.ai/
   ollama pull llama3.1:8b
   ```

## 📊 Usage

### 1. Basic Pipeline

Run the complete pipeline:
```bash
python crash_transformer_pipeline.py
```

Run in test mode (reduced dataset and epochs):
```bash
python crash_transformer_pipeline.py --test
```

### 2. Comprehensive Evaluation

Run comprehensive evaluation on existing results:
```bash
python run_comprehensive_evaluation.py
```

With cross-validation:
```bash
python run_comprehensive_evaluation.py --cross-validation --cv-folds 5 --cv-epochs 2
```

### 3. Visualization Generation (No Inference Required)

Create all visualizations from existing saved data:
```bash
python create_visualizations.py
```

Create specific types of visualizations:
```bash
# Basic visualizations only
python create_visualizations.py --visualization-type basic

# Enhanced visualizations only
python create_visualizations.py --visualization-type enhanced

# Cross-validation visualizations only
python create_visualizations.py --visualization-type cv
```

Check available data and visualization options:
```bash
python quick_visualizations.py
```

### 4. Inference and Analysis

Load saved model and results:
```bash
python inference_script.py --evaluate
```

Generate summary for specific text:
```bash
python inference_script.py --text "Your crash narrative here"
```

View saved metrics:
```bash
python inference_script.py --metrics
```

### 5. Cross-Validation

Run cross-validation independently:
```bash
python cross_validation_module.py
```

## 📈 Evaluation Metrics

### ROUGE Scores
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence

### BERTScore
- **Precision**: How much of the prediction is semantically similar to reference
- **Recall**: How much of the reference is captured in the prediction
- **F1**: Harmonic mean of precision and recall

### BLEU Score
- **BLEU-1**: Unigram precision
- **BLEU-2**: Bigram precision
- **BLEU-3**: Trigram precision
- **BLEU-4**: 4-gram precision

### Semantic Similarity
- **Cosine Similarity**: Semantic similarity using sentence transformers
- **Distribution Analysis**: Statistical analysis of similarity scores

### Length Metrics
- **Length Ratio**: Prediction length / Reference length
- **Length Distribution**: Statistical analysis of text lengths
- **Performance by Length**: How metrics vary with text length

## 📊 Visualizations

### Generated Plots
1. **rouge_scores.png**: ROUGE score comparisons
2. **bertscore_distribution.png**: BERTScore distribution analysis
3. **length_analysis.png**: Length analysis and ratios
4. **semantic_similarity.png**: Semantic similarity distribution
5. **bleu_distribution.png**: BLEU score distribution
6. **summary_quality_analysis.png**: Quality score analysis
7. **word_cloud_analysis.png**: Word frequency analysis
8. **correlation_matrix.png**: Metric correlations
9. **performance_by_length.png**: Performance vs length
10. **error_analysis.png**: Error analysis and categorization

### Cross-Validation Plots
1. **fold_performance.png**: Performance across folds
2. **metric_distribution.png**: Metric distributions across folds
3. **learning_curves.png**: Training curves (if available)
4. **prediction_quality.png**: Quality analysis by fold

## 💾 Data Storage

### Raw Data
All raw data is saved for reproducibility:
- **evaluation_data.json**: Predictions, references, and narratives
- **comprehensive_metrics.json**: All computed metrics
- **summary_statistics.txt**: Human-readable summary

### Cross-Validation Data
- **cv_results.json**: Complete cross-validation results
- **aggregated_metrics.json**: Aggregated metrics across folds
- **cv_summary.txt**: Cross-validation summary

## 🎨 Visualization Without Inference

**All visualizations can be created without running full inference!** The system stores raw data and provides multiple ways to generate plots:

### Methods to Create Visualizations
1. **Standalone Script**: `python create_visualizations.py`
2. **Comprehensive Evaluation**: `python run_comprehensive_evaluation.py`
3. **Direct Function Call**: From enhanced evaluation module
4. **Inference Script**: `python inference_script.py --evaluate`

### What's Stored for Reproducibility
- ✅ **Predictions**: All model predictions saved to CSV
- ✅ **References**: Ground truth summaries
- ✅ **Narratives**: Original crash narratives
- ✅ **Metrics**: All computed evaluation metrics
- ✅ **Raw Data**: Complete evaluation data in JSON format
- ✅ **Cross-Validation Results**: All fold results and aggregations

### Benefits
- 🚀 **Fast**: No need to rerun training or inference
- 🔄 **Reproducible**: All data preserved for future analysis
- 🎯 **Flexible**: Multiple ways to generate visualizations
- 📊 **Comprehensive**: 10+ different types of plots available

## 🔧 Configuration

### Pipeline Parameters
- **Model**: BART-base (configurable)
- **Training epochs**: 3 (reduced in test mode)
- **Batch size**: 4 (reduced in test mode)
- **Learning rate**: 2e-5
- **Max sequence length**: 512 (input), 64 (output)

### Evaluation Parameters
- **Evaluation samples**: 100 (configurable)
- **Cross-validation folds**: 5 (configurable)
- **Cross-validation epochs**: 2 (configurable)

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not available**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3.1:8b
   ```

2. **Missing dependencies**:
   ```bash
   uv add bert-score nltk sentence-transformers wordcloud
   ```

3. **CUDA out of memory**:
   - Reduce batch size in training arguments
   - Use CPU if GPU memory is insufficient

4. **Evaluation metrics fail**:
   - Check if all required packages are installed
   - Verify data format and content

### Log Files
Check log files in the `logs/` directory for detailed error information:
```bash
tail -f logs/crashtransformer_*.log
```

## 📊 Results Interpretation

### Good Performance Indicators
- **ROUGE-1 F1 > 0.3**: Good unigram overlap
- **ROUGE-2 F1 > 0.1**: Good bigram overlap
- **BERTScore F1 > 0.8**: High semantic similarity
- **Length ratio close to 1.0**: Appropriate summary length

### Cross-Validation Stability
- **Low standard deviation** across folds indicates stable performance
- **Consistent performance** across different data splits
- **Similar training curves** across folds

## 🔬 Advanced Usage

### Custom Evaluation
```python
from src.enhanced_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator("results")
metrics = evaluator.compute_all_metrics(predictions, references, narratives)
evaluator.create_comprehensive_visualizations(predictions, references, narratives)
```

### Custom Cross-Validation
```python
from src.cross_validation_module import CrossValidator

cv = CrossValidator("results")
results = cv.run_cross_validation(narratives, summaries, n_splits=5, epochs=2)
```

### Batch Processing
```python
from crash_transformer_pipeline import generate_causal_summaries_batch

summaries = generate_causal_summaries_batch(narratives, batch_size=5)
```

<!-- ## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{crashtransformer2024,
  title={CrashTransformer: Enhanced Crash Narrative Summarization Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/crashtransformer}
}
``` -->

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers for the BART model
- Ollama for local LLM inference
- The crash analysis community for domain expertise
