# advanced_metrics.py

import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# ROUGE metrics
try:
    from rouge_score import rouge_scorer
    HAVE_ROUGE = True
except ImportError:
    HAVE_ROUGE = False

# BLEU metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    HAVE_BLEU = True
except ImportError:
    HAVE_BLEU = False

# BERTScore
try:
    from bert_score import score as bert_score
    HAVE_BERTSCORE = True
except ImportError:
    HAVE_BERTSCORE = False

# Semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SEMANTIC = True
except ImportError:
    HAVE_SEMANTIC = False

@dataclass
class MetricResult:
    """Container for metric results"""
    metric_name: str
    value: float
    details: Optional[Dict[str, Any]] = None

class AdvancedMetricsCalculator:
    """Calculator for advanced NLP metrics"""
    
    def __init__(self, 
                 rouge_types: List[str] = None,
                 bleu_weights: List[Tuple[float, ...]] = None,
                 bertscore_model: str = "microsoft/deberta-xlarge-mnli",
                 semantic_model: str = "all-MiniLM-L6-v2",
                 logger=None):
        """
        Initialize the metrics calculator
        
        Args:
            rouge_types: List of ROUGE types to calculate (default: ['rouge1', 'rouge2', 'rougeL'])
            bleu_weights: List of BLEU n-gram weights (default: [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)])
            bertscore_model: BERTScore model name
            semantic_model: Sentence transformer model for semantic similarity
            logger: Logger instance
        """
        self.logger = logger
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.bleu_weights = bleu_weights or [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
        self.bertscore_model = bertscore_model
        self.semantic_model_name = semantic_model
        
        # Initialize components
        self._init_rouge()
        self._init_bleu()
        self._init_bertscore()
        self._init_semantic()
    
    def _init_rouge(self):
        """Initialize ROUGE scorer"""
        if HAVE_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
            if self.logger:
                self.logger.info(f"ROUGE scorer initialized with types: {self.rouge_types}")
        else:
            self.rouge_scorer = None
            if self.logger:
                self.logger.warning("ROUGE not available. Install with: pip install rouge-score")
    
    def _init_bleu(self):
        """Initialize BLEU components"""
        if HAVE_BLEU:
            self.smoothing = SmoothingFunction().method1
            if self.logger:
                self.logger.info("BLEU scorer initialized")
        else:
            self.smoothing = None
            if self.logger:
                self.logger.warning("BLEU not available. Install with: pip install nltk")
    
    def _init_bertscore(self):
        """Initialize BERTScore"""
        if HAVE_BERTSCORE:
            if self.logger:
                self.logger.info(f"BERTScore initialized with model: {self.bertscore_model}")
        else:
            if self.logger:
                self.logger.warning("BERTScore not available. Install with: pip install bert-score")
    
    def _init_semantic(self):
        """Initialize semantic similarity model"""
        if HAVE_SEMANTIC:
            try:
                self.semantic_model = SentenceTransformer(self.semantic_model_name)
                if self.logger:
                    self.logger.info(f"Semantic similarity model loaded: {self.semantic_model_name}")
            except Exception as e:
                self.semantic_model = None
                if self.logger:
                    self.logger.warning(f"Failed to load semantic model: {e}")
        else:
            self.semantic_model = None
            if self.logger:
                self.logger.warning("Semantic similarity not available. Install with: pip install sentence-transformers scikit-learn")
    
    def calculate_rouge_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE metrics"""
        if not HAVE_ROUGE or not self.rouge_scorer:
            return {}
        
        try:
            rouge_scores = {}
            
            for rouge_type in self.rouge_types:
                scores = []
                for pred, ref in zip(predictions, references):
                    score = self.rouge_scorer.score(ref, pred)
                    scores.append(score[rouge_type].fmeasure)
                
                rouge_scores[f"rouge_{rouge_type}_f1"] = np.mean(scores)
                rouge_scores[f"rouge_{rouge_type}_precision"] = np.mean([
                    self.rouge_scorer.score(ref, pred)[rouge_type].precision 
                    for pred, ref in zip(predictions, references)
                ])
                rouge_scores[f"rouge_{rouge_type}_recall"] = np.mean([
                    self.rouge_scorer.score(ref, pred)[rouge_type].recall 
                    for pred, ref in zip(predictions, references)
                ])
            
            return rouge_scores
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating ROUGE metrics: {e}")
            return {}
    
    def calculate_bleu_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU metrics"""
        if not HAVE_BLEU or not self.smoothing:
            return {}
        
        try:
            bleu_scores = {}
            
            for i, weights in enumerate(self.bleu_weights):
                scores = []
                for pred, ref in zip(predictions, references):
                    # Tokenize
                    pred_tokens = word_tokenize(pred.lower())
                    ref_tokens = [word_tokenize(r.lower()) for r in [ref]]  # BLEU expects list of references
                    
                    # Calculate BLEU
                    score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=self.smoothing)
                    scores.append(score)
                
                bleu_scores[f"bleu_{i+1}"] = np.mean(scores)
            
            return bleu_scores
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating BLEU metrics: {e}")
            return {}
    
    def calculate_bertscore_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore metrics"""
        if not HAVE_BERTSCORE:
            return {}
        
        try:
            # BERTScore calculation
            P, R, F1 = bert_score(predictions, references, model_type=self.bertscore_model, verbose=False)
            
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item()
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating BERTScore metrics: {e}")
            return {}
    
    def calculate_semantic_similarity(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity metrics"""
        if not HAVE_SEMANTIC or not self.semantic_model:
            return {}
        
        try:
            # Get embeddings
            pred_embeddings = self.semantic_model.encode(predictions)
            ref_embeddings = self.semantic_model.encode(references)
            
            # Calculate cosine similarity
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return {
                "semantic_similarity": np.mean(similarities),
                "semantic_similarity_std": np.std(similarities)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating semantic similarity: {e}")
            return {}
    
    def calculate_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate all available metrics"""
        all_metrics = {}
        
        # ROUGE metrics
        rouge_metrics = self.calculate_rouge_metrics(predictions, references)
        all_metrics.update(rouge_metrics)
        
        # BLEU metrics
        bleu_metrics = self.calculate_bleu_metrics(predictions, references)
        all_metrics.update(bleu_metrics)
        
        # BERTScore metrics
        bertscore_metrics = self.calculate_bertscore_metrics(predictions, references)
        all_metrics.update(bertscore_metrics)
        
        # Semantic similarity
        semantic_metrics = self.calculate_semantic_similarity(predictions, references)
        all_metrics.update(semantic_metrics)
        
        return all_metrics
    
    def calculate_single_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate metrics for a single prediction-reference pair"""
        return self.calculate_all_metrics([prediction], [reference])

def normalize_text(text: str) -> str:
    """Normalize text for better metric calculation"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation for some metrics
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def calculate_length_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate length-based metrics"""
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    # Length ratios
    length_ratios = [pred_len / max(ref_len, 1) for pred_len, ref_len in zip(pred_lengths, ref_lengths)]
    
    return {
        "avg_prediction_length": np.mean(pred_lengths),
        "avg_reference_length": np.mean(ref_lengths),
        "avg_length_ratio": np.mean(length_ratios),
        "length_ratio_std": np.std(length_ratios)
    }

# Convenience function for quick metric calculation
def calculate_metrics(predictions: List[str], 
                     references: List[str], 
                     rouge_types: List[str] = None,
                     bleu_weights: List[Tuple[float, ...]] = None,
                     bertscore_model: str = "microsoft/deberta-xlarge-mnli",
                     semantic_model: str = "all-MiniLM-L6-v2",
                     logger=None) -> Dict[str, float]:
    """
    Calculate all available metrics for predictions and references
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        rouge_types: ROUGE types to calculate
        bleu_weights: BLEU n-gram weights
        bertscore_model: BERTScore model name
        semantic_model: Semantic similarity model
        logger: Logger instance
        
    Returns:
        Dictionary of metric names and values
    """
    calculator = AdvancedMetricsCalculator(
        rouge_types=rouge_types,
        bleu_weights=bleu_weights,
        bertscore_model=bertscore_model,
        semantic_model=semantic_model,
        logger=logger
    )
    
    # Calculate all metrics
    metrics = calculator.calculate_all_metrics(predictions, references)
    
    # Add length metrics
    length_metrics = calculate_length_metrics(predictions, references)
    metrics.update(length_metrics)
    
    return metrics

if __name__ == "__main__":
    # Example usage
    predictions = [
        "Unit 1 failed to control speed and struck Unit 2 from behind.",
        "Vehicle 1 collided with Vehicle 2 due to excessive speed."
    ]
    references = [
        "Unit 1 failed to control speed and struck Unit 2 on the back end.",
        "Unit 1 was traveling too fast and hit Unit 2 from behind."
    ]
    
    metrics = calculate_metrics(predictions, references)
    print("Calculated metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
