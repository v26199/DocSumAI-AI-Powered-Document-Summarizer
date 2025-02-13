import logging
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_summary(reference_text, generated_text):
    """Evaluate summary using BLEU, ROUGE, and METEOR metrics."""
    if not reference_text or not generated_text:
        logging.warning("Empty input for evaluation")
        return {}
    
    try:
        # Tokenize inputs for BLEU calculation
        ref_tokens = word_tokenize(reference_text.lower())
        gen_tokens = word_tokenize(generated_text.lower())
        
        logging.info("Reference tokens (first 20): %s", ref_tokens[:20])
        logging.info("Generated tokens (first 20): %s", gen_tokens[:20])
        
        # Prepare data for corpus BLEU
        list_of_references = [ref_tokens]
        hypotheses = [gen_tokens]
        
        metrics = {
            "BLEU-4": 0,
            "ROUGE-1": 0,
            "ROUGE-2": 0,
            "ROUGE-L": 0,
            "METEOR": 0
        }
        
        # Calculate BLEU-4 score
        try:
            metrics["BLEU-4"] = corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        except Exception as e:
            logging.error(f"BLEU calculation error: {e}")
        
        # Calculate ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_text, generated_text)
            metrics["ROUGE-1"] = scores['rouge1'].fmeasure
            metrics["ROUGE-2"] = scores['rouge2'].fmeasure
            metrics["ROUGE-L"] = scores['rougeL'].fmeasure
        except Exception as e:
            logging.error(f"ROUGE calculation error: {e}")
        
        # Calculate METEOR score
        try:
            metrics["METEOR"] = meteor_score([reference_text], generated_text)
        except Exception as e:
            logging.error(f"METEOR calculation error: {e}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return {}

def display_evaluation(scores):
    """
    Create a bar chart visualization of the evaluation metrics.
    Returns a matplotlib figure object.
    """
    if not scores:
        return None
    
    try:
        plt.clf()  # Clear previous figures
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        labels = list(scores.keys())
        values = [round(v, 4) if v is not None else 0 for v in scores.values()]
        colors = sns.color_palette("husl", len(labels))
        
        # Create bar plot
        bars = ax.bar(labels, values, color=colors)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Configure plot
        ax.set_ylim(0, 1)
        ax.set_title('Summary Quality Metrics', fontsize=14, pad=20)
        ax.set_xlabel('Metrics', labelpad=10)
        ax.set_ylabel('Scores', labelpad=10)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return None