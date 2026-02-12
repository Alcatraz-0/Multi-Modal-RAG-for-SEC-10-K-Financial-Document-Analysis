"""
Evaluation metrics for retrieval and QA performance
"""
from typing import List, Set, Any
import numpy as np
from collections import Counter
import re


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Normalize strings
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    return 1.0 if pred_norm == gt_norm else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_recall_at_k(retrieved: List[Any], relevant: Set[Any], k: int) -> float:
    """
    Compute Recall@k

    Args:
        retrieved: List of retrieved items (ordered by rank)
        relevant: Set of relevant items
        k: Cutoff position

    Returns:
        Recall@k score
    """
    if len(relevant) == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    num_relevant_retrieved = len(retrieved_at_k & relevant)

    return num_relevant_retrieved / len(relevant)


def compute_mrr(retrieved: List[Any], relevant: Set[Any]) -> float:
    """
    Compute Mean Reciprocal Rank

    Args:
        retrieved: List of retrieved items (ordered by rank)
        relevant: Set of relevant items

    Returns:
        MRR score
    """
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0


def compute_mae_mape(predictions: List[float], ground_truth: List[float]) -> tuple:
    """
    Compute MAE and MAPE for numeric answers

    Args:
        predictions: List of predicted numbers
        ground_truth: List of ground truth numbers

    Returns:
        Tuple of (MAE, MAPE)
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # MAE
    mae = np.mean(np.abs(predictions - ground_truth))

    # MAPE
    mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100

    return mae, mape


def compute_faithfulness(answer: str, evidence: List[str]) -> float:
    """
    Compute faithfulness: fraction of answer claims in evidence

    Args:
        answer: Generated answer
        evidence: List of evidence texts

    Returns:
        Faithfulness score (0-1)
    """
    # Simple implementation: check if answer tokens appear in evidence
    answer_tokens = set(normalize_answer(answer).split())

    if len(answer_tokens) == 0:
        return 0.0

    evidence_text = " ".join(evidence)
    evidence_tokens = set(normalize_answer(evidence_text).split())

    supported_tokens = answer_tokens & evidence_tokens
    faithfulness = len(supported_tokens) / len(answer_tokens)

    return faithfulness


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison"""
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text
