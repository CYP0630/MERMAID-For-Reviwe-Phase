import sys
import re
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

def setup_logging(log_file: str) -> None:
    """
    Configure logging to write to both console and a log file.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers (colorlog config may exist from imports)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    
    
def get_factscore_atomic_dicts(dictionaries: List[Dict]) -> List[Dict]:
    """Extract atomic dictionaries with pattern matching"""
    pattern = re.compile(r'^perplexity_\d+_\d+_\d+_\d+$')
    results = []
    
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if pattern.match(key):
                    results.append(value)
                recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(dictionaries)
    return results


def coerce_lower(value: Any, default: Any = "") -> str:
    """Return a lowercase string representation with a safe fallback."""
    target = value if value is not None else default
    if isinstance(target, str):
        normalized = target.strip()
    else:
        normalized = str(target).strip()
    return normalized.lower()




def load_dataset(dataset_name: str) -> List[str]:
    """Load lines from a dataset file."""
    # Load dataset
    
    claim_records: List[Dict[str, Any]] = []
    
    if dataset_name == "factscore":
        data_path = "data/04_feature/PerplexityAI.json"
        with open(data_path) as f:
            dev_data = json.load(f)
        for datum in dev_data:
            claim_records.extend(get_factscore_atomic_dicts(datum))
        prediction_label_mapping = {
        "supported": "S",
        "not supported": "NS",
        "irrelevant": "IR"
    }
    elif dataset_name == "factool_qa":
        data_path = "dataset/factool_qa/data.jsonl"
        with open(data_path) as f:
            claim_records = [json.loads(line) for line in f]
        prediction_label_mapping = {
        "true": 'true',
        "false": 'false'
    }
    elif dataset_name == "factcheckbench":
        data_path = "dataset/factcheckbench/data.jsonl"
        with open(data_path) as f:
            claim_records = [json.loads(line) for line in f]
        prediction_label_mapping = {
        "true": 'true',
        "false": 'false'
    }
    elif dataset_name == "bingcheck":
        data_path = "dataset/bingcheck/data.jsonl"
        with open(data_path) as f:
            claim_records = [json.loads(line) for line in f]
        prediction_label_mapping = {
        "true": 'true',
        "false": 'false'
    }
    elif dataset_name == "hover2":
        data_path = "dataset/HoVerDev/processed/two_hop_df_new.jsonl"
        with open(data_path) as f:
            claim_records = [json.loads(line) for line in f]
        prediction_label_mapping = {
        "SUPPORTED": 'supported',
        "NOT_SUPPORTED": 'not_supported'
    }
    elif dataset_name == "scifact":
        data_path = "dataset/SciFact-Open/processed/df_new.jsonl"
        with open(data_path) as f:
            claim_records = [json.loads(line) for line in f]
        prediction_label_mapping = {
        "SUPPORT": 'support',
        "CONTRADICT": 'contradict'
    }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return claim_records, prediction_label_mapping


def calculate_metrics(predictions: List, labels: List) -> Dict:
    """
    Calculate accuracy, precision, recall, and F1 for both overall and per-class.

    Args:
        predictions: List of predicted labels
        labels: List of ground truth labels

    Returns:
        Dictionary containing all metrics
    """
    # Normalize predictions and labels to boolean
    def normalize_label(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ["true", "1", "yes"]
        return bool(val)

    preds_bool = [normalize_label(p) for p in predictions]
    labels_bool = [normalize_label(l) for l in labels]

    # Calculate confusion matrix components
    tp = sum(1 for p, l in zip(preds_bool, labels_bool) if p and l)
    tn = sum(1 for p, l in zip(preds_bool, labels_bool) if not p and not l)
    fp = sum(1 for p, l in zip(preds_bool, labels_bool) if p and not l)
    fn = sum(1 for p, l in zip(preds_bool, labels_bool) if not p and l)

    total = len(preds_bool)
    accuracy = (tp + tn) / total if total > 0 else 0

    # True label metrics (positive class)
    true_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    true_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_f1 = 2 * true_precision * true_recall / (true_precision + true_recall) if (true_precision + true_recall) > 0 else 0

    # False label metrics (negative class)
    false_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    false_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_f1 = 2 * false_precision * false_recall / (false_precision + false_recall) if (false_precision + false_recall) > 0 else 0

    # Overall metrics (macro average of both classes)
    precision = (true_precision + false_precision) / 2
    recall = (true_recall + false_recall) / 2
    f1 = (true_f1 + false_f1) / 2

    metrics = {
        "total": total,
        "accuracy": accuracy,
        "overall": {"precision": precision, "recall": recall, "f1": f1},
        "true_label": {"precision": true_precision, "recall": true_recall, "f1": true_f1},
        "false_label": {"precision": false_precision, "recall": false_recall, "f1": false_f1},
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }

    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nOverall:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\nTrue Label:")
    print(f"  Precision: {true_precision:.4f}")
    print(f"  Recall:    {true_recall:.4f}")
    print(f"  F1:        {true_f1:.4f}")
    print(f"\nFalse Label:")
    print(f"  Precision: {false_precision:.4f}")
    print(f"  Recall:    {false_recall:.4f}")
    print(f"  F1:        {false_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    return metrics