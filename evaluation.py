import json
import argparse
from pathlib import Path


def load_results(file_path: str) -> list:
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize_label(label: str) -> bool:
    """Normalize label to boolean."""
    return label.lower() == 'true'


def calculate_metrics(results: list) -> dict:
    """Calculate precision, recall, and F1 score (micro-F1)."""
    tp = fp = fn = tn = 0

    for item in results:
        label = normalize_label(item['label'])
        prediction = normalize_label(item['prediction'])

        if label and prediction:
            tp += 1
        elif not label and prediction:
            fp += 1
        elif label and not prediction:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'total': len(results)
    }


def calculate_macro_f1(results: list) -> dict:
    """Calculate Macro-F1 score (average F1 across all classes)."""
    tp = fp = fn = tn = 0

    for item in results:
        label = normalize_label(item['label'])
        prediction = normalize_label(item['prediction'])

        if label and prediction:
            tp += 1
        elif not label and prediction:
            fp += 1
        elif label and not prediction:
            fn += 1
        else:
            tn += 1

    # Positive class (True) metrics
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0.0

    # Negative class (False) metrics
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0.0

    # Macro averages
    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'f1_pos': f1_pos,
        'precision_neg': precision_neg,
        'recall_neg': recall_neg,
        'f1_neg': f1_neg,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total': len(results)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate fact-checking results')
    parser.add_argument('--file', '-f', type=str,
                        default='results/factool_qa_gpt-4o_single_agent.json',
                        help='Path to results JSON file')
    args = parser.parse_args()

    results = load_results(args.file)
    metrics = calculate_metrics(results)
    macro_metrics = calculate_macro_f1(results)
    
    print(f"File: {args.file}")
    print(f"Total samples: {metrics['total']}")
    print("-" * 40)
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    print("-" * 40)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    print(f"F1 (True):  {macro_metrics['f1_pos']:.4f}")
    print(f"F1 (False): {macro_metrics['f1_neg']:.4f}")
    print(f"Macro-F1:   {macro_metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
