from __future__ import annotations


def update_correct_total(predicted, labels, correct: int, total: int) -> tuple[int, int]:
    """
    Update running correct/total counters.

    Args:
        predicted: Tensor of predicted class indices.
        labels: Tensor of ground-truth class indices.
        correct: Running count of correct predictions.
        total: Running count of total samples.

    Returns:
        Updated (correct, total).
    """
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    return correct, total


def compute_accuracy(correct: int, total: int) -> float:
    """
    Compute classification accuracy percentage.

    Args:
        correct: Number of correct predictions.
        total: Number of total predictions.

    Returns:
        Accuracy in percentage [0, 100].
    """
    if total == 0:
        return 0.0
    return 100.0 * correct / total