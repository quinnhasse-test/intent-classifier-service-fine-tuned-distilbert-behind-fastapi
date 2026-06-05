"""Evaluation script — accuracy, macro-F1, per-class F1, and confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute accuracy and macro-F1 from flat prediction and label arrays.

    Parameters
    ----------
    predictions:
        1-D array of predicted class indices.
    labels:
        1-D array of ground-truth class indices, same shape as *predictions*.

    Returns
    -------
    Dictionary with keys ``accuracy`` and ``macro_f1``.
    """
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    split: str = "test",
    output_dir: str | Path = "reports",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Run full evaluation of a saved checkpoint against a CLINC150 split.

    Loads the model and tokenizer from *checkpoint_dir*, runs inference on
    *split*, and writes a confusion-matrix PNG and a JSON metrics file to
    *output_dir*.

    Parameters
    ----------
    checkpoint_dir:
        Path produced by a previous training run.
    split:
        Dataset split to evaluate on (``"test"`` or ``"validation"``).
    output_dir:
        Directory where ``metrics.json`` and ``confusion_matrix.png`` are saved.
    batch_size:
        Inference batch size.

    Returns
    -------
    Dictionary with ``accuracy``, ``macro_f1``, and ``per_class_f1`` keys.
    """
    raise NotImplementedError
