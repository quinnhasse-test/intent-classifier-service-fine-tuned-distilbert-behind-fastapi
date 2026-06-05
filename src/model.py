"""DistilBERT model wrapper for intent classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.data import MODEL_NAME


def build_model(
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> DistilBertForSequenceClassification:
    """Create a DistilBERT sequence-classification model.

    Loads ``distilbert-base-uncased`` weights and attaches a classification
    head with *num_labels* outputs.  The head is randomly initialised; the
    pre-trained encoder weights are kept as starting points for fine-tuning.

    Parameters
    ----------
    num_labels:
        Number of intent classes (150 for CLINC150 in-scope only).
    id2label:
        Mapping from class index to intent name, stored in model config.
    label2id:
        Inverse of *id2label*, stored in model config.

    Returns
    -------
    DistilBertForSequenceClassification ready for training.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def load_checkpoint(
    checkpoint_dir: str | Path,
    device: Optional[str] = None,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """Load a fine-tuned model and its tokenizer from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir:
        Path to a directory produced by ``Trainer.save_model()`` or
        ``model.save_pretrained()``.  Must contain ``config.json`` and
        the weight file.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).  When
        ``None``, defaults to CUDA if available, else CPU.

    Returns
    -------
    Tuple of (model, tokenizer), both moved to *device* and set to eval mode.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DistilBertForSequenceClassification.from_pretrained(str(checkpoint_dir))
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_intent(
    text: str,
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    device: str = "cpu",
    max_length: int = 64,
) -> tuple[str, float]:
    """Run a single inference pass and return the top intent and its confidence.

    Parameters
    ----------
    text:
        Raw utterance string.
    model:
        Fine-tuned classification model in eval mode.
    tokenizer:
        Matching tokenizer.
    device:
        Torch device string.
    max_length:
        Truncation length, must match training settings.

    Returns
    -------
    Tuple of (intent_name, confidence) where confidence is the softmax
    probability of the top class.
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    top_idx = int(probs.argmax())
    confidence = float(probs[top_idx])
    intent_name = model.config.id2label[top_idx]
    return intent_name, confidence
