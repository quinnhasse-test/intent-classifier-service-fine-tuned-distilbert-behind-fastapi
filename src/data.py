"""Dataset loading and preprocessing for CLINC150 intent classification."""

from __future__ import annotations

from typing import Dict, Tuple

from datasets import DatasetDict, load_dataset
from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64


def load_clinc150(exclude_oos: bool = True) -> DatasetDict:
    """Load the CLINC150 dataset from Hugging Face Hub.

    Parameters
    ----------
    exclude_oos:
        When True, drop the out-of-scope class (label 42) so the
        classifier trains on the 150 in-scope intents only.

    Returns
    -------
    DatasetDict with ``train``, ``validation``, and ``test`` splits.
    Each example has ``text`` (str) and ``intent`` (int) fields.
    """
    ds: DatasetDict = load_dataset("clinc_oos", "plus")
    # Rename 'label' -> 'intent' for clarity throughout the codebase.
    ds = ds.rename_column("label", "intent")
    if exclude_oos:
        # Label 42 is the out-of-scope catch-all in the 'plus' config.
        ds = ds.filter(lambda ex: ex["intent"] != 42)
    return ds


def load_tokenizer() -> DistilBertTokenizerFast:
    """Return a DistilBERT fast tokenizer.

    Returns
    -------
    DistilBertTokenizerFast pre-loaded from ``distilbert-base-uncased``.
    """
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_dataset(
    ds: DatasetDict,
    tokenizer: DistilBertTokenizerFast,
    max_length: int = MAX_LENGTH,
) -> DatasetDict:
    """Tokenize every split in *ds* with *tokenizer*.

    Applies padding and truncation to *max_length*.  Adds ``input_ids``,
    ``attention_mask``, and ``labels`` columns (renamed from ``intent``)
    required by the Hugging Face ``Trainer``.

    Parameters
    ----------
    ds:
        Raw ``DatasetDict`` as returned by :func:`load_clinc150`.
    tokenizer:
        Fast tokenizer from :func:`load_tokenizer`.
    max_length:
        Token sequence length cap.

    Returns
    -------
    DatasetDict with tokenized columns and PyTorch tensor format set.
    """

    def _tokenize(batch: Dict) -> Dict:
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch["intent"]
        return enc

    tokenized = ds.map(_tokenize, batched=True, remove_columns=["text", "intent"])
    tokenized.set_format("torch")
    return tokenized


def get_label_names(ds: DatasetDict) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Extract intent label mappings from the dataset.

    Parameters
    ----------
    ds:
        Raw ``DatasetDict`` as returned by :func:`load_clinc150`.

    Returns
    -------
    Tuple of (id2label, label2id) dictionaries.
    """
    features = ds["train"].features
    names = features["intent"].names
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in enumerate(names)}
    return id2label, label2id
