"""Fine-tuning loop for the CLINC150 intent classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
)


def make_training_args(
    output_dir: str | Path,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = False,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    """Build a ``TrainingArguments`` object for fine-tuning.

    Parameters
    ----------
    output_dir:
        Directory where checkpoints and the final model are saved.
    num_epochs:
        Total training epochs.
    batch_size:
        Per-device batch size for both training and evaluation.
    learning_rate:
        Peak learning rate for AdamW with linear decay.
    warmup_ratio:
        Fraction of total steps used for the linear warm-up phase.
    weight_decay:
        L2 regularisation coefficient.
    fp16:
        Enable mixed-precision training (requires a CUDA device).
    run_name:
        W&B run name.  When ``None``, Hugging Face generates one
        automatically.

    Returns
    -------
    TrainingArguments ready to be passed to ``Trainer``.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(Path(output_dir) / "logs"),
        report_to="wandb" if run_name is not None else "none",
        run_name=run_name,
    )


def run_training(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    tokenized_ds,
    output_dir: str | Path,
    training_args: Optional[TrainingArguments] = None,
) -> None:
    """Fine-tune *model* on *tokenized_ds* and save the best checkpoint.

    Parameters
    ----------
    model:
        Freshly built classification model from :func:`src.model.build_model`.
    tokenizer:
        Tokenizer to save alongside the model.
    tokenized_ds:
        ``DatasetDict`` with ``train``, ``validation``, and ``test`` splits
        as returned by :func:`src.data.tokenize_dataset`.
    output_dir:
        Destination for the final model files.
    training_args:
        Pre-built ``TrainingArguments``.  When ``None``,
        :func:`make_training_args` is called with defaults.
    """
    from transformers import Trainer

    if training_args is None:
        training_args = make_training_args(output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
