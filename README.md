# intent-classifier-service

Fine-tunes `distilbert-base-uncased` on [CLINC150](https://huggingface.co/datasets/clinc_oos) and serves the resulting model behind a FastAPI `/predict` endpoint.

## What's here

```
src/
  data.py        dataset loading, tokenization (CLINC150 via HF datasets)
  model.py       model construction and checkpoint loading
  train.py       fine-tuning loop (Hugging Face Trainer + W&B logging)
  evaluate.py    accuracy, macro-F1, per-class F1, confusion matrix
  api/
    main.py      FastAPI app with /health and /predict endpoints
```

## Install

```bash
pip install -r requirements.txt
```

## Run the API

Set `CHECKPOINT_DIR` to a trained model directory, then:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

`/health` returns `{"status": "ok", "model_loaded": true}`.  
`POST /predict` accepts `{"text": "what is the weather today"}` and returns the top intent name and confidence.

## Train

```python
from src.data import load_clinc150, load_tokenizer, tokenize_dataset, get_label_names
from src.model import build_model
from src.train import run_training

ds = load_clinc150()
tokenizer = load_tokenizer()
tokenized = tokenize_dataset(ds, tokenizer)
id2label, label2id = get_label_names(ds)
model = build_model(num_labels=len(id2label), id2label=id2label, label2id=label2id)
run_training(model, tokenizer, tokenized, output_dir="checkpoints/run1")
```

Training with W&B: pass `run_name="my-run"` to `make_training_args`.

## Status

Foundation only — training, evaluation, Docker, and tests are added in subsequent commits.
