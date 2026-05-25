"""
train.py
--------
Training pipeline — 5 domain experiments + RoBERTa comparison.
"""

from __future__ import annotations
import numpy as np
from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score

from data.dataset_loader import load_amazon_domain
from models.text_classifier import (
    MAX_LENGTH, load_adapted_model, load_baseline_model,
    load_roberta_model, run_domain_adaptive_pretraining,
)


def _tokenize(dataset: Dataset, tokenizer) -> Dataset:
    def preprocess(batch):
        return tokenizer(
            batch["text"], truncation=True,
            padding="max_length", max_length=MAX_LENGTH,
        )
    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="macro"),
    }


def _get_training_args(output_dir: str, epochs: int = 3, batch_size: int = 8):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
    )


def _run_trainer(model, tokenizer, train_ds, eval_ds, output_dir, epochs):
    tok_train = _tokenize(train_ds, tokenizer)
    tok_eval  = _tokenize(eval_ds,  tokenizer)
    trainer = Trainer(
        model=model,
        args=_get_training_args(output_dir, epochs=epochs),
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


# ── Baseline DistilBERT ────────────────────────────────────────────────────────
def train_baseline(
    source_domain: str = "Electronics",
    train_samples: int = 2000,
    eval_samples: int = 400,
    output_dir: str = "outputs/baseline_model",
    epochs: int = 3,
) -> str:
    print("\n" + "="*60)
    print("BASELINE: DistilBERT (no adaptation)")
    print(f"  Source: {source_domain} | Samples: {train_samples}")
    print("="*60)
    train_ds = load_amazon_domain(source_domain, split="train", limit=train_samples)
    eval_ds  = load_amazon_domain(source_domain, split="test",  limit=eval_samples)
    tokenizer, model = load_baseline_model()
    return _run_trainer(model, tokenizer, train_ds, eval_ds, output_dir, epochs)


# ── RoBERTa Vanilla ────────────────────────────────────────────────────────────
def train_roberta(
    source_domain: str = "Electronics",
    train_samples: int = 2000,
    eval_samples: int = 400,
    output_dir: str = "outputs/roberta_model",
    epochs: int = 3,
) -> str:
    print("\n" + "="*60)
    print("COMPARISON: RoBERTa-Base Vanilla")
    print(f"  Source: {source_domain} | Samples: {train_samples}")
    print("="*60)
    train_ds = load_amazon_domain(source_domain, split="train", limit=train_samples)
    eval_ds  = load_amazon_domain(source_domain, split="test",  limit=eval_samples)
    tokenizer, model = load_roberta_model()
    return _run_trainer(model, tokenizer, train_ds, eval_ds, output_dir, epochs)


# ── DAPT DistilBERT ────────────────────────────────────────────────────────────
def train_with_dapt(
    source_domain: str = "Electronics",
    target_domain: str = "Sports_and_Outdoors",
    train_samples: int = 2000,
    eval_samples: int = 400,
    dapt_samples: int = 5000,
    dapt_epochs: int = 3,
    finetune_epochs: int = 3,
    dapt_dir: str = "outputs/dapt_pretrained",
    output_dir: str = "outputs/dapt_finetuned",
) -> str:
    print("\n" + "="*60)
    print("DAPT: DistilBERT with Domain Adaptation")
    print(f"  Source: {source_domain} → Target: {target_domain}")
    print("="*60)

    # Phase 1: DAPT
    print("\n--- PHASE 1: Domain-Adaptive Pre-Training (NO LABELS) ---")
    target_raw   = load_amazon_domain(target_domain, split="train", limit=dapt_samples)
    domain_texts = target_raw["text"]
    dapt_dir     = run_domain_adaptive_pretraining(
        domain_texts=domain_texts, output_dir=dapt_dir, num_epochs=dapt_epochs,
    )

    # Phase 2: Fine-tune
    print("\n--- PHASE 2: Fine-tuning on Source Domain ---")
    train_ds = load_amazon_domain(source_domain, split="train", limit=train_samples)
    eval_ds  = load_amazon_domain(source_domain, split="test",  limit=eval_samples)
    tokenizer, model = load_adapted_model(dapt_dir)
    return _run_trainer(model, tokenizer, train_ds, eval_ds, output_dir, finetune_epochs)


# ── Multi-Domain Experiment ────────────────────────────────────────────────────
def run_multi_domain_experiment(
    source_domain: str = "Electronics",
    target_domains: list = None,
    train_samples: int = 500,
    eval_samples: int = 100,
    dapt_samples: int = 500,
    epochs: int = 1,
    output_base: str = "outputs/multi_domain",
) -> dict:
    """
    Run experiments across 5 domains — proves generalization.
    Source trained once, tested on all target domains.
    """
    if target_domains is None:
        target_domains = [
            "Sports_and_Outdoors",
            "Clothing_Shoes_and_Jewelry",
            "Home_and_Kitchen",
            "Toys_and_Games",
        ]

    print("\n" + "="*60)
    print("MULTI-DOMAIN GENERALIZATION EXPERIMENT")
    print(f"  Source: {source_domain}")
    print(f"  Targets: {target_domains}")
    print("="*60)

    results = {}
    for target in target_domains:
        print(f"\n  Running: {source_domain} → {target}")
        out_dir = f"{output_base}/{target.lower()[:8]}"
        try:
            dapt_out = train_with_dapt(
                source_domain=source_domain,
                target_domain=target,
                train_samples=train_samples,
                eval_samples=eval_samples,
                dapt_samples=dapt_samples,
                dapt_epochs=epochs,
                finetune_epochs=epochs,
                dapt_dir=f"{out_dir}_dapt",
                output_dir=f"{out_dir}_finetuned",
            )
            results[target] = {"status": "done", "model_dir": dapt_out}
        except Exception as e:
            print(f"  [!] {target} failed: {e}")
            results[target] = {"status": "failed", "error": str(e)}

    return results
