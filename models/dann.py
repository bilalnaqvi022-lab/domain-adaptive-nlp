"""
dann.py
-------
DANN: Domain-Adversarial Neural Network
Gururangan et al. / Ganin et al. 2016

Architecture:
    DistilBERT encoder
         ↓
    [CLS] representation
         ↓
    ┌────────────────────────────┐
    │  Sentiment Classifier      │  → positive/negative
    └────────────────────────────┘
         ↓
    Gradient Reversal Layer (GRL)
         ↓
    ┌────────────────────────────┐
    │  Domain Classifier         │  → source/target domain
    └────────────────────────────┘

Key idea:
    - Sentiment classifier learns TO classify correctly
    - Domain classifier learns to distinguish source vs target
    - GRL REVERSES gradients → forces encoder to produce
      domain-INVARIANT features
    - Result: model cannot tell which domain it's in
              but still classifies sentiment correctly
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT REVERSAL LAYER
# ══════════════════════════════════════════════════════════════════════════════
class GradientReversalFunction(torch.autograd.Function):
    """
    Forward pass: identity (passes input unchanged)
    Backward pass: multiplies gradient by -lambda
    This is what makes domain adaptation work!
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ══════════════════════════════════════════════════════════════════════════════
# DANN MODEL
# ══════════════════════════════════════════════════════════════════════════════
class DANNModel(nn.Module):
    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        num_sentiment_labels: int = 2,
        lambda_: float = 1.0,
    ):
        super().__init__()

        # Shared encoder (DistilBERT)
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size  = self.encoder.config.hidden_size  # 768

        # Sentiment classifier head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_sentiment_labels),
        )

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_=lambda_)

        # Domain classifier head (source=0, target=1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, lambda_=None):
        # Update lambda if provided (for scheduling)
        if lambda_ is not None:
            self.grl.lambda_ = lambda_

        # Shared encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Sentiment classification (normal gradient flow)
        sentiment_logits = self.sentiment_classifier(cls_output)

        # Domain classification (reversed gradient flow)
        reversed_features = self.grl(cls_output)
        domain_logits     = self.domain_classifier(reversed_features)

        return sentiment_logits, domain_logits


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def train_dann(
    source_texts: list[str],
    source_labels: list[int],
    target_texts: list[str],           # NO labels needed for target!
    output_dir: str = "outputs/dann_model",
    base_model: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    lambda_: float = 1.0,
) -> str:
    """
    Train DANN model.

    Args:
        source_texts  : Labeled source domain texts (Electronics)
        source_labels : Sentiment labels for source (0/1)
        target_texts  : UNLABELED target domain texts (Sports)
        output_dir    : Where to save model
        lambda_       : GRL strength (higher = stronger domain confusion)

    Returns:
        Path to saved model
    """
    print(f"\n[DANN] Training Domain-Adversarial Neural Network")
    print(f"[DANN] Source samples : {len(source_texts)} (labeled)")
    print(f"[DANN] Target samples : {len(target_texts)} (unlabeled)")
    print(f"[DANN] Lambda (GRL)   : {lambda_}")
    print(f"[DANN] Epochs         : {epochs}\n")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = DANNModel(base_model=base_model, lambda_=lambda_).to(device)

    # Tokenize source (labeled)
    src_enc = tokenizer(
        source_texts, truncation=True, padding=True,
        max_length=max_length, return_tensors="pt"
    )
    src_labels_t  = torch.tensor(source_labels)
    src_domain_t  = torch.zeros(len(source_texts), dtype=torch.long)  # domain=0

    # Tokenize target (unlabeled — only domain labels)
    tgt_enc = tokenizer(
        target_texts, truncation=True, padding=True,
        max_length=max_length, return_tensors="pt"
    )
    tgt_domain_t = torch.ones(len(target_texts), dtype=torch.long)   # domain=1

    # DataLoaders
    src_dataset = TensorDataset(
        src_enc["input_ids"], src_enc["attention_mask"],
        src_labels_t, src_domain_t
    )
    tgt_dataset = TensorDataset(
        tgt_enc["input_ids"], tgt_enc["attention_mask"],
        tgt_domain_t
    )
    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True)

    # Loss functions
    sentiment_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn    = nn.CrossEntropyLoss()
    optimizer         = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tgt_iter   = iter(tgt_loader)

        for step, src_batch in enumerate(src_loader):
            # Lambda scheduling: gradually increase GRL strength
            p      = (step + epoch * len(src_loader)) / (epochs * len(src_loader))
            lambda_scheduled = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            src_ids, src_mask, src_sent_labels, src_dom_labels = [
                x.to(device) for x in src_batch
            ]

            # Forward on source
            sent_logits, dom_logits_src = model(
                src_ids, src_mask, lambda_=lambda_scheduled
            )

            # Sentiment loss (source only — labeled)
            loss_sentiment = sentiment_loss_fn(sent_logits, src_sent_labels)

            # Domain loss on source
            loss_domain_src = domain_loss_fn(dom_logits_src, src_dom_labels)

            # Forward on target (domain loss only)
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter  = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            tgt_ids, tgt_mask, tgt_dom_labels = [x.to(device) for x in tgt_batch]
            _, dom_logits_tgt = model(tgt_ids, tgt_mask, lambda_=lambda_scheduled)
            loss_domain_tgt   = domain_loss_fn(dom_logits_tgt, tgt_dom_labels)

            # Total loss
            loss = loss_sentiment + 0.5 * (loss_domain_src + loss_domain_tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(src_loader)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Lambda: {lambda_scheduled:.3f}")

    # Save model + tokenizer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/dann_model.pt")
    tokenizer.save_pretrained(output_dir)

    # Save config for inference
    import json
    json.dump({"base_model": base_model}, open(f"{output_dir}/config.json", "w"))

    print(f"\n[DANN] Model saved → {output_dir}")
    return output_dir


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_dann(
    model_dir: str,
    test_texts: list[str],
    test_labels: list[int],
    base_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    batch_size: int = 32,
) -> dict:
    """Evaluate DANN on target domain test set."""
    import json
    config    = json.load(open(f"{model_dir}/config.json"))
    base_model = config.get("base_model", base_model)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = DANNModel(base_model=base_model).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/dann_model.pt", map_location=device))
    model.eval()

    all_preds = []
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i: i + batch_size]
        enc   = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            sent_logits, _ = model(enc["input_ids"], enc["attention_mask"])
            preds = torch.argmax(sent_logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)

    return {
        "accuracy": accuracy_score(test_labels, all_preds),
        "f1_macro": f1_score(test_labels, all_preds, average="macro"),
        "predictions": all_preds,
    }
