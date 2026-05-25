"""
text_classifier.py
------------------
Models:
    1. DistilBERT Baseline
    2. DAPT-DistilBERT (our solution)
    3. RoBERTa Vanilla (comparison)
    4. Aspect Extraction with categorization
"""

from __future__ import annotations
import torch
import numpy as np
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


BASE_MODEL    = "distilbert-base-uncased"
ROBERTA_MODEL = "roberta-base"
MAX_LENGTH    = 256
NUM_LABELS    = 2


def load_tokenizer(model_name: str = BASE_MODEL):
    return AutoTokenizer.from_pretrained(model_name)


def load_baseline_model(model_name: str = BASE_MODEL, num_labels: int = NUM_LABELS):
    """Vanilla DistilBERT — no adaptation. BASELINE."""
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return tokenizer, model


def load_roberta_model(num_labels: int = NUM_LABELS):
    """RoBERTa-Base Vanilla — for comparison with our DAPT model."""
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL, num_labels=num_labels
    )
    return tokenizer, model


def run_domain_adaptive_pretraining(
    domain_texts: list[str],
    output_dir: str = "outputs/dapt_model",
    base_model: str = BASE_MODEL,
    num_epochs: int = 3,
    batch_size: int = 8,
) -> str:
    """DAPT: MLM pre-training on unlabeled target domain text."""
    print(f"\n[DAPT] {len(domain_texts)} texts | Model: {base_model}")
    print(f"[DAPT] Phase 1 — NO LABELS NEEDED\n")

    tokenizer = load_tokenizer(base_model)
    model     = AutoModelForMaskedLM.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True,
            padding="max_length", max_length=MAX_LENGTH
        )

    dataset   = Dataset.from_dict({"text": domain_texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DAPT] Saved → {output_dir}")
    return output_dir


def load_adapted_model(dapt_model_dir: str, num_labels: int = NUM_LABELS):
    tokenizer = load_tokenizer(dapt_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        dapt_model_dir, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    return tokenizer, model


def load_for_inference(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT EXTRACTION WITH CATEGORIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Aspect keyword → category mapping (thesis requirement)
ASPECT_CATEGORIES = {
    # Performance
    "battery":      "Performance", "speed":     "Performance",
    "performance":  "Performance", "fast":      "Performance",
    "slow":         "Performance", "power":     "Performance",
    "drains":       "Performance", "efficient": "Performance",
    "charging":     "Performance",
    # Quality
    "quality":      "Quality",     "durable":   "Quality",
    "durability":   "Quality",     "material":  "Quality",
    "sturdy":       "Quality",     "cheap":     "Quality",
    "solid":        "Quality",     "built":     "Quality",
    "feels":        "Quality",     "finish":    "Quality",
    # Design
    "design":       "Design",      "weight":    "Design",
    "lightweight":  "Design",      "heavy":     "Design",
    "size":         "Design",      "display":   "Design",
    "screen":       "Design",      "color":     "Design",
    "looks":        "Design",      "compact":   "Design",
    # Comfort/Usability
    "grip":         "Usability",   "comfort":   "Usability",
    "comfortable":  "Usability",   "easy":      "Usability",
    "handle":       "Usability",   "ergonomic": "Usability",
    "awkward":      "Usability",   "smooth":    "Usability",
    # Cost
    "price":        "Cost",        "cost":      "Cost",
    "expensive":    "Cost",        "cheap":     "Cost",
    "worth":        "Cost",        "money":     "Cost",
    "affordable":   "Cost",        "value":     "Cost",
    # Logistics
    "delivery":     "Logistics",   "shipping":  "Logistics",
    "arrived":      "Logistics",   "packaging": "Logistics",
    "package":      "Logistics",   "days":      "Logistics",
    # Domain-specific Sports
    "tension":      "Performance", "strings":   "Usability",
    "flex":         "Performance", "traction":  "Usability",
    "aerodynamics": "Performance", "spin":      "Performance",
    # Domain-specific Electronics
    "resolution":   "Quality",     "camera":    "Quality",
    "connectivity": "Performance", "bluetooth": "Performance",
    "signal":       "Performance", "noise":     "Quality",
}

STOP_WORDS = {
    "this", "that", "with", "have", "from", "they", "will", "been",
    "were", "their", "what", "when", "your", "which", "there", "very",
    "just", "also", "more", "some", "than", "then", "them", "into",
    "would", "could", "about", "after", "before", "being", "other",
    "product", "item", "review", "bought", "purchase", "ordered",
    "received", "really", "great", "good", "nice", "love", "like",
    "best", "seems", "gameplay", "poor", "using",
    # Product/brand names — not aspects
    "laptop", "phone", "mobile", "tablet", "computer", "device",
    "samsung", "apple", "sony", "dell", "lenovo", "asus", "iphone",
    "ipad", "macbook", "windows", "android", "brand", "model",
    # Filler words
    "weak", "have", "good", "bad", "okay", "fine", "well",
}


def extract_aspects(
    text: str,
    model_dir: str,
    top_k: int = 5,
    min_word_len: int = 4,
) -> dict:
    """
    Extract + categorize product aspects using attention weights.
    Maps detected aspects to categories: Performance, Quality, Cost, etc.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_attentions=True
    )
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Sentiment
    probs      = torch.softmax(outputs.logits, dim=-1)[0]
    pred       = torch.argmax(probs).item()
    confidence = round(probs[pred].item() * 100, 1)
    sentiment  = "Positive ✅" if pred == 1 else "Negative ❌"

    # Attention-based aspect scoring
    attentions       = outputs.attentions
    avg_attention    = torch.stack(attentions).squeeze(1).mean(dim=0).mean(dim=0)
    token_importance = avg_attention.mean(dim=0)
    tokens           = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    word_scores = {}
    for token, score in zip(tokens, token_importance.tolist()):
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]:
            continue
        if token.startswith("##") or token.startswith("Ġ"):
            token = token.lstrip("##Ġ")
        word = token.lower().strip()
        if len(word) < min_word_len or not word.isalpha():
            continue
        if word in STOP_WORDS:
            continue
        word_scores[word] = word_scores.get(word, 0) + score

    sorted_aspects = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    top_aspects    = [w for w, _ in sorted_aspects[:top_k]]

    # Map aspects to categories
    categorized = {}
    for aspect in top_aspects:
        category = ASPECT_CATEGORIES.get(aspect, "General")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(aspect)

    return {
        "text":        text[:100] + "..." if len(text) > 100 else text,
        "sentiment":   sentiment,
        "confidence":  confidence,
        "aspects":     top_aspects,
        "categorized": categorized,
        "all_scores":  dict(sorted_aspects[:10]),
    }
