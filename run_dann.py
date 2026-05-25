"""
run_dann.py
-----------
Run DANN experiment — small scale proof of concept.
Electronics (source) → Sports (target) with gradient reversal.

Usage: python run_dann.py
"""

from data.dataset_loader import load_amazon_domain, dataset_to_pandas
from models.dann import train_dann, evaluate_dann
from models.text_classifier import extract_aspects

print("\n" + "="*60)
print("DANN: Domain-Adversarial Neural Network Experiment")
print("Gradient Reversal Layer — Domain Invariant Features")
print("="*60)

# Load source (labeled)
print("\nLoading source domain (Electronics — labeled)...")
src_ds     = load_amazon_domain("Electronics", split="train", limit=400)
src_df     = dataset_to_pandas(src_ds)
src_texts  = src_df["text"].tolist()
src_labels = src_df["label"].tolist()

# Load target (unlabeled — only text needed)
print("Loading target domain (Sports — UNLABELED)...")
tgt_ds    = load_amazon_domain("Sports_and_Outdoors", split="train", limit=200)
tgt_df    = dataset_to_pandas(tgt_ds)
tgt_texts = tgt_df["text"].tolist()   # labels intentionally NOT used

# Load test set
print("Loading target test set...")
test_ds     = load_amazon_domain("Sports_and_Outdoors", split="test", limit=150)
test_df     = dataset_to_pandas(test_ds)
test_texts  = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

# Train DANN
dann_dir = train_dann(
    source_texts=src_texts,
    source_labels=src_labels,
    target_texts=tgt_texts,
    output_dir="outputs/dann_model",
    epochs=2,
    batch_size=8,
    lambda_=0.5,
)

# Evaluate
print("\n[DANN] Evaluating on target domain (Sports)...")
results = evaluate_dann(
    model_dir=dann_dir,
    test_texts=test_texts,
    test_labels=test_labels,
)

print("\n" + "="*60)
print("DANN RESULTS")
print("="*60)
print(f"  Target Domain Accuracy : {results['accuracy']:.4f}")
print(f"  Target Domain F1       : {results['f1_macro']:.4f}")
print("\n  Compare with:")
print("  Baseline DistilBERT F1 : ~0.81")
print("  DAPT-DistilBERT F1     : ~0.93")
print(f"  DANN F1                : {results['f1_macro']:.4f}")
print("\n  DANN uses NO target labels — only domain alignment!")
print("="*60)
