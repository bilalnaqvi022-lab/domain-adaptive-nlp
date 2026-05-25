"""
evaluate.py
-----------
Cross-domain evaluation — 3 models, 5 domains.
Produces thesis result tables + plots.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.dataset_loader import dataset_to_pandas, load_amazon_domain


def evaluate_on_domain(model_dir, domain, split="test", test_samples=500, batch_size=32):
    print(f"  Evaluating [{Path(model_dir).name}] on {domain}...")
    dataset = load_amazon_domain(domain, split=split, limit=test_samples)
    df      = dataset_to_pandas(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()

    all_preds, texts = [], df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i+batch_size], truncation=True,
            padding=True, max_length=256, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            all_preds.extend(torch.argmax(model(**enc).logits, dim=-1).cpu().tolist())

    labels = df["label"].tolist()
    return {
        "domain":      domain,
        "accuracy":    accuracy_score(labels, all_preds),
        "f1_macro":    f1_score(labels, all_preds, average="macro"),
        "predictions": all_preds,
        "labels":      labels,
        "texts":       texts,
        "report":      classification_report(labels, all_preds, output_dict=True),
    }


def evaluate_domain_shift_gap(
    baseline_dir="outputs/baseline_model",
    dapt_dir="outputs/dapt_finetuned",
    roberta_dir=None,
    source_domain="Electronics",
    target_domain="Sports_and_Outdoors",
    test_samples=500,
    output_dir="outputs/evaluation",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("CROSS-DOMAIN EVALUATION")
    print(f"  {source_domain} → {target_domain}")
    print("="*60 + "\n")

    # Build model list
    models_to_eval = [
        ("DistilBERT Baseline", baseline_dir),
        ("DAPT-DistilBERT (ours)", dapt_dir),
    ]
    if roberta_dir and Path(roberta_dir).exists():
        models_to_eval.append(("RoBERTa-Base Vanilla", roberta_dir))

    results, rows = {}, []
    for name, mdir in models_to_eval:
        results[name] = {
            "source": evaluate_on_domain(mdir, source_domain, test_samples=test_samples),
            "target": evaluate_on_domain(mdir, target_domain, test_samples=test_samples),
        }
        src_f1 = results[name]["source"]["f1_macro"]
        tgt_f1 = results[name]["target"]["f1_macro"]
        rows.append({
            "Model":            name,
            "Source F1":        round(src_f1, 4),
            "Target F1":        round(tgt_f1, 4),
            "Target Acc":       round(results[name]["target"]["accuracy"], 4),
            "Domain Shift Gap": round(tgt_f1 - src_f1, 4),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path / "domain_shift_results.csv", index=False)

    print("\n── THESIS RESULTS TABLE ────────────────────────────────")
    print(summary_df.to_string(index=False))
    print("─"*56)

    _plot_comparison(summary_df, output_path)
    _plot_confusion_matrices(results, output_path)

    print(f"\n[Evaluation] All outputs → {output_dir}/")
    return {"summary": rows, "output_dir": str(output_path)}


def evaluate_multi_domain(
    baseline_dir, dapt_dir,
    source_domain="Electronics",
    target_domains=None,
    test_samples=200,
    output_dir="outputs/evaluation_multi",
) -> dict:
    """Evaluate across 5 domains — proves generalization."""
    if target_domains is None:
        target_domains = [
            "Sports_and_Outdoors",
            "Clothing_Shoes_and_Jewelry",
            "Home_and_Kitchen",
            "Toys_and_Games",
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("MULTI-DOMAIN GENERALIZATION RESULTS")
    print("="*60)

    rows = []
    for target in target_domains:
        print(f"\n  Testing on: {target}")
        try:
            b = evaluate_on_domain(baseline_dir, target, test_samples=test_samples)
            d = evaluate_on_domain(dapt_dir,     target, test_samples=test_samples)
            rows.append({
                "Target Domain":  target.replace("_", " "),
                "Baseline F1":    round(b["f1_macro"], 4),
                "DAPT F1":        round(d["f1_macro"], 4),
                "Improvement":    round(d["f1_macro"] - b["f1_macro"], 4),
                "Baseline Acc":   round(b["accuracy"], 4),
                "DAPT Acc":       round(d["accuracy"], 4),
            })
        except Exception as e:
            print(f"  [!] Skipping {target}: {e}")

    if not rows:
        return {"summary": [], "output_dir": str(output_path)}

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "multi_domain_results.csv", index=False)

    print("\n── MULTI-DOMAIN RESULTS ─────────────────────────────────")
    print(df.to_string(index=False))

    _plot_multi_domain(df, output_path)
    return {"summary": rows, "output_dir": str(output_path)}


def _plot_comparison(summary_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = summary_df["Model"].tolist()
    colors = ["#4A9F7E", "#E07B6A", "#7B68EE"][:len(models)]
    x, w   = np.arange(len(models)), 0.35

    axes[0].bar(x - w/2, summary_df["Source F1"], w, label="Source", color="#4A9F7E", alpha=0.85)
    axes[0].bar(x + w/2, summary_df["Target F1"], w, label="Target", color="#E07B6A", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, fontsize=8, rotation=10)
    axes[0].set_ylabel("Macro F1"); axes[0].set_title("Model Comparison: Source vs Target F1")
    axes[0].legend(); axes[0].set_ylim(0, 1.1)
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{bar.get_height():.3f}", ha="center", fontsize=7)

    gaps = summary_df["Domain Shift Gap"].tolist()
    bar_colors = ["#27AE60" if g >= 0 else "#C0392B" for g in gaps]
    axes[1].bar(models, gaps, color=bar_colors, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticklabels(models, fontsize=8, rotation=10)
    axes[1].set_ylabel("F1 Gap (Target − Source)")
    axes[1].set_title("Domain Shift Gap per Model")
    for i, g in enumerate(gaps):
        axes[1].text(i, g + 0.005 if g >= 0 else g - 0.015,
                     f"{g:+.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] model_comparison.png")


def _plot_confusion_matrices(results, output_path):
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))
    if n_models == 1:
        axes = [axes]

    for row, (model_name, res) in enumerate(results.items()):
        for col, domain_key in enumerate(["source", "target"]):
            r  = res[domain_key]
            cm = confusion_matrix(r["labels"], r["predictions"])
            ax = axes[row][col]
            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            ax.set_title(f"{model_name}\n({domain_key})", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Neg", "Pos"], fontsize=8)
            ax.set_yticklabels(["Neg", "Pos"], fontsize=8)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Confusion Matrices", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] confusion_matrices.png")


def _plot_multi_domain(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["Baseline F1"], w, label="Baseline DistilBERT", color="#E07B6A", alpha=0.85)
    ax.bar(x + w/2, df["DAPT F1"],     w, label="DAPT-DistilBERT (ours)", color="#4A9F7E", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Target Domain"], rotation=15, fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_title("Generalization Across 5 E-Commerce Domains\n(Trained on Electronics, Tested on All)")
    ax.legend(); ax.set_ylim(0, 1.1)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / "multi_domain_generalization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] multi_domain_generalization.png")
