"""
resume.py — Updated for DistilBERT
Skips baseline (already done), runs DAPT + evaluation.
"""

from training.train import train_with_dapt
from evaluation.evaluate import evaluate_domain_shift_gap

dapt_dir = train_with_dapt(
    source_domain="Electronics",
    target_domain="Sports_and_Outdoors",
    train_samples=200,
    eval_samples=50,
    dapt_samples=300,
    dapt_epochs=1,
    finetune_epochs=1,
    dapt_dir="outputs/test_dapt",
    output_dir="outputs/test_adapted",
)

metrics = evaluate_domain_shift_gap(
    baseline_dir="outputs/test_baseline",
    dapt_dir=dapt_dir,
    source_domain="Electronics",
    target_domain="Sports_and_Outdoors",
    test_samples=100,
    output_dir="outputs/test_eval",
)

print("\n✅ DONE!")
for row in metrics["summary"]:
    print(f"\n  [{row['Model']}]")
    print(f"    Target F1 : {row['Target F1']:.4f}")
    print(f"    Shift Gap : {row['Domain Shift Gap']:+.4f}")
