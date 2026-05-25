"""
main.py
-------
Full thesis experiment runner.

Modes:
    python main.py --mode test     # 5 min smoke test
    python main.py --mode full     # full thesis experiment (4-6 hrs)
    python main.py --mode multi    # multi-domain generalization
"""

import argparse
from evaluation.evaluate import evaluate_domain_shift_gap, evaluate_multi_domain
from training.train import train_baseline, train_roberta, train_with_dapt, run_multi_domain_experiment

SOURCE_DOMAIN  = "Electronics"
TARGET_DOMAIN  = "Sports_and_Outdoors"
TARGET_DOMAINS = ["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry",
                  "Home_and_Kitchen", "Toys_and_Games"]

TRAIN_SAMPLES  = 2000
EVAL_SAMPLES   = 400
DAPT_SAMPLES   = 5000
TEST_SAMPLES   = 500


def run_full():
    print("\n" + "█"*60)
    print("  MS THESIS — FULL EXPERIMENT")
    print("  Domain-Adaptive DistilBERT for E-Commerce NLP")
    print("█"*60)

    # 1. Baseline DistilBERT
    baseline_dir = train_baseline(
        source_domain=SOURCE_DOMAIN, train_samples=TRAIN_SAMPLES,
        eval_samples=EVAL_SAMPLES, output_dir="outputs/baseline_model", epochs=3,
    )

    # 2. RoBERTa Vanilla (comparison)
    roberta_dir = train_roberta(
        source_domain=SOURCE_DOMAIN, train_samples=TRAIN_SAMPLES,
        eval_samples=EVAL_SAMPLES, output_dir="outputs/roberta_model", epochs=3,
    )

    # 3. DAPT DistilBERT (our solution)
    dapt_dir = train_with_dapt(
        source_domain=SOURCE_DOMAIN, target_domain=TARGET_DOMAIN,
        train_samples=TRAIN_SAMPLES, eval_samples=EVAL_SAMPLES,
        dapt_samples=DAPT_SAMPLES, dapt_epochs=3, finetune_epochs=3,
        dapt_dir="outputs/dapt_pretrained", output_dir="outputs/dapt_finetuned",
    )

    # 4. Evaluate all 3 models
    metrics = evaluate_domain_shift_gap(
        baseline_dir=baseline_dir, dapt_dir=dapt_dir, roberta_dir=roberta_dir,
        source_domain=SOURCE_DOMAIN, target_domain=TARGET_DOMAIN,
        test_samples=TEST_SAMPLES, output_dir="outputs/evaluation",
    )

    print("\n" + "═"*60)
    print("  THESIS RESULTS")
    print("═"*60)
    for row in metrics["summary"]:
        print(f"\n  [{row['Model']}]")
        print(f"    Source F1 : {row['Source F1']:.4f}")
        print(f"    Target F1 : {row['Target F1']:.4f}")
        print(f"    Shift Gap : {row['Domain Shift Gap']:+.4f}")
    print(f"\n  Plots → outputs/evaluation/")
    print("═"*60)


def run_test():
    print("\n[QUICK TEST MODE] Minimal samples...\n")

    baseline_dir = train_baseline(
        source_domain=SOURCE_DOMAIN, train_samples=200, eval_samples=50,
        output_dir="outputs/test_baseline", epochs=1,
    )
    roberta_dir = train_roberta(
        source_domain=SOURCE_DOMAIN, train_samples=200, eval_samples=50,
        output_dir="outputs/test_roberta", epochs=1,
    )
    dapt_dir = train_with_dapt(
        source_domain=SOURCE_DOMAIN, target_domain=TARGET_DOMAIN,
        train_samples=200, eval_samples=50, dapt_samples=300,
        dapt_epochs=1, finetune_epochs=1,
        dapt_dir="outputs/test_dapt", output_dir="outputs/test_adapted",
    )
    metrics = evaluate_domain_shift_gap(
        baseline_dir=baseline_dir, dapt_dir=dapt_dir, roberta_dir=roberta_dir,
        source_domain=SOURCE_DOMAIN, target_domain=TARGET_DOMAIN,
        test_samples=100, output_dir="outputs/test_eval",
    )
    print("\n[QUICK TEST DONE]")
    for row in metrics["summary"]:
        print(f"  [{row['Model']}] Target F1: {row['Target F1']:.4f}")


def run_multi():
    print("\n[MULTI-DOMAIN MODE] Testing generalization across 5 domains...\n")

    # Use existing trained models
    import os
    baseline_dir = "outputs/baseline_model" if os.path.exists("outputs/baseline_model") else "outputs/test_baseline"
    dapt_dir     = "outputs/dapt_finetuned"  if os.path.exists("outputs/dapt_finetuned")  else "outputs/test_adapted"

    # Train DAPT for each target domain
    run_multi_domain_experiment(
        source_domain=SOURCE_DOMAIN,
        target_domains=TARGET_DOMAINS,
        train_samples=500, eval_samples=100,
        dapt_samples=500, epochs=1,
        output_base="outputs/multi_domain",
    )

    # Evaluate on all domains using main DAPT model
    evaluate_multi_domain(
        baseline_dir=baseline_dir,
        dapt_dir=dapt_dir,
        source_domain=SOURCE_DOMAIN,
        target_domains=TARGET_DOMAINS,
        test_samples=200,
        output_dir="outputs/evaluation_multi",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "test", "multi"], default="full")
    args = parser.parse_args()

    if args.mode == "test":
        run_test()
    elif args.mode == "multi":
        run_multi()
    else:
        run_full()
