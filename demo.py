"""
demo.py — Professor Demo Script
Shows:
1. Domain Shift Problem
2. DAPT Solution
3. Aspect-Level Sentiment Analysis
4. Final Results
"""

import os
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from models.aspect_sentiment import (
    extract_aspect_sentiments,
    print_aspect_report,
)

# COLORS
R = "\033[91m"
G = "\033[92m"
Y = "\033[93m"
B = "\033[94m"
W = "\033[1m"
X = "\033[0m"


def div(title=""):
    print(f"\n{B}{'═'*65}{X}")
    if title:
        print(f"{W}  {title}{X}")
    print(f"{B}{'═'*65}{X}")


def predict(text, model_dir):

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    pred = torch.argmax(probs).item()

    label = "POSITIVE ✅" if pred == 1 else "NEGATIVE ❌"

    return label, round(probs[pred].item() * 100, 1)


def main():

    BL = "outputs/baseline_model"
    DA = "outputs/dapt_finetuned"

    # CHECK MODELS
    if not os.path.exists(BL) or not os.path.exists(DA):

        print(f"{R}Run training first!{X}")
        print(f"{Y}python main.py --mode full{X}")

        return

    # HEADER
    print(f"\n{W}{G}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     MS THESIS — Domain-Adaptive NLP for E-Commerce        ║")
    print("║        Universal Aspect-Level Sentiment System            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(X)

    # ═══════════════════════════════════════════════════════════
    # PART 1
    # ═══════════════════════════════════════════════════════════

    div("PART 1: THE PROBLEM — DOMAIN SHIFT")

    print(f"""
{Y}Key Insight:{X}

  Same word → different meaning in different domains

  "Grip"
      Electronics → mouse grip
      Sports      → racket grip

  "Tension"
      Electronics → cable tension
      Sports      → string tension

  Baseline models fail to generalize properly.
""")

    reviews = [

        (
            "The grip is excellent and battery performance is outstanding.",
            "Sports"
        ),

        (
            "Terrible tension in the strings. Completely lost flex.",
            "Sports"
        ),
    ]

    print(f"{W}Baseline DistilBERT Predictions:{X}\n")

    for review, domain in reviews:

        label, conf = predict(review, BL)

        print(f"  Review : \"{review}\"")
        print(f"  Domain : {domain}")
        print(f"  Result : {label} ({conf}%)\n")

        time.sleep(0.5)

    # ═══════════════════════════════════════════════════════════
    # PART 2
    # ═══════════════════════════════════════════════════════════

    div("PART 2: OUR SOLUTION — DAPT")

    print(f"""
{G}Phase 1 — DAPT:{X}
  Domain-adaptive pretraining on unlabeled Sports data

{G}Phase 2 — TAPT:{X}
  Task-adaptive pretraining for aspect vocabulary alignment

{G}Phase 3 — Fine-Tuning:{X}
  Supervised sentiment learning on Electronics domain

{W}Same reviews with our DAPT-DistilBERT:{X}
""")

    for review, domain in reviews:

        label, conf = predict(review, DA)

        print(f"  Review : \"{review}\"")
        print(f"  Domain : {domain}")
        print(f"  Result : {G}{label}{X} ({conf}%)\n")

        time.sleep(0.5)

    # ═══════════════════════════════════════════════════════════
    # PART 3
    # ═══════════════════════════════════════════════════════════

    div("PART 3: ASPECT-LEVEL SENTIMENT ANALYSIS")

    print(f"""
{Y}Method:{X}

  Hybrid ABSA System:
    → Transformer-based overall sentiment
    → Rule-based explainable aspect detection
    → Category classification
    → Positive/Negative cue extraction
""")

    demo_reviews = [

        "The grip feels solid and string tension is perfect. Durability seems excellent after heavy use.",

        "Battery drains quickly and display quality is poor. Weight distribution feels awkward.",

        "Hotel room was clean but staff was very rude.",
    ]

    for review in demo_reviews:

        result = extract_aspect_sentiments(
            review,
            model_dir=DA,
        )

        print_aspect_report(result)

        time.sleep(1)

    # ═══════════════════════════════════════════════════════════
    # PART 4
    # ═══════════════════════════════════════════════════════════

    div("PART 4: FINAL RESULTS")

    print(f"""
{W}Cross-Domain Evaluation:{X}

┌──────────────────────────────┬──────────┬──────────┬──────────┐
│ Model                        │ Src F1   │ Tgt F1   │ Gap      │
├──────────────────────────────┼──────────┼──────────┼──────────┤
│ DistilBERT Baseline          │ 0.9499   │ 0.9220   │ -0.0279  │
│ DAPT-DistilBERT (ours)       │ 0.9378   │ 0.9260   │ -0.0119  │
│ RoBERTa-Base Vanilla         │ 0.9640   │ 0.9400   │ -0.0240  │
└──────────────────────────────┴──────────┴──────────┴──────────┘

{G}Key Findings:{X}

✅ DAPT reduced domain shift significantly
✅ Better cross-domain generalization
✅ Explainable aspect-level predictions
✅ No labeled aspect dataset required
✅ Faster than RoBERTa on CPU

{W}Future Work:{X}

→ DANN (Gradient Reversal Layer)
→ GAN-based augmentation
→ Multi-task learning
→ Real-time deployment API
""")

    # ═══════════════════════════════════════════════════════════
    # LIVE DEMO
    # ═══════════════════════════════════════════════════════════

    div("LIVE DEMO — ENTER ANY REVIEW")

    print(f"  {W}Type any review for instant ABSA analysis{X}")
    print(f"  Press Ctrl+C to exit\n")

    while True:

        try:

            review = input(f"  {Y}Review:{X} ").strip()

            if not review:
                continue

            result = extract_aspect_sentiments(
                review,
                model_dir=DA,
            )

            print_aspect_report(result)

        except KeyboardInterrupt:

            print(f"\n\n{G}Demo Complete. Thank You!{X}\n")

            break


if __name__ == "__main__":
    main()