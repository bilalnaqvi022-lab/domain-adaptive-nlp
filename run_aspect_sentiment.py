"""
run_aspect_sentiment.py
-----------------------
Aspect-Level Sentiment Analysis Demo

Shows:
    - Standard sentiment: whole review = positive/negative
    - Aspect sentiment:   each feature = its own sentiment

Usage: python run_aspect_sentiment.py
"""

from models.aspect_sentiment import extract_aspect_sentiments, print_aspect_report
import os

MODEL_DIR = "outputs/test_adapted"
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = None
    print("[!] No trained model found, using rule-based only")

print("\n" + "="*60)
print("ASPECT-LEVEL SENTIMENT ANALYSIS")
print("Standard vs Aspect-level comparison")
print("="*60)

test_reviews = [
    # Mixed review — standard model might say "positive" but aspects differ
    "The grip is perfect and very comfortable to use. "
    "However, the battery drains quickly and charging takes forever. "
    "Overall build quality is solid but price is too expensive.",

    # Sports equipment review
    "String tension is excellent and the flex is amazing for power shots. "
    "The grip feels a bit awkward initially but you get used to it. "
    "Durability seems poor after just two weeks of heavy use.",

    # Electronics review
    "Camera quality is outstanding and the display is crystal clear. "
    "Battery performance is terrible though, barely lasts 4 hours. "
    "The design is sleek and lightweight which I love.",

    # Mostly positive
    "Perfect grip and amazing build quality. "
    "Speed and performance exceed expectations. "
    "Very comfortable and easy to use. Great value for money.",
]

for i, review in enumerate(test_reviews, 1):
    print(f"\n{'='*60}")
    print(f"Review #{i}")
    result = extract_aspect_sentiments(review, model_dir=MODEL_DIR)
    print_aspect_report(result)

print("\n" + "="*60)
print("KEY INSIGHT:")
print("  Standard model gives ONE label for entire review.")
print("  Aspect-level gives SEPARATE sentiment per feature.")
print("  Example: battery=Negative, camera=Positive")
print("  This is much more useful for e-commerce businesses!")
print("="*60)
