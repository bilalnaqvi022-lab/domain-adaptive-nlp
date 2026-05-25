"""
aspect_sentiment.py
-------------------
Aspect-Level Sentiment Analysis (ABSA)

Standard sentiment: "This product is great" → Positive
Aspect-level:       "Battery is poor but camera is excellent"
                    → battery: Negative
                    → camera:  Positive

Supports ALL domains:
    Electronics, Sports, Clothing, Books, Food, Medical,
    Furniture, Restaurant, Travel, Beauty — koi bhi!
"""

from __future__ import annotations
from multiprocessing.util import info
from unittest import result

import torch # type: ignore
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification # pyright: ignore[reportMissingImports]
from models.text_classifier import ASPECT_CATEGORIES # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT KEYWORDS — Sab domains cover kiye hain
# Naya domain add karna ho toh bas us category mein words daalo
# ══════════════════════════════════════════════════════════════════════════════
ASPECT_KEYWORDS = {

    "Performance": [
        # Electronics
        "battery", "speed", "performance", "fast", "slow", "power",
        "drains", "charging", "efficient", "lag", "response",
        "processor", "cpu", "gpu", "heating", "overheating",
        "connectivity", "wifi", "bluetooth", "signal", "throttling",
        # Sports
        "tension", "flex", "spin", "aerodynamics", "traction",
        # General
        "weak", "strong", "powerful", "effective", "ineffective",
        "disappointing", "impressive", "outstanding", "exceptional",
        # Food / Medical
        "taste", "flavor", "fresh", "spicy", "sweet", "sour",
        "delicious", "tasty", "bland", "stale", "expired",
        "effective", "relief", "healing", "potent", "dosage",
        # Travel / Service
        "service", "staff", "response", "support", "helpful",
    ],

    "Quality": [
        # General
        "quality", "durable", "durability", "sturdy", "solid",
        "built", "finish", "premium", "flimsy", "cheap",
        # Electronics
        "resolution", "camera", "noise", "display", "screen",
        # Clothing / Fashion
        "material", "fabric", "cotton", "leather", "stitching",
        "stitch", "seam", "thread", "fitting", "texture",
        # Books / Content
        "writing", "content", "plot", "story", "ending",
        "style", "craft", "workmanship", "editing", "grammar",
        # Food
        "ingredients", "smell", "aroma", "freshness", "hygiene",
        # Medical
        "authentic", "genuine", "original", "counterfeit",
        # Furniture / Home
        "wood", "metal", "plastic", "glass", "finish", "polish",
        # General negative/positive quality words
        "poor", "excellent", "terrible", "superb", "awful",
    ],

    "Design": [
        # General
        "design", "looks", "appearance", "color", "style",
        "compact", "slim", "thin", "size", "shape",
        # Electronics
        "display", "screen", "keyboard", "trackpad", "ports",
        "speakers", "webcam", "weight", "lightweight", "heavy",
        # Clothing
        "pattern", "print", "embroidery", "cut", "length",
        # Furniture / Home
        "aesthetic", "finish", "modern", "classic", "elegant",
        # Food / Restaurant
        "presentation", "plating", "portions", "serving",
        # Packaging
        "packaging", "box", "wrapper", "container", "sealed",
    ],

    "Usability": [
        # General
        "easy", "difficult", "comfortable", "uncomfortable",
        "convenient", "inconvenient", "user", "interface",
        # Electronics
        "grip", "ergonomic", "smooth", "navigation", "setup",
        "install", "interface", "touchpad", "handle",
        # Sports
        "strings", "grip", "hold", "swing", "balance",
        # Clothing
        "soft", "rough", "wearing", "wearable", "fitting",
        "tight", "loose", "stretchy",
        # Books
        "readable", "engaging", "boring", "confusing",
        # Medical
        "dosage", "swallow", "tablet", "capsule",
        # Furniture
        "assembly", "setup", "instructions", "comfortable",
        # Food
        "portion", "filling", "heavy", "light",
    ],

    "Cost": [
        # General
        "price", "cost", "expensive", "cheap", "affordable",
        "worth", "value", "money", "budget", "overpriced",
        "reasonable", "economical", "costly", "bargain",
        # E-commerce
        "discount", "deal", "offer", "refund", "return",
    ],

    "Logistics": [
        # Delivery
        "delivery", "shipping", "arrived", "late", "early",
        "days", "weeks", "delayed", "fast", "slow",
        # Packaging
        "packaging", "package", "box", "damage", "damaged",
        "broken", "sealed", "wrapped", "condition",
        # Customer service
        "seller", "support", "response", "refund", "return",
        "exchange", "replacement",
    ],

    "Health_Safety": [
        # Medical / Food
        "effects", "side", "reaction", "allergy", "safe",
        "unsafe", "toxic", "harmful", "beneficial", "healthy",
        "unhealthy", "calories", "nutrition", "organic",
        "chemical", "natural", "artificial", "preservatives",
    ],

    "Experience": [
        # Restaurant / Travel / Service
        "experience", "atmosphere", "ambiance", "environment",
        "clean", "dirty", "noisy", "quiet", "crowded",
        "staff", "service", "waiter", "rude", "polite",
        "friendly", "professional", "helpful", "welcoming",
        # Hotel / Travel
        "room", "bed", "bathroom", "location", "view",
        "checkin", "checkout", "wifi", "breakfast",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT FILTER — Ye words aspects nahi hain
# ══════════════════════════════════════════════════════════════════════════════
PRODUCT_FILTER = {
    # Generic product words
    "product", "item", "thing", "stuff", "brand", "model",
    "amazon", "seller", "review", "purchase", "bought",
    # Electronics brands
    "samsung", "apple", "sony", "dell", "lenovo", "asus",
    "iphone", "ipad", "macbook", "windows", "android",
    # Product names
    "laptop", "phone", "mobile", "tablet", "computer", "device",
    "dress", "shirt", "pants", "jacket", "shoes",
    "book", "novel", "movie", "film",
    "food", "meal", "dish", "restaurant",
    "medicine", "drug", "pill",
}


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT WORDS
# ══════════════════════════════════════════════════════════════════════════════
POSITIVE_WORDS = {
    # General
    "excellent", "great", "good", "perfect", "amazing", "wonderful",
    "fantastic", "superb", "outstanding", "solid", "smooth",
    "efficient", "fast", "durable", "best", "nice", "well",
    "easy", "lightweight", "sturdy", "reliable", "strong", "sharp",
    # Food specific
    "delicious", "tasty", "fresh", "flavorful", "yummy", "scrumptious",
    # Service specific
    "helpful", "friendly", "polite", "professional", "welcoming",
    # General positive
    "love", "comfortable", "effective", "impressive", "beautiful",
    "gorgeous", "stunning", "brilliant", "exceptional", "remarkable",
    "satisfying", "pleased", "happy", "recommend", "worth",
}

NEGATIVE_WORDS = {
    # General
    "poor", "bad", "terrible", "awful", "horrible", "worst",
    "slow", "heavy", "broken", "weak", "difficult", "awkward",
    "drains", "fails", "disappointing", "uncomfortable", "flimsy",
    "loose", "blurry", "noisy", "overpriced",
    # Food specific
    "stale", "bland", "tasteless", "disgusting", "expired", "rotten",
    # Service specific
    "rude", "unhelpful", "unprofessional", "slow",
    # General negative
    "damaged", "defective", "useless", "waste", "regret",
    "dirty", "unhygienic", "toxic", "harmful", "fake", "counterfeit",
    "misleading", "overrated", "frustrated", "annoying",
}

NEGATION_WORDS = {
    "not", "no", "never", "isn't", "doesn't", "won't",
    "barely", "hardly", "without", "neither", "nor",
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _get_sentence_sentiment(sentence: str):
    """
    Rule-based sentence sentiment with negation handling.
    Also returns strong sentiment cue words.
    """

    words = sentence.lower().split()

    pos_count = 0
    neg_count = 0

    positive_hits = []
    negative_hits = []

    for i, word in enumerate(words):

        clean = re.sub(r'[^a-z]', '', word)

        negated = any(
            re.sub(r'[^a-z]', '', words[j]) in NEGATION_WORDS
            for j in range(max(0, i - 3), i)
        )

        # POSITIVE WORD
        if clean in POSITIVE_WORDS:

            if negated:
                neg_count += 1
                negative_hits.append(f"not {clean}")
            else:
                pos_count += 1
                positive_hits.append(clean)

        # NEGATIVE WORD
        elif clean in NEGATIVE_WORDS:

            if negated:
                pos_count += 1
                positive_hits.append(f"not {clean}")
            else:
                neg_count += 1
                negative_hits.append(clean)

    # FINAL SENTIMENT
    if pos_count > neg_count:
        sentiment = "Positive ✅"

    elif neg_count > pos_count:
        sentiment = "Negative ❌"

    else:
        sentiment = "Neutral 😐"

    return sentiment, positive_hits, negative_hits


def extract_aspect_sentiments(
    text: str,
    model_dir: str = None,
) -> dict:
    """
    Extract aspect-level sentiments from ANY domain review.
    No training needed for new domains — just keywords above.
    """
    sentences = re.split(r'[.!?;,]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    aspect_sentiments = {}

    for sentence in sentences:
        sent_lower     = sentence.lower()
        sent_sentiment, pos_hits, neg_hits = _get_sentence_sentiment(sentence)

        for category, keywords in ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in sent_lower and keyword not in PRODUCT_FILTER:
                    if keyword not in aspect_sentiments:
                        aspect_sentiments[keyword] = {
                            "sentiment": sent_sentiment,
                            "category":  category,
                            "evidence":  sentence[:80],
                            "positive_words": pos_hits,
                            "negative_words": neg_hits,
                        }

    # Overall sentiment
    overall = "Neutral 😐"
    if model_dir:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.eval()
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
                pred  = torch.argmax(probs).item()
            overall = f"{'Positive ✅' if pred == 1 else 'Negative ❌'} ({round(probs[pred].item()*100, 1)}%)"
        except:
            overall = _get_sentence_sentiment(text)
    else:
        overall = _get_sentence_sentiment(text)

    # Category summary
    category_summary = {}
    for aspect, info in aspect_sentiments.items():
        cat = info["category"]
        if cat not in category_summary:
            category_summary[cat] = {"positive": 0, "negative": 0, "aspects": []}
        category_summary[cat]["aspects"].append(aspect)
        if "Positive" in info["sentiment"]:
            category_summary[cat]["positive"] += 1
        elif "Negative" in info["sentiment"]:
            category_summary[cat]["negative"] += 1

    category_verdicts = {}
    for cat, counts in category_summary.items():
        if counts["positive"] > counts["negative"]:
            verdict = "Positive ✅"
        elif counts["negative"] > counts["positive"]:
            verdict = "Negative ❌"
        else:
            verdict = "Mixed 🔄"
        category_verdicts[cat] = {
            "verdict": verdict,
            "aspects": counts["aspects"],
        }

    return {
        "text":               text[:120] + "..." if len(text) > 120 else text,
        "overall_sentiment":  overall,
        "aspect_sentiments":  aspect_sentiments,
        "category_verdicts":  category_verdicts,
    }


def print_aspect_report(result: dict):
    """Pretty print aspect-level sentiment report."""
    print(f"\n  {'─'*55}")
    print(f"  Review   : {result['text']}")
    print(f"  Overall  : {result['overall_sentiment']}")
    print(f"  {'─'*55}")

    if not result["aspect_sentiments"]:
        print("  No specific aspects detected.")
        return

    print(f"  {'Aspect':<15} {'Category':<15} {'Sentiment':<15} Evidence")
    print(f"  {'─'*55}")
    for aspect, info in result["aspect_sentiments"].items():

        evidence = info["evidence"][:40] + "..."

    pos_words = (
        ", ".join(info["positive_words"])
        if info["positive_words"]
        else "-"
    )

    neg_words = (
        ", ".join(info["negative_words"])
        if info["negative_words"]
        else "-"
    )

    print(
        f"  {aspect:<15} "
        f"{info['category']:<15} "
        f"{info['sentiment']:<15} "
        f"\"{evidence}\""
    )

    print(f"      ↳ Positive cues : {pos_words}")
    print(f"      ↳ Negative cues : {neg_words}")

    if result["category_verdicts"]:
        print(f"\n  Category Summary:")
        for cat, v in result["category_verdicts"].items():

            aspects_str = ", ".join(v["aspects"])

            print(f"\n    {cat:<15} → {v['verdict']}")
            print(f"      Aspects : {aspects_str}")
    print(f"  {'─'*55}\n")
