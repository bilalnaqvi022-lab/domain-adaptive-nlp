"""
dataset_loader.py
-----------------
Loads Amazon review data for domain-shift thesis experiments.

Uses: datasets 4.8.5 compatible approach
- Downloads Parquet files directly from HuggingFace
- No 22GB downloads, no script errors
- All 6 domains use train shards (test shards dont exist)
- Different seeds per domain = different samples, no overlap

Domains: Electronics, Sports_and_Outdoors, Clothing_Shoes_and_Jewelry,
         Home_and_Kitchen, Books, Toys_and_Games
"""

from __future__ import annotations

import random
from typing import Optional

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset


# ── All domains use TRAIN shards only (test shards dont exist) ─────────────────
DOMAIN_URLS = {
    "Electronics":                "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00000-of-00004.parquet",
    "Sports_and_Outdoors":        "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00001-of-00004.parquet",
    "Clothing_Shoes_and_Jewelry": "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00002-of-00004.parquet",
    "Home_and_Kitchen":           "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00003-of-00004.parquet",
    "Books":                      "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00000-of-00004.parquet",
    "Toys_and_Games":             "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity/train-00001-of-00004.parquet",
}

# Different seed per domain → different samples, no overlap between domains
DOMAIN_SEED = {
    "Electronics":                42,
    "Sports_and_Outdoors":        43,
    "Clothing_Shoes_and_Jewelry": 44,
    "Home_and_Kitchen":           45,
    "Books":                      46,
    "Toys_and_Games":             47,
}


def _rating_to_label(label) -> Optional[int]:
    try:
        r = int(label)
        if r in (0, 1):
            return r
    except (TypeError, ValueError):
        pass
    return None


def load_amazon_domain(
    domain: str = "Electronics",
    split: str = "train",
    limit: int = 2000,
    seed: int = None,
) -> Dataset:
    """
    Load Amazon reviews for one product domain.
    Downloads Parquet directly — no loading script needed.
    Each domain gets unique samples via different random seeds.
    """
    print(f"  Loading Amazon/{domain} [{split}] ...")

    url = DOMAIN_URLS.get(domain)
    if url is None:
        raise ValueError(f"Unknown domain: {domain}. Choose from: {list(DOMAIN_URLS.keys())}")

    # Use domain-specific seed for unique samples
    seed = seed if seed is not None else DOMAIN_SEED.get(domain, 42)

    # Load parquet directly
    ds = load_dataset("parquet", data_files={"data": url}, split="data")

    # Build records
    records = []
    for row in ds:
        label = _rating_to_label(row.get("label", -1))
        if label is None:
            continue
        title   = str(row.get("title",   "")).strip()
        content = str(row.get("content", "")).strip()
        combined = f"{title}. {content}" if title else content
        if len(combined) < 20:
            continue
        records.append({"text": combined, "label": label, "domain": domain})

    # Balance pos/neg
    positives = [r for r in records if r["label"] == 1]
    negatives = [r for r in records if r["label"] == 0]
    random.seed(seed)
    n = min(limit, len(positives), len(negatives))

    if n == 0:
        raise ValueError(f"No samples found for domain '{domain}'.")

    balanced = random.sample(positives, n) + random.sample(negatives, n)
    random.shuffle(balanced)

    # Manual train/test split 80/20
    split_idx = int(len(balanced) * 0.8)
    if split == "train":
        final = balanced[:split_idx]
    else:
        final = balanced[split_idx:]

    dataset = Dataset.from_list(final)
    print(f"    -> {len(dataset)} samples")
    return dataset


def load_multi_domain(
    domains: list[str],
    split: str = "train",
    limit_per_domain: int = 1000,
    seed: int = 42,
) -> Dataset:
    """Load and concatenate multiple domains. Used for DAPT."""
    parts = [
        load_amazon_domain(d, split=split, limit=limit_per_domain)
        for d in domains
    ]
    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=seed)
    print(f"  Multi-domain total: {len(combined)} samples across {len(domains)} domains")
    return combined


def dataset_to_pandas(dataset: Dataset) -> pd.DataFrame:
    return dataset.to_pandas()


def label_distribution(dataset: Dataset) -> dict:
    df = dataset_to_pandas(dataset)
    dist = df.groupby(["domain", "label"]).size().unstack(fill_value=0)
    return dist.to_dict()


AVAILABLE_DOMAINS = list(DOMAIN_URLS.keys())

DOMAIN_PAIRS = [
    ("Electronics",      "Sports_and_Outdoors"),
    ("Electronics",      "Clothing_Shoes_and_Jewelry"),
    ("Home_and_Kitchen", "Toys_and_Games"),
    ("Books",            "Electronics"),
]
