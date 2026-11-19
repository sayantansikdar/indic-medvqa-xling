"""Shared constants and small helpers."""
from __future__ import annotations

import json
import os
import random
import unicodedata
from pathlib import Path

LANGS = ["en", "hi", "bn", "ta", "te"]
TARGET_LANGS = ["hi", "bn", "ta", "te"]
FAMILY = {"hi": "indo-aryan", "bn": "indo-aryan", "ta": "dravidian", "te": "dravidian"}

# ArogyaBodha `language` column -> ISO code
AROGYA_LANG = {"English": "en", "Hindi": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te"}

# IndicTrans2 language tags
IT2_TAG = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu"}

# Unicode blocks used to check that a reply is in the question's script
SCRIPT_RANGE = {
    "hi": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
}

# Variants of the question text:
#   mixed : as released (ArogyaBodha keeps English terms in parentheses)
#   pure  : parenthesised English glosses stripped
#   mt    : machine-translated by IndicTrans2 (VQA-RAD / SLAKE)
#   tt    : translate-test, target-language question translated back to English
VARIANTS = ["mixed", "pure", "mt", "tt"]

RUNS_DIR = Path(os.environ.get("XLING_RUNS", "runs"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def script_share(text: str, lang: str) -> float:
    """Fraction of alphabetic characters that belong to `lang`'s script (Latin for en)."""
    letters = [c for c in text if unicodedata.category(c).startswith("L") or unicodedata.category(c) == "Mc"]
    if not letters:
        return 0.0
    if lang == "en":
        hit = sum(1 for c in letters if c.isascii())
    else:
        lo, hi = SCRIPT_RANGE[lang]
        hit = sum(1 for c in letters if lo <= ord(c) <= hi)
    return hit / len(letters)


def condition_name(lang: str, variant: str) -> str:
    return "en" if lang == "en" else f"{lang}-{variant}"
