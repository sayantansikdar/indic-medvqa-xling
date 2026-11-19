"""Build the parallel evaluation/training sets.

Every item is stored once, with its question text under several *conditions*:
  "en"        English
  "hi-mixed"  Hindi as released by ArogyaBodha (English terms kept in parentheses)
  "hi-pure"   the same Hindi with the parenthesised English glosses removed
  "hi-mt"     Hindi from IndicTrans2 (VQA-RAD), filled in by xling.translate
  "hi-tt"     Hindi question translated back to English (translate-test baseline)

Usage (Colab):
  python -m xling.data arogyabodha --out runs/data
  python -m xling.data vqarad --out runs/data
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import AROGYA_LANG, LANGS, TARGET_LANGS, condition_name, write_jsonl

LETTERS = "ABCDE"

# A parenthesised or bracketed run that contains Latin letters and no Indic-script characters,
# e.g. "சப்யூடேனியஸ் எம்ஃபைசீமா (subcutaneous emphysema)" -> "சப்யூடேனியஸ் எம்ஃபைசீமா".
# Range U+0900-U+0DFF covers every Indic block from Devanagari to Sinhala.
# Caveat: sometimes the English term is the only mention of a concept ("நோயரின் (thorax) சி.டி."),
# so stripping can delete content. "pure" is therefore an exploratory ablation; the monolingual
# condition for the main analysis is "mt" (IndicTrans2 from the English source, xling.translate).
_INDIC = "\\u0900-\\u0DFF"
_GLOSS = re.compile(
    rf"\s*(?:\((?=[^()]*[A-Za-z])[^(){_INDIC}]*\)|\[(?=[^\[\]]*[A-Za-z])[^\[\]{_INDIC}]*\])"
)
_SPACES = re.compile(r"[ \t]{2,}")


def strip_gloss(text: str) -> str:
    """Remove parenthesised English glosses from Indic text; keep the text if nothing would remain."""
    out = _SPACES.sub(" ", _GLOSS.sub("", text)).strip()
    return out if out else text


def split_of(image_hash: str, seed: int, fractions: dict[str, float]) -> str:
    """Deterministic split by image, so no image appears in two splits."""
    u = int(hashlib.sha1(f"{seed}:{image_hash}".encode()).hexdigest(), 16) % 10_000 / 10_000
    acc = 0.0
    for name, frac in fractions.items():
        acc += frac
        if u < acc:
            return name
    return list(fractions)[-1]


def _save_image(blob: dict, img_dir: Path) -> tuple[str, str]:
    from PIL import Image

    data = blob.get("bytes")
    if data is None:
        data = Path(blob["path"]).read_bytes()
    h = hashlib.sha1(data).hexdigest()[:20]
    fmt = (Image.open(io.BytesIO(data)).format or "png").lower()
    rel = f"images/{h}.{fmt}"
    dst = img_dir.parent / rel
    if not dst.exists():
        dst.write_bytes(data)
    return rel, h


def _clean_options(opts: dict) -> dict:
    return {k: (opts.get(k) or "").strip() for k in LETTERS if (opts.get(k) or "").strip()}


def build_arogyabodha(out: Path, seed: int, fractions: dict[str, float], single_image: bool = True) -> list[dict]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset("iit-patna-cse-ai/ArogyaBodha", split="train")
    ds = ds.filter(lambda lang: lang in AROGYA_LANG, input_columns="language")
    for f in LETTERS:
        ds = ds.cast_column(f"Figure_{f}", HFImage(decode=False))

    by_cid: dict[str, dict[str, dict]] = defaultdict(dict)
    meta_cols = ["cid", "language", "Question", "Options", "Correct_answer", "source", "domain_system", "difficulty_level"]
    meta = ds.select_columns(meta_cols + [f"Figure_{f}" for f in "BCDE"]).to_list()
    for i, r in enumerate(meta):
        n_fig_extra = sum(1 for f in "BCDE" if r[f"Figure_{f}"])
        by_cid[r["cid"]][AROGYA_LANG[r["language"]]] = {**{k: r[k] for k in meta_cols}, "row": i, "extra_figs": n_fig_extra}

    stats = Counter()
    keep: list[tuple[str, dict[str, dict]]] = []
    for cid, rows in by_cid.items():
        if set(rows) != set(LANGS):
            stats["missing_language"] += 1
            continue
        golds = {r["Correct_answer"].strip() for r in rows.values()}
        letters = {tuple(_clean_options(r["Options"])) for r in rows.values()}
        if len(golds) != 1 or len(letters) != 1:
            stats["inconsistent_across_languages"] += 1
            continue
        if single_image and any(r["extra_figs"] for r in rows.values()):
            stats["multi_image"] += 1
            continue
        if golds.pop() not in next(iter(letters)):
            stats["gold_not_in_options"] += 1
            continue
        keep.append((cid, rows))

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig_a = ds.select_columns(["Figure_A"])
    items = []
    for cid, rows in keep:
        en = rows["en"]
        blob = fig_a[en["row"]]["Figure_A"]
        if not blob:
            stats["no_image"] += 1
            continue
        rel, h = _save_image(blob, img_dir)
        text = {"en": {"question": en["Question"].strip(), "options": _clean_options(en["Options"])}}
        for lang in TARGET_LANGS:
            r = rows[lang]
            q, o = r["Question"].strip(), _clean_options(r["Options"])
            text[condition_name(lang, "mixed")] = {"question": q, "options": o}
            text[condition_name(lang, "pure")] = {
                "question": strip_gloss(q),
                "options": {k: strip_gloss(v) for k, v in o.items()},
            }
        items.append({
            "id": cid,
            "dataset": "arogyabodha",
            "image": rel,
            "image_hash": h,
            "source": en["source"],
            "domain": en["domain_system"],
            "difficulty": en["difficulty_level"],
            "gold": en["Correct_answer"].strip(),
            "letters": list(text["en"]["options"]),
            "split": split_of(h, seed, fractions),
            "text": text,
        })
        stats["kept"] += 1
    print("ArogyaBodha filter:", dict(stats))
    return items


YES_NO = {
    "en": ("Yes", "No"),
    "hi": ("हाँ", "नहीं"),
    "bn": ("হ্যাঁ", "না"),
    "ta": ("ஆம்", "இல்லை"),
    "te": ("అవును", "కాదు"),
}


def build_vqarad(out: Path, seed: int) -> list[dict]:
    """Closed-ended (yes/no) VQA-RAD test questions as two-option MCQ. Indic text is added by xling.translate."""
    from datasets import load_dataset

    ds = load_dataset("flaviagiammarino/vqa-rad", split="test")
    rng = random.Random(seed)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i, r in enumerate(ds):
        ans = r["answer"].strip().lower()
        if ans not in ("yes", "no"):
            continue
        buf = io.BytesIO()
        r["image"].convert("RGB").save(buf, format="PNG")
        rel, h = _save_image({"bytes": buf.getvalue()}, img_dir)
        yes_first = rng.random() < 0.5  # randomise letter order so "A" is not always "Yes"
        order = ("yes", "no") if yes_first else ("no", "yes")
        gold = "A" if order[0] == ans else "B"
        en_opts = {L: YES_NO["en"][0 if o == "yes" else 1] for L, o in zip("AB", order)}
        text = {"en": {"question": r["question"].strip(), "options": en_opts}}
        items.append({
            "id": f"vqarad~{i}",
            "dataset": "vqarad",
            "image": rel,
            "image_hash": h,
            "source": "vqa-rad",
            "domain": "radiology",
            "difficulty": None,
            "gold": gold,
            "letters": ["A", "B"],
            "split": "test",
            "text": text,
            "yes_no_order": list(order),
        })
    print(f"VQA-RAD closed-ended test items: {len(items)}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["arogyabodha", "vqarad"])
    ap.add_argument("--out", type=Path, default=Path("runs/data"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--test", type=float, default=0.5)
    ap.add_argument("--calib", type=float, default=0.2)
    ap.add_argument("--train", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "arogyabodha":
        items = build_arogyabodha(args.out, args.seed, {"test": args.test, "calib": args.calib, "train": args.train})
    else:
        items = build_vqarad(args.out, args.seed)
    path = args.out / f"{args.dataset}.jsonl"
    write_jsonl(path, items)
    print(f"wrote {len(items)} items -> {path}")
    print("splits:", dict(Counter(it["split"] for it in items)))
    print("sources:", dict(Counter(it["source"] for it in items).most_common()))


if __name__ == "__main__":
    main()
