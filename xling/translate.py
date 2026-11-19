"""IndicTrans2 forward translation (en -> hi/bn/ta/te) with a back-translation filter,
and Indic -> English translate-test (baseline B1).

  # fill hi-mt, bn-mt, ... (monolingual MT from the English source); failing items lose that condition
  python -m xling.translate forward --items runs/data/vqarad.jsonl
  python -m xling.translate forward --items runs/data/arogyabodha.jsonl --split test
  # ArogyaBodha translate-test: fill hi-tt, ... from the pure Indic text
  python -m xling.translate backward --items runs/data/arogyabodha.jsonl --from-variant pure --split test

Needs `pip install IndicTransToolkit sentence-transformers` and accepted terms for the
ai4bharat/indictrans2-* models on Hugging Face.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import IT2_TAG, TARGET_LANGS, condition_name, read_jsonl, write_jsonl

EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"


class Translator:
    def __init__(self, model_name: str, device: str = "cuda"):
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()
        self.ip = IndicProcessor(inference=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, sents: list[str], src: str, tgt: str, batch_size: int = 32) -> list[str]:
        out: list[str] = []
        for i in range(0, len(sents), batch_size):
            chunk = sents[i : i + batch_size]
            batch = self.ip.preprocess_batch(chunk, src_lang=IT2_TAG[src], tgt_lang=IT2_TAG[tgt])
            enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
            gen = self.model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=5)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang=IT2_TAG[tgt]))
        return out


def _segments(item: dict, cond: str) -> list[str]:
    t = item["text"][cond]
    return [t["question"]] + [t["options"][L] for L in item["letters"]]


def _assemble(item: dict, segs: list[str]) -> dict:
    return {"question": segs[0], "options": dict(zip(item["letters"], segs[1:]))}


def forward(items: list[dict], threshold: float, glossary: list[str]) -> list[dict]:
    """Translate question + options to each target language. The back-translated *question* must reach
    LaBSE similarity >= threshold and keep every glossary term; otherwise that item's <lang>-mt
    condition is left out (the item stays for the other conditions)."""
    from sentence_transformers import SentenceTransformer, util

    from .data import YES_NO

    fwd = Translator(EN_INDIC)
    bwd = Translator(INDIC_EN)
    labse = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    questions = [it["text"]["en"]["question"] for it in items]
    q_emb = labse.encode(questions, convert_to_tensor=True)
    for lang in TARGET_LANGS:
        flat, index = [], []
        for i, it in enumerate(items):
            # yes/no options use fixed translations; everything else goes through MT
            segs = _segments(it, "en") if not it.get("yes_no_order") else [questions[i]]
            index.append((i, len(flat), len(segs)))
            flat.extend(segs)
        out = fwd(flat, "en", lang)
        tq = [out[start] for _, start, _ in index]
        back = bwd(tq, lang, "en")
        sim = util.cos_sim(q_emb, labse.encode(back, convert_to_tensor=True)).diagonal()
        n_ok = 0
        for (i, start, n), it in zip(index, items):
            lost = [g for g in glossary if g in questions[i].lower() and g not in back[i].lower()]
            ok = float(sim[i]) >= threshold and not lost
            it.setdefault("qc", {})[lang] = {"backtranslation": back[i], "labse": round(float(sim[i]), 4), "lost_terms": lost, "pass": ok}
            if not ok:
                continue
            n_ok += 1
            order = it.get("yes_no_order")
            if order:
                t = {"question": tq[i], "options": {L: YES_NO[lang][0 if o == "yes" else 1] for L, o in zip("AB", order)}}
            else:
                t = _assemble(it, out[start : start + n])
            it["text"][condition_name(lang, "mt")] = t
        print(f"{lang}: {n_ok}/{len(items)} items pass the back-translation filter (LaBSE >= {threshold})")
    return items


def backward(items: list[dict], from_variant: str) -> list[dict]:
    bwd = Translator(INDIC_EN)
    for lang in TARGET_LANGS:
        src = condition_name(lang, from_variant)
        flat, index = [], []
        for i, it in enumerate(items):
            if src not in it["text"]:  # e.g. failed the back-translation filter
                continue
            segs = _segments(it, src)
            index.append((i, len(flat), len(segs)))
            flat.extend(segs)
        out = bwd(flat, lang, "en")
        for i, start, n in index:
            items[i]["text"][condition_name(lang, "tt")] = _assemble(items[i], out[start : start + n])
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", choices=["forward", "backward"])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: overwrite --items")
    ap.add_argument("--split", default=None, help="only translate this split (others are kept unchanged)")
    ap.add_argument("--from-variant", default="pure")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--glossary", type=Path, help="one clinical term per line; items losing one are dropped")
    args = ap.parse_args()

    items = read_jsonl(args.items)
    sel = [it for it in items if args.split is None or it["split"] == args.split]
    rest = [it for it in items if not (args.split is None or it["split"] == args.split)]
    if args.direction == "forward":
        glossary = [g.strip().lower() for g in args.glossary.read_text().splitlines() if g.strip()] if args.glossary else []
        sel = forward(sel, args.threshold, glossary)
    else:
        sel = backward(sel, args.from_variant)
    write_jsonl(args.out or args.items, rest + sel)


if __name__ == "__main__":
    main()
