"""Phase B5: cross-lingual activation patching at the final input position.

Modes (each produces one recovery value per layer or per window):
  resid      replace the whole residual stream after layer l. Cumulative: R -> 1 at the last layer
             by construction, so it is reported for reference only and NOT used to pick the band.
  attn, mlp  replace only layer l's additive attention / MLP contribution (non-cumulative).
  win{w}     replace attention+MLP contributions of layers [s, s+w) together. This is what a band
             adapter can change, and is the quantity B* is selected on (xling.band).
  ctl-win{w} same as win{w} but the donor is a *different* item's English run. Recovery here would
             mean the patch just makes the state "look English" rather than carrying item content.
  rev-*      reverse direction: target-language donors patched into the English run (necessity).

Normalised recovery for item i:  R = (p_patch(y) - p_tg(y)) / (p_en(y) - p_tg(y)), aggregated in
xling.band as a ratio of sums over items, which is stable when individual denominators are small.

  python -m xling.patching --items runs/data/arogyabodha.jsonl --split calib --conds hi-mt,bn-mt,ta-mt,te-mt \
      --modes resid,attn,mlp,win4,win6,win8,ctl-win6 --max-items 150
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .common import RUNS_DIR, read_jsonl
from .model import capture_final, load, load_image, patch_final


def gold_prob(vlm, logits, letters, gold):
    return torch.softmax(vlm.letter_logits(logits, letters), -1)[..., letters.index(gold)]


def _plans(mode: str, L: int, donors: dict):
    """Yield (row_labels, plan) where plan maps (site, layer) -> {row: vector}."""
    base = mode.removeprefix("ctl-").removeprefix("rev-")
    if base in ("resid", "attn", "mlp"):
        rows = list(range(L))
        return rows, {(base, l): {l: donors[base][l][0]} for l in rows}
    if base.startswith("win"):
        w = int(base[3:])
        starts = list(range(L - w + 1))
        plan: dict = {}
        for r, s in enumerate(starts):
            for l in range(s, s + w):
                for site in ("attn", "mlp"):
                    plan.setdefault((site, l), {})[r] = donors[site][l][0]
        return starts, plan
    raise ValueError(mode)


def run_mode(vlm, inputs, mode, donors, letters, gold, batch):
    rows, plan = _plans(mode, vlm.n_layers, donors)
    out = []
    for lo in range(0, len(rows), batch):
        hi = min(lo + batch, len(rows))
        sub = {k: {r - lo: v for r, v in m.items() if lo <= r < hi} for k, m in plan.items()}
        sub = {k: m for k, m in sub.items() if m}
        with torch.no_grad(), patch_final(vlm, sub):
            logits = vlm.model(**vlm.repeat(inputs, hi - lo), use_cache=False).logits[:, -1]
        out.extend(gold_prob(vlm, logits, letters, gold).float().cpu().tolist())
    return rows, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--split", default="calib")
    ap.add_argument("--conds", required=True)
    ap.add_argument("--modes", default="resid,attn,mlp,win4,win6,win8,ctl-win6")
    ap.add_argument("--preds", type=Path, help="base infer run dir (default runs/<model>/<dataset>/<split>/base)")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--exclude-sources", default="", help="comma-separated item sources to leave out, e.g. MedXpertQA_MM_test")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data_dir = args.data_dir or args.items.parent
    items = {it["id"]: it for it in read_jsonl(args.items) if it["split"] == args.split}
    dataset = next(iter(items.values()))["dataset"]
    pred_dir = args.preds or RUNS_DIR / args.model / dataset / args.split / "base"
    out_dir = args.out or RUNS_DIR / args.model / dataset / args.split / "patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = args.modes.split(",")
    vlm = load(args.model, attn="sdpa")
    en_pred = {p["id"]: p for p in read_jsonl(pred_dir / "preds_en.jsonl")}

    for cond in args.conds.split(","):
        tg_pred = {p["id"]: p for p in read_jsonl(pred_dir / f"preds_{cond}.jsonl")}
        # Items the model gets right in English and wrong in the target language
        excluded = set(filter(None, args.exclude_sources.split(",")))
        ids = [i for i in tg_pred if en_pred[i]["pred"] == en_pred[i]["gold"] and tg_pred[i]["pred"] != tg_pred[i]["gold"]
               and items[i].get("source") not in excluded]
        random.Random(args.seed).shuffle(ids)
        ids = ids[: args.max_items]
        print(f"{cond}: patching {len(ids)} items (EN correct, {cond} wrong)")
        path = out_dir / f"patch_{cond}.jsonl"
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        prev_en_donors = None
        with open(path, "a", encoding="utf-8") as f:
            for iid in tqdm(ids, desc=cond):
                it = items[iid]
                letters, gold = it["letters"], it["gold"]
                image = load_image(data_dir, it["image"])
                build = lambda c: vlm.build(image, vlm.prompt_text(it["text"][c]["question"], it["text"][c]["options"], letters))
                en_in, tg_in = build("en"), build(cond)
                with torch.no_grad(), capture_final(vlm, ["resid", "attn", "mlp"]) as en_d:
                    p_en = float(gold_prob(vlm, vlm.model(**en_in, use_cache=False).logits[0, -1], letters, gold))
                with torch.no_grad(), capture_final(vlm, ["resid", "attn", "mlp"]) as tg_d:
                    p_tg = float(gold_prob(vlm, vlm.model(**tg_in, use_cache=False).logits[0, -1], letters, gold))
                if iid in done:
                    prev_en_donors = en_d
                    continue
                rec = {"id": iid, "cond": cond, "source": it.get("source"), "p_en": p_en, "p_tg": p_tg, "modes": {}}
                for mode in modes:
                    if mode.startswith("ctl-"):
                        if prev_en_donors is None:
                            continue
                        rows, ps = run_mode(vlm, tg_in, mode, prev_en_donors, letters, gold, args.batch)
                    elif mode.startswith("rev-"):
                        rows, ps = run_mode(vlm, en_in, mode, tg_d, letters, gold, args.batch)
                    else:
                        rows, ps = run_mode(vlm, tg_in, mode, en_d, letters, gold, args.batch)
                    rec["modes"][mode] = [round(p, 6) for p in ps]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                prev_en_donors = en_d


if __name__ == "__main__":
    main()
