"""Baseline B3 (MedVL-XLRepE-style): choose layer and scale for training-free steering on calib.

  python -m xling.steer --items runs/data/arogyabodha.jsonl --variant mt \
      --vectors runs/medgemma/arogyabodha/calib/base/steer_mt.npz --layers 8:28:2 --alphas 0.5,1,2 --limit 200
Then evaluate the chosen setting on test with xling.infer --steer-file/--steer-layer/--steer-alpha.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common import TARGET_LANGS, read_jsonl
from .infer import run_condition
from .model import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--variant", default="mt")
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--layers", default="8:28:2", help="start:stop:step")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    data_dir = args.data_dir or args.items.parent
    items = [it for it in read_jsonl(args.items) if it["split"] == "calib"][: args.limit]
    vecs = np.load(args.vectors)
    vlm = load(args.model)
    run_args = SimpleNamespace(capture=False, instruction="letter", steer_layer=None, steer_alpha=None)
    conds = [f"{l}-{args.variant}" for l in TARGET_LANGS]
    a, b, c = map(int, args.layers.split(":"))
    results = []
    for layer in range(a, b, c):
        for alpha in map(float, args.alphas.split(",")):
            run_args.steer_layer, run_args.steer_alpha = layer, alpha
            accs = {}
            for cond in conds:
                v = torch.tensor(vecs[cond][layer], device=vlm.device)
                preds = run_condition(vlm, items, cond, data_dir, run_args, v)[0]
                accs[cond] = float(np.mean([p["pred"] == p["gold"] for p in preds]))
            results.append({"layer": layer, "alpha": alpha, "accs": accs, "mean": float(np.mean(list(accs.values())))})
            print(f"layer {layer:2d} alpha {alpha:4.1f}: mean acc {results[-1]['mean']:.3f}  {accs}")
    best = max(results, key=lambda r: r["mean"])
    out = args.vectors.parent / f"steer_sweep_{args.variant}.json"
    out.write_text(json.dumps({"best": best, "all": results}, indent=1))
    print(f"best: layer {best['layer']} alpha {best['alpha']} -> {out}")


if __name__ == "__main__":
    main()
