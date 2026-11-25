"""Effect of where the adapter sits: pooled per-item accuracy change vs. the base run, by adapted-layer band.

  python -m xling.position --root runs/medgemma/vqarad/test --train-dir runs/train \
      --runs ours_c,ours_c_s1,random_band_c,random_band_c_s1,pos00-03,pos08-11,pos16-19,pos28-31,all_layers --plot

Layers come from each adapter's train_config.json; the diagnosed band B* is the run trained with --band-json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .report import compare, load_preds
from .stats import holm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="runs/<model>/<dataset>/<split>")
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--variant", default="mt")
    ap.add_argument("--exclude-sources", default="", help="comma-separated item sources to leave out")
    ap.add_argument("--n-layers", type=int, default=34)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    exclude = set(filter(None, args.exclude_sources.split(",")))
    suffix = "_excl-" + "-".join(sorted(exclude)) if exclude else ""

    base = load_preds(args.root / args.base, exclude)
    rows = []
    for tag in args.runs.split(","):
        cfg = json.loads((args.train_dir / tag / "train_config.json").read_text())
        layers = cfg["adapted_layers"]
        pooled = next(r for r in compare(load_preds(args.root / tag, exclude), base) if r["cond"] == f"{args.variant} (pooled)")
        rows.append({"run": tag, "start": min(layers), "end": max(layers), "width": len(layers), "seed": cfg["seed"],
                     "diagnosed": cfg["placement"] == "band" and bool(cfg.get("band_json")),
                     "delta": pooled["delta"], "delta_lo": pooled["delta_lo"], "delta_hi": pooled["delta_hi"],
                     "p_perm": pooled["p_perm"]})
    for tag, p in holm({r["run"]: r["p_perm"] for r in rows}).items():
        next(r for r in rows if r["run"] == tag)["p_holm"] = p
    rows.sort(key=lambda r: (r["width"], r["start"], r["seed"]))

    with open(args.root / f"position_{args.variant}{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.root.parent.name}{suffix}: {args.variant} (pooled) accuracy change vs {args.base}, by adapted layers")
    print(f"{'run':18s} {'layers':>8s} seed  {'delta':>7s}  95% CI              p_holm")
    for r in rows:
        star = "  <- B*" if r["diagnosed"] else ""
        print(f"{r['run']:18s} {r['start']:>3d}-{r['end']:<4d} {r['seed']:>4d}  {r['delta']:+.3f}  "
              f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]  {r['p_holm']:.3g}{star}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.axhline(0, color="grey", lw=0.8)
        for r in rows:
            if r["width"] >= args.n_layers:
                ax.axhspan(r["delta_lo"], r["delta_hi"], color="red", alpha=0.1)
                ax.axhline(r["delta"], color="red", ls="--", lw=1, label=f"all layers ({r['run']})")
                continue
            x = (r["start"] + r["end"]) / 2 + 0.25 * r["seed"]  # nudge seeds apart
            ax.errorbar(x, r["delta"], yerr=[[r["delta"] - r["delta_lo"]], [r["delta_hi"] - r["delta"]]],
                        xerr=(r["end"] - r["start"]) / 2, fmt="o", capsize=3,
                        color="tab:orange" if r["diagnosed"] else "tab:blue", mfc="white" if r["seed"] else None)
        ax.plot([], [], "o", color="tab:orange", label="diagnosed band B*")
        ax.plot([], [], "o", color="tab:blue", label="other bands")
        ax.plot([], [], "o", color="k", mfc="white", label="seed 1")
        title = args.root.parent.name + (f" (without {', '.join(sorted(exclude))})" if exclude else "")
        ax.set(xlim=(-1, args.n_layers), xlabel="adapted layers (bar = band extent)",
               ylabel=f"{args.variant} accuracy change vs {args.base}", title=title)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.root / f"position_{args.variant}{suffix}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
