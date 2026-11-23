"""Aggregate patching results and select the critical band B*.

Selection rule (fix this before looking at the test split):
  1. For each width w in {4, 6, 8}, take the window start s maximising window recovery R_win{w}(s)
     on the calib split, pooled over the four target languages.
  2. Choose the smallest w whose best recovery is >= 0.9 x the best recovery over all widths.
  3. Report per-family bands as a secondary analysis (H1: Indo-Aryan vs Dravidian).

  python -m xling.band --patch-dir runs/medgemma/arogyabodha/calib/patching
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import FAMILY, read_jsonl
from .stats import bootstrap_ratio_ci


def recovery(recs: list[dict], mode: str):
    rows = [r for r in recs if mode in r["modes"] and r["p_en"] - r["p_tg"] > 0]
    if not rows:
        return None
    p_en = np.array([r["p_en"] for r in rows])[:, None]
    p_tg = np.array([r["p_tg"] for r in rows])[:, None]
    p = np.array([r["modes"][mode] for r in rows])
    num = (p_en - p) if mode.startswith("rev-") else (p - p_tg)
    point, lo, hi = bootstrap_ratio_ci(num, np.broadcast_to(p_en - p_tg, num.shape))
    return {"R": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n": len(rows)}


def select_band(curves: dict, widths=(4, 6, 8), tol: float = 0.9):
    best = {}
    for w in widths:
        c = curves.get(f"win{w}")
        if c:
            s = int(np.argmax(c["R"]))
            best[w] = {"start": s, "width": w, "layers": list(range(s, s + w)), "R": c["R"][s], "lo": c["lo"][s], "hi": c["hi"][s]}
    if not best:
        return None
    top = max(b["R"] for b in best.values())
    for w in sorted(best):
        if best[w]["R"] >= tol * top:
            return best[w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    by_cond = {p.stem.removeprefix("patch_"): read_jsonl(p) for p in sorted(args.patch_dir.glob("patch_*.jsonl"))}
    groups = dict(by_cond)
    fam = defaultdict(list)
    for cond, recs in by_cond.items():
        fam[FAMILY[cond.split("-")[0]]].extend(recs)
    groups.update(fam)
    groups["pooled"] = [r for recs in by_cond.values() for r in recs]

    modes = sorted({m for recs in by_cond.values() for r in recs for m in r["modes"]})
    summary = {}
    for g, recs in groups.items():
        curves = {m: recovery(recs, m) for m in modes}
        curves = {m: c for m, c in curves.items() if c}
        summary[g] = {"curves": curves, "band": select_band(curves)}
    for g, s in summary.items():
        b = s["band"]
        n = next(iter(s["curves"].values()))["n"] if s["curves"] else 0
        print(f"{g:>12} (n={n}): " + (f"B* = layers {b['layers'][0]}-{b['layers'][-1]}  R={b['R']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]" if b else "no window modes"))
        if "ctl-win6" in s["curves"] and "win6" in s["curves"]:
            print(f"{'':>12}  win6 max R={max(s['curves']['win6']['R']):.2f}  vs mismatched-donor control max R={max(s['curves']['ctl-win6']['R']):.2f}")

    # Pre-registered H2 check on a held-out split, for the band frozen on calib
    frozen = args.patch_dir.parent.parent / "calib" / "patching" / "band.json"
    if args.patch_dir.parent.name != "calib" and frozen.exists():
        b = json.loads(frozen.read_text())["pooled"]["band"]
        mode, start = f"win{b['width']}", b["layers"][0]
        print(f"\nH2 check on this split for the frozen calib band B* = layers {b['layers'][0]}-{b['layers'][-1]}")
        for g in ("pooled", "indo-aryan", "dravidian"):
            cv = summary.get(g, {}).get("curves", {})
            if mode not in cv:
                continue
            r, lo, hi = cv[mode]["R"][start], cv[mode]["lo"][start], cv[mode]["hi"][start]
            med = float(np.median(cv[mode]["R"]))
            ctl = max(cv["ctl-win6"]["R"]) if "ctl-win6" in cv else float("nan")
            ok = r >= 0.5 and r >= 2 * ctl and r >= 2 * med
            print(f"{g:>12}: R(B*)={r:.2f} [{lo:.2f}, {hi:.2f}]  median window={med:.2f}  control max={ctl:.2f}  "
                  f"-> H2 {'SUPPORTED' if ok else 'NOT supported'} (R>=0.5, >=2x control, >=2x median)")
        summary["h2_frozen_band"] = {"band": b, "mode": mode}

    # recovery curves (every 2nd layer / window start) for the pooled and family groups
    for g in ("pooled", "indo-aryan", "dravidian"):
        cv = summary.get(g, {}).get("curves", {})
        if not cv:
            continue
        print(f"\n{g}: recovery R by layer (window modes: by window start)")
        names = [m for m in ("resid", "attn", "mlp", "win4", "win6", "win8", "ctl-win6", "rev-win6") if m in cv]
        n = max(len(cv[m]["R"]) for m in names)
        print("layer " + " ".join(f"{m:>8}" for m in names))
        for l in range(0, n, 2):
            print(f"{l:>5} " + " ".join(f"{cv[m]['R'][l]:>8.2f}" if l < len(cv[m]["R"]) else f"{'':>8}" for m in names))

    out = args.out or args.patch_dir / "band.json"
    out.write_text(json.dumps(summary, indent=1))

    if args.plot:
        import matplotlib.pyplot as plt

        for g, s in summary.items():
            if "curves" not in s:
                continue
            fig, ax = plt.subplots(figsize=(7, 3.5))
            for m, c in s["curves"].items():
                x = np.arange(len(c["R"]))
                ax.plot(x, c["R"], label=m)
                ax.fill_between(x, c["lo"], c["hi"], alpha=0.15)
            if s["band"]:
                ax.axvspan(s["band"]["layers"][0], s["band"]["layers"][-1], color="orange", alpha=0.15, label="B*")
            ax.set(xlabel="layer (window start for win*)", ylabel="normalised recovery R", title=g)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(args.patch_dir / f"recovery_{g}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
