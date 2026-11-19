"""Phase D tables: accuracy, gap, consistency, calibration, GRR and paired tests vs. the base run.

  python -m xling.report --root runs/medgemma/arogyabodha/test --base base --runs steer,band_noalign,ours
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import read_jsonl
from .stats import bootstrap_ci, ece, gap_recovery_ratio, holm, mcnemar


def load_preds(d: Path) -> dict[str, dict[str, dict]]:
    return {p.stem.removeprefix("preds_"): {r["id"]: r for r in read_jsonl(p)} for p in sorted(d.glob("preds_*.jsonl"))}


def summarise(run: dict, base: dict | None):
    en = run["en"]
    ids = list(en)
    rows, pvals = [], {}
    base_en_acc = np.mean([base["en"][i]["pred"] == base["en"][i]["gold"] for i in ids]) if base else None
    for cond, preds in run.items():
        ids_c = [i for i in ids if i in preds]
        corr = np.array([preds[i]["pred"] == preds[i]["gold"] for i in ids_c])
        conf = np.array([max(preds[i]["probs"]) for i in ids_c])
        acc, lo, hi = bootstrap_ci(corr)
        row = {
            "cond": cond, "n": len(ids_c), "acc": acc, "acc_lo": lo, "acc_hi": hi,
            "gap_vs_en": np.mean([en[i]["pred"] == en[i]["gold"] for i in ids_c]) - acc,
            "clc": np.mean([preds[i]["pred"] == en[i]["pred"] for i in ids_c]),
            "ece": ece(conf, corr),
            "letter_mass": np.mean([preds[i]["letter_mass"] for i in ids_c]),
            "fertility": np.mean([preds[i]["fertility"] for i in ids_c]),
        }
        if base and cond in base:
            b = np.array([base[cond][i]["pred"] == base[cond][i]["gold"] for i in ids_c])
            row["delta_vs_base"] = acc - b.mean()
            if cond == "en":
                row["en_retention"] = acc - base_en_acc
            else:
                row["grr"] = gap_recovery_ratio(acc, b.mean(), base_en_acc)
                pvals[cond] = mcnemar(b, corr)
        rows.append(row)
    for r, p in holm(pvals).items():
        next(x for x in rows if x["cond"] == r)["p_holm"] = p
    return rows


def fmt(rows, title):
    cols = ["cond", "n", "acc", "gap_vs_en", "clc", "ece", "delta_vs_base", "grr", "en_retention", "p_holm", "letter_mass"]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c == "acc":
                v = f"{r['acc']:.3f} [{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]"
            elif isinstance(v, float):
                v = f"{v:.3g}" if c == "p_holm" else f"{v:+.3f}" if c in ("gap_vs_en", "delta_vs_base", "en_retention") else f"{v:.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="runs/<model>/<dataset>/<split>")
    ap.add_argument("--base", default="base")
    ap.add_argument("--runs", default="", help="comma-separated run tags to compare against --base")
    args = ap.parse_args()

    base = load_preds(args.root / args.base)
    md = [fmt(summarise(base, None), f"{args.base} (zero-shot)")]
    all_rows = [{"run": args.base, **r} for r in summarise(base, None)]
    for tag in filter(None, args.runs.split(",")):
        rows = summarise(load_preds(args.root / tag), base)
        md.append(fmt(rows, tag))
        all_rows += [{"run": tag, **r} for r in rows]
    (args.root / "report.md").write_text("\n".join(md))
    keys = sorted({k for r in all_rows for k in r}, key=lambda k: (k != "run", k != "cond", k))
    with open(args.root / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
