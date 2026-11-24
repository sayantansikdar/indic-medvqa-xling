"""Phase D tables: accuracy, gap, consistency, calibration, GRR and paired tests vs. the base run.

  python -m xling.report --root runs/medgemma/arogyabodha/test --base base --runs steer,band_noalign,ours
  python -m xling.report --root runs/medgemma/vqarad/test --compare ours_c:random_band_c,ours_c:all_layers   # paired run vs run
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import read_jsonl
from .stats import bootstrap_ci, ece, gap_recovery_ratio, holm, mcnemar, sign_flip


def load_preds(d: Path, exclude: set[str] = frozenset()) -> dict[str, dict[str, dict]]:
    return {p.stem.removeprefix("preds_"): {r["id"]: r for r in read_jsonl(p) if r.get("source") not in exclude}
            for p in sorted(d.glob("preds_*.jsonl"))}


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


def compare(a: dict, b: dict):
    """Paired A-vs-B on the items both runs answered.

    Per condition: accuracy difference with a bootstrap CI and an exact McNemar test, Holm-corrected over the
    non-English conditions. Per variant (mt, mixed, ...): a pooled row that averages each item's A-B difference
    over the languages, so items stay the independent unit; tested by sign-flip permutation.
    """
    rows, pvals, per_item = [], {}, {}
    for cond in [c for c in a if c in b]:
        ids = [i for i in a[cond] if i in b[cond]]
        ca = np.array([a[cond][i]["pred"] == a[cond][i]["gold"] for i in ids])
        cb = np.array([b[cond][i]["pred"] == b[cond][i]["gold"] for i in ids])
        d, lo, hi = bootstrap_ci(ca.astype(float) - cb)
        rows.append({"cond": cond, "n": len(ids), "acc_a": ca.mean(), "acc_b": cb.mean(),
                     "delta": d, "delta_lo": lo, "delta_hi": hi, "a_only": int((ca & ~cb).sum()), "b_only": int((~ca & cb).sum())})
        if cond != "en":
            pvals[cond] = mcnemar(cb, ca)
            variant = cond.split("-", 1)[1]
            for i, x in zip(ids, ca.astype(float) - cb):
                per_item.setdefault(variant, {}).setdefault(i, []).append(x)
    for r, p in holm(pvals).items():
        next(x for x in rows if x["cond"] == r)["p_holm"] = p
    for variant, items in sorted(per_item.items()):
        diffs = np.array([np.mean(v) for v in items.values()])
        d, lo, hi = bootstrap_ci(diffs)
        rows.append({"cond": f"{variant} (pooled)", "n": len(diffs), "delta": d, "delta_lo": lo, "delta_hi": hi,
                     "p_perm": sign_flip(diffs)})
    return rows


def fmt_compare(rows, title):
    cols = ["cond", "n", "acc_a", "acc_b", "delta", "a_only", "b_only", "p_holm", "p_perm"]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c == "delta":
                v = f"{r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
            elif isinstance(v, float):
                v = f"{v:.3g}" if c.startswith("p_") else f"{v:.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--exclude-sources", default="", help="comma-separated item sources to leave out")
    ap.add_argument("--compare", default="", help="paired run-vs-run tests instead of the report, e.g. ours_c:random_band_c,ours_c:all_layers")
    args = ap.parse_args()
    exclude = set(filter(None, args.exclude_sources.split(",")))
    suffix = "_excl-" + "-".join(sorted(exclude)) if exclude else ""

    if args.compare:
        md, all_rows = [], []
        for pair in filter(None, args.compare.split(",")):
            ta, tb = pair.split(":")
            rows = compare(load_preds(args.root / ta, exclude), load_preds(args.root / tb, exclude))
            md.append(fmt_compare(rows, f"{ta} (A) vs {tb} (B)"))
            all_rows += [{"a": ta, "b": tb, **r} for r in rows]
        (args.root / f"paired{suffix}.md").write_text("\n".join(md))
        keys = sorted({k for r in all_rows for k in r}, key=lambda k: (k not in ("a", "b"), k != "cond", k))
        with open(args.root / f"paired{suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print("\n".join(md))
        return

    base = load_preds(args.root / args.base, exclude)
    md = [fmt(summarise(base, None), f"{args.base} (zero-shot)")]
    all_rows = [{"run": args.base, **r} for r in summarise(base, None)]
    for tag in filter(None, args.runs.split(",")):
        rows = summarise(load_preds(args.root / tag, exclude), base)
        md.append(fmt(rows, tag))
        all_rows += [{"run": tag, **r} for r in rows]
    (args.root / f"report{suffix}.md").write_text("\n".join(md))
    keys = sorted({k for r in all_rows for k in r}, key=lambda k: (k != "run", k != "cond", k))
    with open(args.root / f"report{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
