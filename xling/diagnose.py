"""Phase B1–B4 from the cached residuals (CPU is fine).

  python -m xling.diagnose --test-dir runs/medgemma/arogyabodha/test/base \
                           --calib-dir runs/medgemma/arogyabodha/calib/base --variant mt --plot

Writes <test-dir>/diagnosis.csv (long format: metric, cond, layer, value, lo, hi),
plots, and <calib-dir>/steer.npz (mean English-minus-target vectors per layer, for baseline B3).

Cosine drift is reported twice: on raw states, and after subtracting the per-layer mean over all
languages. Gemma-family residual streams carry a few very large shared dimensions that make raw
cosine similarity look high everywhere; the centred version is the one to interpret.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import FAMILY, read_jsonl
from .stats import bootstrap_ci, linear_cka


def load_run(d: Path, conds: list[str]):
    R, P, lens, var = {}, {}, {}, {}
    for c in conds:
        R[c] = np.load(d / f"resid_{c}.npy", mmap_mode="r")
        P[c] = read_jsonl(d / f"preds_{c}.jsonl")
        z = np.load(d / f"lens_{c}.npz")
        lens[c] = (z["p_gold"], z["top_is_gold"])
        var[c] = np.load(d / f"var_{c}.npy")
    ids = [p["id"] for p in P["en"]]
    for c in conds:
        assert [p["id"] for p in P[c]] == ids, f"item order differs for {c}"
    return R, P, lens, var


def cos_rows(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def emergence_layer(top_is_gold: np.ndarray) -> np.ndarray:
    """First layer after which the gold letter stays top-ranked; L if never (incl. wrong final answers)."""
    N, L = top_is_gold.shape
    out = np.full(N, L, dtype=float)
    for i in range(N):
        t = top_is_gold[i]
        if t[-1]:
            bad = np.where(~t)[0]
            out[i] = 0 if len(bad) == 0 else bad[-1] + 1
    return out


def probe_curve(Xtr_by_layer, ytr, Xte_by_layer, yte, n_components=128, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    accs = []
    for Xtr, Xte in zip(Xtr_by_layer, Xte_by_layer):
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(n_components, *Xtr.shape), random_state=seed),
                            LogisticRegression(max_iter=2000))
        clf.fit(Xtr, ytr)
        accs.append(clf.score(Xte, yte) if not isinstance(yte, dict) else {k: clf.score(Xte[k], v) for k, v in yte.items()})
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--variant", default="mt", help="which Indic variant to diagnose (mt / mixed / pure)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    targets = [f"{l}-{args.variant}" for l in FAMILY]
    conds = ["en"] + targets
    Rt, Pt, lens_t, var_t = load_run(args.test_dir, conds)
    Rc, Pc, _, _ = load_run(args.calib_dir, conds)
    L = Rt["en"].shape[1]
    rows = []

    def add(metric, cond, layer, v, lo=None, hi=None):
        rows.append({"metric": metric, "cond": cond, "layer": layer, "value": v, "lo": lo, "hi": hi})

    # B1: drift and CKA ------------------------------------------------------
    print("B1 drift / CKA")
    mu = np.mean([np.asarray(Rt[c], dtype=np.float32).mean(0) for c in conds], axis=0)  # (L, d)
    en = np.asarray(Rt["en"], dtype=np.float32)
    for c in targets:
        tg = np.asarray(Rt[c], dtype=np.float32)
        for l in range(L):
            m, lo, hi = bootstrap_ci(1 - cos_rows(en[:, l], tg[:, l]))
            add("drift_raw", c, l, m, lo, hi)
            m, lo, hi = bootstrap_ci(1 - cos_rows(en[:, l] - mu[l], tg[:, l] - mu[l]))
            add("drift_centred", c, l, m, lo, hi)
            add("cka", c, l, linear_cka(en[:, l], tg[:, l]))
            add("norm_ratio", c, l, float(np.linalg.norm(tg[:, l], axis=-1).mean() / np.linalg.norm(en[:, l], axis=-1).mean()))

    # B2: language-ID probe and answer-probe transfer -------------------------
    print("B2 probes (PCA-128 + logistic regression per layer)")
    Xc = {c: np.asarray(Rc[c], dtype=np.float32) for c in conds}
    Xt = {c: np.asarray(Rt[c], dtype=np.float32) for c in conds}
    lid_tr = [np.concatenate([Xc[c][:, l] for c in conds]) for l in range(L)]
    lid_te = [np.concatenate([Xt[c][:, l] for c in conds]) for l in range(L)]
    y_tr = np.concatenate([[i] * len(Xc[c]) for i, c in enumerate(conds)])
    y_te = np.concatenate([[i] * len(Xt[c]) for i, c in enumerate(conds)])
    for l, a in enumerate(probe_curve(lid_tr, y_tr, lid_te, y_te)):
        add("langid_acc", "all", l, a)
    gold_c = np.array(["ABCDE".index(p["gold"]) for p in Pc["en"]])
    gold_t = np.array(["ABCDE".index(p["gold"]) for p in Pt["en"]])
    ans = probe_curve([Xc["en"][:, l] for l in range(L)], gold_c,
                      [{c: Xt[c][:, l] for c in conds} for l in range(L)], {c: gold_t for c in conds})
    for l, accs in enumerate(ans):
        for c in targets:
            add("answer_probe_transfer", c, l, accs[c] / max(accs["en"], 1e-8))
        add("answer_probe_acc", "en", l, accs["en"])

    # B3: lens ---------------------------------------------------------------
    print("B3 logit lens")
    for c in conds:
        p_gold, top = lens_t[c]
        for l in range(p_gold.shape[1]):
            add("lens_p_gold", c, l, float(p_gold[:, l].mean()))
        both_right = np.array([pe["pred"] == pe["gold"] for pe in Pt["en"]])
        em = emergence_layer(top.astype(bool))
        m, lo, hi = bootstrap_ci(em[both_right])
        add("emergence_layer_on_en_correct", c, -1, m, lo, hi)

    # B4: visual-attention ratio --------------------------------------------
    print("B4 visual attention")
    for c in conds:
        for l in range(L):
            m, lo, hi = bootstrap_ci(var_t[c][:, l])
            add("var", c, l, m, lo, hi)
    for c in targets:
        for l in range(L):
            m, lo, hi = bootstrap_ci(var_t["en"][:, l] - var_t[c][:, l])
            add("var_deficit", c, l, m, lo, hi)

    out = args.test_dir / f"diagnosis_{args.variant}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "cond", "layer", "value", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    # Steering vectors for baseline B3, from the calib split only
    np.savez(args.calib_dir / f"steer_{args.variant}.npz",
             **{c: (Xc["en"] - Xc[c]).mean(0) for c in targets})

    if args.plot:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(rows)
        panels = ["drift_centred", "cka", "langid_acc", "answer_probe_transfer", "lens_p_gold", "var_deficit"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
        for ax, metric in zip(axes.flat, panels):
            for c, g in df[(df.metric == metric) & (df.layer >= 0)].groupby("cond"):
                ax.plot(g.layer, g.value, label=c)
                if g.lo.notna().all():
                    ax.fill_between(g.layer, g.lo.astype(float), g.hi.astype(float), alpha=0.15)
            ax.set_title(metric)
            ax.set_xlabel("decoder layer")
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.test_dir / f"diagnosis_{args.variant}.png", dpi=150)


if __name__ == "__main__":
    main()
