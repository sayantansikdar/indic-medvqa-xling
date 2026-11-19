"""Phase A (behavioural baseline) and Phase B cache in one pass.

For every item and condition it records the option-letter distribution at the first answer
position. With --capture it also stores, per decoder layer at the final input position:
  resid  pre-norm residual stream after the layer           -> resid_<cond>.npy  (N, L, d) fp16
  lens   logit-lens P(gold) over the option letters          -> lens_<cond>.npz
  VAR    attention mass from the final position to the image -> var_<cond>.npy   (N, L)

  python -m xling.infer --model medgemma --items runs/data/arogyabodha.jsonl --split test --capture
  python -m xling.infer --model medgemma --items ... --split test --adapter runs/train/ours --tag ours
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .common import RUNS_DIR, read_jsonl, script_share, write_jsonl
from .model import capture_final, load, load_image, steer


def conditions_of(items: list[dict], wanted: list[str] | None) -> list[str]:
    avail = sorted({c for it in items for c in it["text"]}, key=lambda c: (c != "en", c))
    return [c for c in avail if wanted is None or c in wanted]


def run_condition(vlm, items, cond, data_dir, args, steer_vec=None):
    capture = args.capture
    L = vlm.n_layers
    preds, resid, lens_p, lens_top, var = [], [], [], [], []
    for it in tqdm(items, desc=cond, leave=False):
        t = it["text"][cond]
        letters = it["letters"]
        text = vlm.prompt_text(t["question"], t["options"], letters, args.instruction)
        inputs = vlm.build(load_image(data_dir, it["image"]), text)
        img = vlm.image_mask(inputs)
        with torch.no_grad(), capture_final(vlm, ["resid"] if capture else []) as store:
            if steer_vec is not None:
                start = int(img.nonzero().max()) + 1 if img.any() else 0
                with steer(vlm, args.steer_layer, steer_vec, args.steer_alpha, start):
                    out = vlm.model(**inputs, use_cache=False, output_attentions=capture)
            else:
                out = vlm.model(**inputs, use_cache=False, output_attentions=capture)
        last = out.logits[0, -1]
        ll = vlm.letter_logits(last, letters)
        probs = torch.softmax(ll, -1)
        all_ids = [i for Lt in letters for i in vlm.letter_ids[Lt]]
        mass = float(torch.softmax(last.float(), -1)[all_ids].sum())
        q_tokens = len(vlm.tok.encode(t["question"], add_special_tokens=False))
        rec = {
            "id": it["id"],
            "gold": it["gold"],
            "letters": letters,
            "probs": [round(float(p), 6) for p in probs],
            "pred": letters[int(probs.argmax())],
            "letter_mass": round(mass, 4),
            "n_tokens": int(inputs["input_ids"].shape[1]),
            "q_tokens": q_tokens,
            "fertility": round(q_tokens / max(1, len(t["question"].split())), 3),
            "source": it.get("source"),
        }
        preds.append(rec)
        if capture:
            hs = torch.stack([h[0] for h in store["resid"]])  # (L, d) pre-norm layer outputs, final position
            resid.append(hs.to(torch.float16).cpu().numpy())
            g = letters.index(it["gold"])
            lens_logits = vlm.lens_letter_logits(hs[:-1], letters)  # layers 0..L-2, pre-norm
            lp = torch.softmax(lens_logits, -1)
            lp = torch.cat([lp, probs[None].to(lp.device)], 0)  # final layer = the model's actual output
            lens_p.append(lp[:, g].cpu().numpy())
            lens_top.append((lp.argmax(-1) == g).cpu().numpy())
            att = torch.stack([a[0, :, -1, :] for a in out.attentions])  # (L, H, T)
            var.append(att[:, :, img].sum(-1).mean(-1).float().cpu().numpy())
    return preds, resid, lens_p, lens_top, var


def fidelity_probe(vlm, items, cond, data_dir, n, max_new_tokens=48):
    """Response-language fidelity: free reply to the question without the letter instruction."""
    lang = cond.split("-")[0]
    rows = []
    for it in items[:n]:
        t = it["text"][cond]
        text = vlm.prompt_text(t["question"], t["options"], it["letters"], instruction="")
        inputs = vlm.build(load_image(data_dir, it["image"]), text)
        with torch.no_grad():
            gen = vlm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = vlm.tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"id": it["id"], "reply": reply, "script_share": round(script_share(reply, lang), 3)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, help="default: directory of --items")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conds", help="comma-separated, e.g. en,hi-pure,ta-pure (default: all in the file)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--capture", action="store_true", help="store residuals, lens and visual attention (eager attention)")
    ap.add_argument("--instruction", default="letter")
    ap.add_argument("--quant", default="none", choices=["none", "4bit"])
    ap.add_argument("--adapter")
    ap.add_argument("--steer-file", type=Path, help="npz from xling.diagnose with per-condition steering vectors")
    ap.add_argument("--steer-layer", type=int)
    ap.add_argument("--steer-alpha", type=float, default=1.0)
    ap.add_argument("--fidelity", type=int, default=0, help="also run the free-reply probe on the first N items")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data_dir = args.data_dir or args.items.parent
    items = [it for it in read_jsonl(args.items) if it["split"] == args.split]
    if args.limit:
        items = items[: args.limit]
    dataset = items[0]["dataset"]
    out = args.out or RUNS_DIR / args.model / dataset / args.split / args.tag
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions_of(items, args.conds.split(",") if args.conds else None)

    vlm = load(args.model, quant=args.quant, attn="eager" if args.capture else "sdpa", adapter=args.adapter)
    steer_vecs = np.load(args.steer_file) if args.steer_file else None
    (out / "config.json").write_text(json.dumps({**vars(args), "conds": conds, "n_items": len(items), "n_layers": vlm.n_layers}, default=str, indent=2))

    for cond in conds:
        its = [it for it in items if cond in it["text"]]
        pred_path = out / f"preds_{cond}.jsonl"
        if pred_path.exists() and len(read_jsonl(pred_path)) == len(its) and (not args.capture or (out / f"resid_{cond}.npy").exists()):
            print(f"[skip] {cond} already done")
            continue
        sv = None
        if steer_vecs is not None and cond != "en":
            sv = torch.tensor(steer_vecs[cond][args.steer_layer], device=vlm.device)
        preds, resid, lens_p, lens_top, var = run_condition(vlm, its, cond, data_dir, args, sv)
        write_jsonl(pred_path, preds)
        acc = np.mean([p["pred"] == p["gold"] for p in preds])
        mass = np.mean([p["letter_mass"] for p in preds])
        print(f"{cond:>10}: acc={acc:.3f}  n={len(preds)}  mean letter mass={mass:.3f}")
        if mass < 0.5:
            print("  WARNING: <50% of probability is on option letters; the model wants to emit something else first.")
        if args.capture:
            np.save(out / f"resid_{cond}.npy", np.stack(resid))
            np.savez(out / f"lens_{cond}.npz", p_gold=np.stack(lens_p), top_is_gold=np.stack(lens_top))
            np.save(out / f"var_{cond}.npy", np.stack(var))
        if args.fidelity and cond != "en":
            write_jsonl(out / f"fidelity_{cond}.jsonl", fidelity_probe(vlm, its, cond, data_dir, args.fidelity))


if __name__ == "__main__":
    main()
