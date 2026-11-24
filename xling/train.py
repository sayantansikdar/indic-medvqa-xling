"""Phase C: band-restricted LoRA with the representation-alignment loss (Eq. 6.6).

  L = CE(y | I, q_tg) + CE(y | I, q_en) + lambda * sum_{l in B} [1 - cos(h_l(I, q_tg), sg[h_l^base(I, q_en)])]

h_l is the pre-norm residual after layer l at the final input position; the English anchors come
from the base model before the adapter is attached, so they cannot drift during training.

  # ours: adapter + alignment on the diagnosed band
  python -m xling.train --band-json runs/medgemma/arogyabodha/calib/patching/band.json --placement band --lam 0.5 --out runs/train/ours
  # A2: same band, no alignment term
  ... --placement band --lam 0 --out runs/train/band_noalign
  # A1: random band of the same width (alignment on that band)
  ... --placement random --lam 0.5 --out runs/train/random_band
  # B4: all layers, task loss only, rank scaled to match the band's parameter count
  ... --placement all --match-budget --lam 0 --out runs/train/all_layers
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .common import TARGET_LANGS, condition_name, read_jsonl, set_seed
from .model import VLM, capture_final, load, load_image

ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def target_module_names(vlm: VLM, layer_idx: list[int], with_mlp: bool) -> list[str]:
    """Full module names of the chosen projections, restricted to the *language* decoder layers
    (the vision tower also has `layers.N.self_attn.q_proj`, so names alone are not enough)."""
    prefix = next(n for n, m in vlm.model.named_modules() if m is vlm.layers)
    want = ATTN + (MLP if with_mlp else [])
    return [f"{prefix}.{i}.{w}" for i in layer_idx for w in want]


def choose_layers(args, L: int) -> tuple[list[int], list[int]]:
    """(adapted layers, alignment layers)."""
    band = None
    if args.band_json:
        b = json.loads(Path(args.band_json).read_text())[args.band_group]["band"]
        band = b["layers"]
    elif args.band:
        s, e = map(int, args.band.split(":"))
        band = list(range(s, e + 1))
    if args.placement == "band":
        return band, band
    if args.placement == "random":
        w = len(band)
        rng = random.Random(args.seed)
        starts = [s for s in range(L - w + 1) if set(range(s, s + w)).isdisjoint(band)]
        s = rng.choice(starts)
        rb = list(range(s, s + w))
        return rb, rb
    if args.placement == "all":
        return list(range(L)), band or []
    raise ValueError(args.placement)


def align_loss(h: torch.Tensor, a: torch.Tensor, mean: torch.Tensor | None) -> torch.Tensor:
    """sum over band layers of 1 - cos(h, a); with `mean`, both are centred on the mean English anchor first."""
    if mean is not None:
        h, a = h - mean, a - mean
    return (1 - F.cosine_similarity(h, a, dim=-1)).sum()


class FinalStates:
    """Forward hooks that keep (with grad) the final-position residual after selected layers."""

    def __init__(self, vlm: VLM, layers: list[int]):
        self.store, self.handles = {}, []
        for i in layers:
            def hook(mod, args, output, i=i):
                hs = output[0] if isinstance(output, tuple) else output
                self.store[i] = hs[:, -1, :]
            self.handles.append(vlm.layers[i].register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--items", type=Path, default=Path("runs/data/arogyabodha.jsonl"))
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--variants", default="mt,mixed", help="comma-separated question variants to train on")
    ap.add_argument("--langs", default=",".join(TARGET_LANGS))
    ap.add_argument("--band-json", type=Path)
    ap.add_argument("--band-group", default="pooled")
    ap.add_argument("--band", help="explicit 'start:end' (inclusive) instead of --band-json")
    ap.add_argument("--placement", choices=["band", "random", "all"], default="band")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--match-budget", action="store_true", help="with --placement all, scale rank to the band's budget")
    ap.add_argument("--mlp", action="store_true", help="also adapt MLP projections (ablation A3)")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--align", choices=["centred", "raw"], default="centred",
                    help="cosine after subtracting the mean English anchor (centred) or on raw states. Raw cosine is "
                         "about 0.99 for any pair on Gemma 3 (a few huge shared dimensions), so it barely constrains anything.")
    ap.add_argument("--en-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=32)
    ap.add_argument("--max-pairs", type=int, help="cap on (item, language) training pairs per epoch")
    ap.add_argument("--quant", default="none", choices=["none", "4bit"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.out / "adapter_config.json").exists():
        print(f"[skip] adapter already trained: {args.out}")
        return
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_log.jsonl").unlink(missing_ok=True)  # a restarted run starts its log afresh
    data_dir = args.data_dir or args.items.parent

    vlm = load(args.model, quant=args.quant, attn="sdpa")
    L = vlm.n_layers
    adapt, align = choose_layers(args, L)
    rank = args.rank
    if args.placement == "all" and args.match_budget:
        band_w = len(align) if align else 6
        rank = max(1, round(args.rank * band_w / L))
    lam = args.lam if align else 0.0

    items = [it for it in read_jsonl(args.items) if it["split"] == "train"]
    conds = [condition_name(l, v) for l in args.langs.split(",") for v in args.variants.split(",")]
    pairs = [(it, c) for it in items for c in conds if c in it["text"]]
    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    n_steps = int(len(pairs) * args.epochs)

    def build(it, cond):
        t = it["text"][cond]
        return vlm.build(load_image(data_dir, it["image"]), vlm.prompt_text(t["question"], t["options"], it["letters"]))

    # English anchors from the frozen base model (before the adapter exists)
    anchors = {}
    if lam > 0:
        for it in tqdm({it["id"]: it for it, _ in pairs}.values(), desc="EN anchors"):
            with torch.no_grad(), capture_final(vlm, ["resid"]) as st:
                vlm.model(**build(it, "en"), use_cache=False, logits_to_keep=1)
            anchors[it["id"]] = torch.stack([st["resid"][l][0] for l in align]).cpu()
    anchor_mean = torch.stack(list(anchors.values())).float().mean(0).to(vlm.device) if anchors else None  # (|B|, d)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = vlm.model
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                     target_modules=target_module_names(vlm, adapt, args.mlp))
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    vlm = VLM(model, vlm.processor)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"adapted layers {adapt[0]}..{adapt[-1]} ({len(adapt)}), rank {rank}, align on {align if lam else 'none'}")
    print(f"trainable params {n_train:,} = {100 * n_train / n_total:.4f}% of {n_total:,}")
    (args.out / "train_config.json").write_text(json.dumps(
        {**vars(args), "adapted_layers": adapt, "align_layers": align if lam else [], "rank_used": rank,
         "trainable_params": n_train, "total_params": n_total, "n_pairs": len(pairs), "n_steps": n_steps},
        default=str, indent=2))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    from transformers import get_cosine_schedule_with_warmup

    n_updates = max(1, n_steps // args.accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * n_updates)), n_updates)
    hooks = FinalStates(vlm, align) if lam > 0 else None
    model.train()
    log = {"ce_tg": 0.0, "ce_en": 0.0, "align": 0.0}
    for step in tqdm(range(n_steps), desc="train"):
        it, cond = pairs[step % len(pairs)]
        target = torch.tensor([vlm.letter_ids[it["gold"]][0]], device=vlm.device)
        out = model(**build(it, cond), use_cache=False, logits_to_keep=1)
        ce_tg = F.cross_entropy(out.logits[:, -1].float(), target)
        loss = ce_tg
        if lam > 0:
            a = anchors[it["id"]].to(vlm.device).float()
            h = torch.stack([hooks.store[l][0] for l in align]).float()
            al = align_loss(h, a, anchor_mean if args.align == "centred" else None)
            loss = loss + lam * al
            log["align"] += al.item()
        if args.en_weight > 0:
            out_en = model(**build(it, "en"), use_cache=False, logits_to_keep=1)
            ce_en = F.cross_entropy(out_en.logits[:, -1].float(), target)
            loss = loss + args.en_weight * ce_en
            log["ce_en"] += ce_en.item()
        log["ce_tg"] += ce_tg.item()
        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tqdm.write(f"step {step + 1}: " + "  ".join(f"{k}={v / args.accum:.4f}" for k, v in log.items()))
            with open(args.out / "train_log.jsonl", "a") as f:
                f.write(json.dumps({"step": step + 1, **{k: v / args.accum for k, v in log.items()}}) + "\n")
            log = dict.fromkeys(log, 0.0)
    model.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
