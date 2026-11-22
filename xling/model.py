"""Model loading, prompting, option scoring and residual-stream hooks.

The hooks are written against the decoder-layer structure, so the same code runs on
MedGemma (Gemma 3) and on Qwen2.5-VL / Lingshu:

  Gemma 3 layer:  h = h + post_attention_layernorm(attn(input_layernorm(h)))
                  h = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
  Qwen2 layer:    h = h + self_attn(input_layernorm(h))[0]
                  h = h + mlp(post_attention_layernorm(h))

So the *additive contribution* of attention / MLP at layer l is the output of
post_attention_layernorm / post_feedforward_layernorm (Gemma 3) or of self_attn / mlp (Qwen2).
Patching those contributions gives per-layer effects that do not accumulate with depth,
which is what the corrected causal test (Phase B5) needs.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

MODELS = {
    "medgemma": "google/medgemma-1.5-4b-it",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "lingshu": "lingshu-medical-mllm/Lingshu-7B",
}

INSTRUCTIONS = {
    "letter": "Answer with the letter of the correct option only.",
    # Baseline B2: ask the model to reason in English, still answer with a letter
    "think_en": "First understand the question in English, then answer with the letter of the correct option only.",
}


def load(name: str, quant: str = "none", attn: str = "sdpa", adapter: str | None = None) -> "VLM":
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before the first CUDA allocation
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODELS.get(name, name)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (Colab: Runtime > Change runtime type > L4 or A100).")
    if not torch.cuda.is_bf16_supported():
        # Gemma 3 overflows in fp16 (inf/NaN activations); T4/P100/V100 are not usable for it.
        raise RuntimeError("This GPU has no bf16 support. Use an L4 or A100 runtime.")
    kwargs = dict(dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda")
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    proc_kwargs = {}
    if "qwen" in model.config.model_type:
        # cap visual tokens so that 7B controls fit comfortably next to MedGemma's 256 image tokens
        proc_kwargs = dict(min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.requires_grad_(False)  # inference/analysis only; xling.train attaches trainable LoRA weights itself
    model.eval()
    return VLM(model, processor)


def _unwrap(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _text_config(model):
    cfg = _unwrap(model).config
    return getattr(cfg, "text_config", cfg)


def find_text_stack(model):
    """Return (text_model, layers) for the language decoder, across Transformers versions."""
    base = _unwrap(model)
    n = _text_config(model).num_hidden_layers
    for path in ("model.language_model", "language_model.model", "language_model", "model"):
        obj = base
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        layers = getattr(obj, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) == n and hasattr(obj, "norm"):
            return obj, layers
    raise RuntimeError("Could not locate the language-model decoder layers; check the Transformers version.")


@dataclass
class VLM:
    model: torch.nn.Module
    processor: object
    text_model: torch.nn.Module = field(init=False)
    layers: torch.nn.ModuleList = field(init=False)

    def __post_init__(self):
        self.text_model, self.layers = find_text_stack(self.model)
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        cfg = _unwrap(self.model).config
        self.image_token_id = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        self.softcap = getattr(_text_config(self.model), "final_logit_softcapping", None)
        self.is_gemma = hasattr(self.layers[0], "post_feedforward_layernorm")
        self.letter_ids = {}
        for L in "ABCDE":
            ids = []
            for s in (L, " " + L):
                enc = self.tok.encode(s, add_special_tokens=False)
                if len(enc) == 1 and enc[0] not in ids:
                    ids.append(enc[0])
            self.letter_ids[L] = ids

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def device(self):
        return next(self.layers[0].parameters()).device

    # ------------------------------------------------------------------ prompts
    def prompt_text(self, question: str, options: dict, letters: list[str], instruction: str = "letter") -> str:
        opts = "\n".join(f"{L}. {options[L]}" for L in letters)
        instr = INSTRUCTIONS.get(instruction, instruction)
        return f"{question}\n\n{opts}\n\n{instr}" if instr else f"{question}\n\n{opts}"

    def build(self, image: Image.Image, text: str):
        """Image-first chat prompt with generation prompt, as a batch of one on the GPU."""
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    @staticmethod
    def repeat(inputs, n: int):
        """Stack n copies of a single-example input (works for Gemma and Qwen pixel layouts)."""
        return {k: torch.cat([v] * n, 0) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def image_mask(self, inputs) -> torch.Tensor:
        return inputs["input_ids"][0] == self.image_token_id

    # ------------------------------------------------------------------ scoring
    def letter_logits(self, logits: torch.Tensor, letters: list[str]) -> torch.Tensor:
        """logits: (..., vocab) -> (..., n_letters); each letter scores logsumexp over its token variants."""
        cols = [torch.logsumexp(logits[..., self.letter_ids[L]].float(), dim=-1) for L in letters]
        return torch.stack(cols, dim=-1)

    def lens_letter_logits(self, h: torch.Tensor, letters: list[str]) -> torch.Tensor:
        """Logit lens: final norm + unembedding, restricted to the option letters. h: (..., d), pre-norm."""
        lm_head = _unwrap(self.model).lm_head
        ids = [i for L in letters for i in self.letter_ids[L]]
        w = lm_head.weight[ids]
        z = self.text_model.norm(h.to(w.dtype)) @ w.T
        if self.softcap:
            z = torch.tanh(z / self.softcap) * self.softcap
        out, j = [], 0
        for L in letters:
            k = len(self.letter_ids[L])
            out.append(torch.logsumexp(z[..., j : j + k].float(), dim=-1))
            j += k
        return torch.stack(out, dim=-1)

    def contrib_modules(self, layer_idx: int):
        """(attention-contribution module, MLP-contribution module) of decoder layer `layer_idx`."""
        layer = self.layers[layer_idx]
        if self.is_gemma:
            return layer.post_attention_layernorm, layer.post_feedforward_layernorm
        return layer.self_attn, layer.mlp

    def module(self, site: str, layer_idx: int):
        """site in {resid, attn, mlp}. `resid` is the decoder layer's output (residual stream after layer)."""
        if site == "resid":
            return self.layers[layer_idx]
        attn, mlp = self.contrib_modules(layer_idx)
        return attn if site == "attn" else mlp


# ---------------------------------------------------------------------- hooks
def _main_tensor(output):
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def capture_final(vlm: VLM, sites: list[str]):
    """Record the final-position vector of every layer for each site. Yields {site: list[L] of (B, d)}."""
    store = {s: [None] * vlm.n_layers for s in sites}
    handles = []
    for s in sites:
        for i in range(vlm.n_layers):
            def hook(mod, args, output, s=s, i=i):
                store[s][i] = _main_tensor(output)[:, -1, :].detach().clone()
            handles.append(vlm.module(s, i).register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def patch_final(vlm: VLM, plan: dict[tuple[str, int], dict[int, torch.Tensor]]):
    """Overwrite final-position outputs. plan[(site, layer)] = {batch_row: replacement vector}."""
    handles = []
    for (site, i), rows in plan.items():
        def hook(mod, args, output, rows=rows):
            hs = _main_tensor(output)
            for b, v in rows.items():
                hs[b, -1, :] = v.to(hs.dtype)
            return output
        handles.append(vlm.module(site, i).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def steer(vlm: VLM, layer_idx: int, vector: torch.Tensor, alpha: float, start: int):
    """Baseline B3: add alpha * vector to the residual stream after `layer_idx`, at positions >= start."""
    def hook(mod, args, output):
        hs = _main_tensor(output)
        hs[:, start:, :] += alpha * vector.to(hs.dtype)
        return output
    h = vlm.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def load_image(data_dir: Path, rel: str) -> Image.Image:
    return Image.open(Path(data_dir) / rel).convert("RGB")
