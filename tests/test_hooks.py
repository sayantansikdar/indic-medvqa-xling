"""Check the hook semantics on tiny random Gemma 3 / Qwen2 text decoders (CPU, no downloads)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from xling.model import VLM, capture_final, patch_final


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        s = s.strip()
        return [ord(s) - ord("A") + 10]


class FakeProc:
    tokenizer = FakeTok()


def tiny(kind):
    if kind == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        cfg = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=16)
        cfg._attn_implementation = "eager"
        return Gemma3ForCausalLM(cfg).eval()
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(params=["gemma3", "qwen2"])
def vlm(request):
    torch.manual_seed(0)
    return VLM(tiny(request.param), FakeProc())


def test_contributions_sum_to_residual(vlm):
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad(), capture_final(vlm, ["resid", "attn", "mlp"]) as st:
        vlm.model(input_ids=ids)
    for i in range(1, vlm.n_layers):
        recon = st["resid"][i - 1] + st["attn"][i] + st["mlp"][i]
        assert torch.allclose(recon, st["resid"][i], atol=1e-5)


def test_resid_patch_at_last_layer_reproduces_donor(vlm):
    a, b = torch.randint(0, 64, (1, 7)), torch.randint(0, 64, (1, 9))
    with torch.no_grad(), capture_final(vlm, ["resid"]) as st:
        donor_logits = vlm.model(input_ids=a).logits[0, -1]
    last = vlm.n_layers - 1
    with torch.no_grad(), patch_final(vlm, {("resid", last): {0: st["resid"][last][0]}}):
        patched = vlm.model(input_ids=b).logits[0, -1]
    assert torch.allclose(patched, donor_logits, atol=1e-4)


def test_batched_patch_only_touches_its_row(vlm):
    b = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        base = vlm.model(input_ids=b).logits[0, -1]
        donor = torch.randn(vlm.model.config.hidden_size)
        with patch_final(vlm, {("mlp", 1): {1: donor}}):
            out = vlm.model(input_ids=b.repeat(3, 1)).logits[:, -1]
    assert torch.allclose(out[0], base, atol=1e-5) and torch.allclose(out[2], base, atol=1e-5)
    assert not torch.allclose(out[1], base, atol=1e-3)


def test_lens_on_last_layer_matches_logits(vlm):
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad(), capture_final(vlm, ["resid"]) as st:
        logits = vlm.model(input_ids=ids).logits[0, -1]
        lens = vlm.lens_letter_logits(st["resid"][-1][0], list("ABCD"))
    assert torch.allclose(lens, vlm.letter_logits(logits, list("ABCD")), atol=1e-4)


def test_eager_attentions_available(vlm):
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        out = vlm.model(input_ids=ids, output_attentions=True)
    assert len(out.attentions) == vlm.n_layers and out.attentions[0].shape[-1] == 7


def test_window_patching_chunked_equals_unchunked(vlm):
    from xling.patching import run_mode

    a, b = torch.randint(0, 64, (1, 7)), torch.randint(0, 64, (1, 9))
    with torch.no_grad(), capture_final(vlm, ["resid", "attn", "mlp"]) as donors:
        vlm.model(input_ids=a)
    inputs = {"input_ids": b}
    _, full = run_mode(vlm, inputs, "win2", donors, list("ABCD"), "B", batch=64)
    _, chunked = run_mode(vlm, inputs, "win2", donors, list("ABCD"), "B", batch=2)
    assert len(full) == vlm.n_layers - 1
    assert torch.allclose(torch.tensor(full), torch.tensor(chunked), atol=1e-5)
    # a window covering every layer yields a single row
    _, all_layers = run_mode(vlm, inputs, f"win{vlm.n_layers}", donors, list("ABCD"), "B", batch=4)
    assert len(all_layers) == 1
