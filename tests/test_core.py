import json

import numpy as np
import pytest

from xling.band import recovery, select_band
from xling.common import from_bf16_bits, resid_path, script_share, to_bf16_bits, write_jsonl
from xling.data import split_of, strip_gloss
from xling.stats import ece, gap_recovery_ratio, holm, linear_cka, mcnemar


def test_strip_gloss_removes_english_parentheses_only():
    ta = "சப்யூடேனியஸ் எம்ஃபைசீமா (subcutaneous emphysema), இருதரப்பு நிமோனியா (bilateral pneumonia)"
    assert strip_gloss(ta) == "சப்யூடேனியஸ் எம்ஃபைசீமா, இருதரப்பு நிமோனியா"
    # parentheses with Indic content (e.g. numerals spelled out) are kept
    as_ = "এগৰাকী ৩৮ (আঠত্ৰিশ) বছৰীয়া"
    assert strip_gloss(as_) == as_
    # text that is only a gloss is left untouched rather than emptied
    assert strip_gloss("(MRI)") == "(MRI)"
    assert strip_gloss("5S rDNA சிக்னல்கள் (signals)") == "5S rDNA சிக்னல்கள்"
    assert strip_gloss("பக்கவாட்டு [lateral] கதிரியக்கப் படம் [radiograph] என்ன?") == "பக்கவாட்டு கதிரியக்கப் படம் என்ன?"


def test_split_is_deterministic_and_proportional():
    fr = {"test": 0.5, "calib": 0.2, "train": 0.3}
    a = [split_of(f"h{i}", 13, fr) for i in range(5000)]
    assert a == [split_of(f"h{i}", 13, fr) for i in range(5000)]
    assert abs(a.count("test") / 5000 - 0.5) < 0.03 and abs(a.count("calib") / 5000 - 0.2) < 0.03


def test_script_share():
    assert script_share("यह एक परीक्षण है", "hi") == 1.0
    assert script_share("This is English", "hi") == 0.0
    assert script_share("This is English", "en") == 1.0


def test_cka_properties():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert linear_cka(X, X) == pytest.approx(1.0)
    assert linear_cka(X, X @ Q * 3.0) == pytest.approx(1.0)  # invariant to rotation and scale
    assert linear_cka(X, rng.normal(size=(200, 16))) < 0.2


def test_paired_stats():
    a = np.array([1] * 50 + [0] * 50, bool)
    b = a.copy()
    assert mcnemar(a, b) == 1.0
    b[50:80] = True  # 30 discordant pairs in one direction
    assert mcnemar(a, b) < 1e-6
    adj = holm({"hi": 0.01, "bn": 0.04, "ta": 0.03, "te": 0.5})
    assert adj["hi"] == pytest.approx(0.04) and adj["ta"] == pytest.approx(0.09) and adj["te"] == pytest.approx(0.5)
    assert ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert gap_recovery_ratio(0.55, 0.40, 0.70) == pytest.approx(0.5)


def test_band_selection_and_ratio_of_sums():
    L = 10
    recs = []
    for i in range(40):
        p_en, p_tg = 0.9, 0.1
        win = [p_tg + (0.7 if 4 <= s <= 5 else 0.1) for s in range(L - 3)]  # win4 peaks at start 4-5
        recs.append({"p_en": p_en, "p_tg": p_tg, "modes": {"win4": win, "rev-win4": win}})
    c = recovery(recs, "win4")
    assert np.argmax(c["R"]) in (4, 5) and c["R"][4] == pytest.approx(0.875)
    band = select_band({"win4": c})
    assert band["layers"][0] in (4, 5) and band["width"] == 4


def test_diagnose_end_to_end(tmp_path, monkeypatch):
    """Synthetic caches where the Indic states diverge from English only in layers 3-4."""
    from xling import diagnose

    rng = np.random.default_rng(0)
    N, L, d = 60, 6, 24
    conds = ["en", "hi-pure", "bn-pure", "ta-pure", "te-pure"]
    for split in ("test", "calib"):
        base = rng.normal(size=(N, L, d)).astype(np.float32)
        gold = rng.choice(list("ABCD"), N)
        D = tmp_path / split
        D.mkdir()
        for c in conds:
            x = base.copy()
            if c != "en":
                x[:, 3:5] += rng.normal(scale=3.0, size=(N, 2, d))
            keep = [i for i in range(N) if not (c == "ta-pure" and i in (5, 17))]  # filtered items, as with MT
            x, lens_n = x[keep], len(keep)
            np.save(resid_path(D, c), to_bf16_bits(x))
            np.savez(D / f"lens_{c}.npz", p_gold=rng.random((lens_n, L)), top_is_gold=rng.random((lens_n, L)) > 0.5)
            np.save(D / f"var_{c}.npy", rng.random((lens_n, L)))
            write_jsonl(D / f"preds_{c}.jsonl", [{"id": f"i{i}", "gold": gold[i], "pred": gold[i] if rng.random() < 0.6 else "A"} for i in keep])
    monkeypatch.setattr("sys.argv", ["diagnose", "--test-dir", str(tmp_path / "test"), "--calib-dir", str(tmp_path / "calib"), "--variant", "pure"])
    diagnose.main()
    import csv

    rows = list(csv.DictReader(open(tmp_path / "test" / "diagnosis_pure.csv")))
    drift = {int(r["layer"]): float(r["value"]) for r in rows if r["metric"] == "drift_centred" and r["cond"] == "hi-pure"}
    assert max(drift, key=drift.get) in (3, 4) and drift[0] < 1e-3
    share = {int(r["layer"]): float(r["value"]) for r in rows if r["metric"] == "lang_var_share"}
    assert max(share, key=share.get) in (3, 4) and share[0] < 1e-3
    assert any(r["metric"] == "drift_langcentred" for r in rows)
    assert set(np.load(tmp_path / "calib" / "steer_pure.npz").files) == {"hi-pure", "bn-pure", "ta-pure", "te-pure"}


def test_bf16_cache_keeps_large_activations():
    x = np.array([[1.0, -3.5, 7e4, -2e5, 1e-3]], dtype=np.float32)  # beyond fp16's 65,504
    y = from_bf16_bits(to_bf16_bits(x))
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, x, rtol=1 / 128)


def test_band_cli_reports_frozen_band_on_test(tmp_path, monkeypatch, capsys):
    from xling import band

    L = 10
    def recs(peak):
        win = [0.1 + (0.8 if s in peak else 0.0) for s in range(L - 3)]
        return [{"p_en": 0.9, "p_tg": 0.1, "modes": {"win4": win, "ctl-win6": [0.15] * (L - 5)}} for _ in range(30)]
    for split, peak in (("calib", (4,)), ("test", (3, 4))):
        d = tmp_path / split / "patching"
        d.mkdir(parents=True)
        write_jsonl(d / "patch_hi-mt.jsonl", recs(peak))
        write_jsonl(d / "patch_ta-mt.jsonl", recs(peak))
    for split in ("calib", "test"):
        monkeypatch.setattr("sys.argv", ["band", "--patch-dir", str(tmp_path / split / "patching"), "--plot"])
        band.main()
    out = capsys.readouterr().out
    assert "frozen calib band B* = layers 4-7" in out and "H2 SUPPORTED" in out
    saved = json.loads((tmp_path / "test" / "patching" / "band.json").read_text())
    assert saved["h2_frozen_band"]["band"]["layers"] == [4, 5, 6, 7]


def test_centred_alignment_sees_content_that_raw_cosine_misses():
    """Gemma-like states: a few huge shared dimensions make raw cosine ~1 for any pair."""
    import torch

    from xling.train import align_loss

    g = torch.Generator().manual_seed(0)
    shared = torch.zeros(4, 64)
    shared[:, :2] = 3000.0  # massive activation dimensions, identical across inputs
    anchors = shared + torch.randn(50, 4, 64, generator=g)  # English anchors for 50 items
    mean = anchors.mean(0)
    a = anchors[0]
    h_same, h_other = a + 0.05 * torch.randn(4, 64, generator=g), anchors[1]  # same item vs a different item
    assert align_loss(h_other, a, None) < 1e-3  # raw: a different item already looks "aligned"
    assert align_loss(h_other, a, mean) > 2.0  # centred: clearly misaligned
    assert align_loss(h_same, a, mean) < 0.1
