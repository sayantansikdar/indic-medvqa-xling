import json

import numpy as np
import pytest

from xling.band import recovery, select_band
from xling.common import script_share, write_jsonl
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
            np.save(D / f"resid_{c}.npy", x.astype(np.float16))
            np.savez(D / f"lens_{c}.npz", p_gold=rng.random((lens_n, L)), top_is_gold=rng.random((lens_n, L)) > 0.5)
            np.save(D / f"var_{c}.npy", rng.random((lens_n, L)))
            write_jsonl(D / f"preds_{c}.jsonl", [{"id": f"i{i}", "gold": gold[i], "pred": gold[i] if rng.random() < 0.6 else "A"} for i in keep])
    monkeypatch.setattr("sys.argv", ["diagnose", "--test-dir", str(tmp_path / "test"), "--calib-dir", str(tmp_path / "calib"), "--variant", "pure"])
    diagnose.main()
    import csv

    rows = list(csv.DictReader(open(tmp_path / "test" / "diagnosis_pure.csv")))
    drift = {int(r["layer"]): float(r["value"]) for r in rows if r["metric"] == "drift_centred" and r["cond"] == "hi-pure"}
    assert max(drift, key=drift.get) in (3, 4) and drift[0] < 1e-3
    assert set(np.load(tmp_path / "calib" / "steer_pure.npz").files) == {"hi-pure", "bn-pure", "ta-pure", "te-pure"}
