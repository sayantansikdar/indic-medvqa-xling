# Where does a medical VLM lose Indic languages?

Code for the M.Tech dissertation *Layer-wise causal diagnosis and band-targeted recovery of
cross-lingual failure in medical VQA* (NSUT, 2026–27). Primary model: MedGemma 1.5 4B-it.
Target languages: Hindi, Bengali (Indo-Aryan), Tamil, Telugu (Dravidian).

Everything runs from [`notebooks/run_colab.ipynb`](notebooks/run_colab.ipynb) on a Colab L4 or A100.
T4 and V100 runtimes will not work, because Gemma 3 produces inf/NaN values in fp16. Outputs go to Google Drive, and every step resumes where it stopped.

## Pipeline

| Phase | Module | Output |
|---|---|---|
| Data | `xling.data` | `arogyabodha.jsonl`, `vqarad.jsonl`, images, split by image hash (test/calib/train 50/20/30) |
| Translation | `xling.translate` | `*-mt` (IndicTrans2 from English, back-translation filter), `*-tt` (translate-test) |
| A: baseline | `xling.infer` | option-letter probabilities, accuracy, letter mass, tokenizer fertility, reply-language fidelity |
| B1–B4 | `xling.infer --capture`, then `xling.diagnose` | per-layer drift (raw and centred), CKA, language-ID probe, answer-probe transfer, logit lens, visual-attention ratio |
| B5 | `xling.patching`, `xling.band` | per-layer and per-window recovery, controls, critical band B* |
| B3 baseline | `xling.steer` | best steering layer and scale (chosen on calib) |
| C | `xling.train` | band LoRA with alignment loss, plus the random-band, no-alignment and all-layer variants |
| D | `xling.report` | accuracy with 95% CI, gap, CLC, ECE, GRR, English retention, McNemar with Holm correction |

Question **conditions** are named `<lang>-<variant>`:

- `mixed`: ArogyaBodha's translations as released. About 37% of their letters are Latin, because English terms are kept in parentheses.
- `mt`: IndicTrans2 translations of the English source. This is the monolingual condition used for the main analysis.
- `pure`: `mixed` with the parenthesised English removed. It is exploratory only: sometimes the English term is the only mention of the concept, so removing it deletes content.
- `tt`: the question translated back to English (baseline B1).

The `mixed` vs `mt` comparison serves as ablation A4: does keeping the clinical terms in English close the gap?

## Changes from the mid-term design, and why

1. **The band is no longer chosen by whole-residual patching.** The residual after layer ℓ is a running total, so patching it at the last layer reproduces the English output exactly (`tests/test_hooks.py::test_resid_patch_at_last_layer_reproduces_donor`). R_ℓ therefore rises towards the output whatever the model does, and the old H2 threshold could not fail. B* is now chosen on **window patching of the attention and MLP contributions** (the additive outputs of each layer), which do not accumulate with depth. Residual patching is still reported for reference.
2. **Controls added.** A mismatched-donor control (`ctl-win6`: the English donor comes from a different item) and reverse patching (`rev-win6`). Suggested restatement of H2: mean window recovery in B* ≥ 0.5, at least 2× the mismatched-donor control, and at least 2× the median window.
3. **Drift is centred.** Gemma residual streams have a few very large shared dimensions, which push raw cosine similarity close to 1 at every layer. Raw drift is still reported, but the centred version is the one to read.
4. **Evaluation set.** All ArogyaBodha items that exist in all five languages with one image are used, split by image hash, instead of the 130-item test subset. That gives far more statistical power, but the numbers are not directly comparable with the 36% in the ArogyaSutra paper.
5. **bf16 LoRA by default** (`--quant none`). MedGemma 4B fits in bf16 on an L4, which avoids diagnosing in bf16 but repairing in 4-bit. QLoRA is still available with `--quant 4bit`; if you use it, evaluate the baseline in 4-bit too.
6. **Constant English instruction** ("Answer with the letter…") in every condition, so only the question and options change across languages.

## Scope for the 4 December 2026 deadline

Must have: MedGemma with Phase A, B1–B5 and C (ours, A1, A2, B4, B3 steering, B1 translate-test, one seed each) on ArogyaBodha, plus VQA-RAD as the second domain.
If time allows: the Qwen2.5-VL-7B control (diagnosis only), more seeds, and the MLP-adapter ablation (A3).
Moved to the journal version: Lingshu-7B, native-speaker review with inter-annotator agreement, rank and width sweeps.

| Dates | Work |
|---|---|
| 20–21 Nov | data, translation, smoke test, Phase A + cache on calib and test |
| 22–23 Nov | diagnosis (B1–B4), patching on calib → freeze B*, patching on test |
| 24–25 Nov | steering sweep; train ours, A2, A1, B4 |
| 26 Nov | evaluate everything on test (ArogyaBodha + VQA-RAD); report tables |
| 27–28 Nov | Qwen control and extra seeds, if compute allows; error analysis |
| 29 Nov – 3 Dec | paper (IEEE, 4–6 pages) |
| 4 Dec | submit to Confluence 2027 |

## Local checks

```bash
pip install -r requirements.txt torch
pytest -q tests
```

The tests use tiny random Gemma 3 and Qwen2 decoders. They check that the attention and MLP contributions sum to the residual, that patching only touches its own batch row, that the logit lens matches the model's output at the last layer, the band-selection rule, and the statistics. They do not download any models.
