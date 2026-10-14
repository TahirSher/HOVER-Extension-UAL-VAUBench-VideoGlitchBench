# HOVER v10 on VAU-Bench: protocol notes

These notes cover the VAU-Bench extension of `HOVER.py`: the commands `inspect_vau`, `run_vau`,
`evaluate_vau` and `selftest_vau`.

The extension is dataset-isolated:

- VideoGlitchBench and UAL-Bench commands, paths, caches and results are unchanged.
- `selftest` (59/59) and `selftest_ual` (72/72) still pass.

## 1. Sources

| What | Where | Verified here |
| --- | --- | --- |
| Paper | Zhu, Chen, Shen & Cun, *VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning*, arXiv:2505.23504 | No. arxiv.org was not reachable from the build environment. |
| Official code and annotations | https://github.com/GVCLab/VAU-R1 (commit `75398807`): `src/evaluation/*.py`, `annotations/test_{msad,ucf_crime,ecva}.csv` | Yes. The metrics are exact ports of these scripts and are cross-checked against them. |
| HF dataset | https://huggingface.co/datasets/7xiang/VAU-Bench (gated) | No. huggingface.co was not reachable from the build environment. |
| Videos | Original MSAD, UCF-Crime and ECVA files | Not redistributed by VAU-Bench. |

The metric definitions come from the official evaluation code, which is the executable form of
the paper's protocol.

The HF loader (`--vau_source hf`) works in two steps:

1. It looks for the official file names (`test_msad.csv`, `test_ucf_crime.csv`,
   `test_ecva.csv`) in the dataset repository.
2. Otherwise it calls `datasets.load_dataset("7xiang/VAU-Bench")` and needs a split whose name
   contains `msad`, `ucf` or `ecva`.

It never guesses the test set from file-name patterns. If the HF layout differs, it stops and
lists the splits and columns it found. In that case use `--vau_annotations_dir <VAU-R1>/annotations`.

## 2. Official metrics (exact ports)

| Task | Official script | Metric |
| --- | --- | --- |
| Multiple-choice QA | `evaluate_qa.py` | Acc = #(`str(pred).lower() == str(Correct Option).lower()`) / #(non-NaN predictions) × 100 |
| Temporal anomaly grounding (TAG) | `evaluation_tag.py` | Over predicted videos whose GT is not (0, 0): IoU = inter / (max end − min start), 0 if `pred_start == pred_end`. mIoU and R@{0.3, 0.5, 0.7} are each reported as `round(x, 4) * 100`. |
| Anomaly classification (CLS) | `evaluation_cls.py` | The prediction is the first of the 20 fixed options found in the answer (inside `<answer>` if present). Binary accuracy (Normal vs any anomaly) and 20-way accuracy are computed over **all** GT rows, so a missing prediction counts as wrong. |
| Anomaly reasoning (VAU-Eval) | `evaluate_reasoning_deepseek.py` | `deepseek-chat` rates (Description, Analysis) against GT (Description, Reasoning) on CLS, KM, FLU, INF and FAC (0–10 each) plus a Total (0–50). The score is the mean over judged videos. |

Port details reproduced from the official code:

- **TAG:** duplicate prediction rows are dropped by raw name (keep the first) before `.mp4` is
  appended, and every lookup returns the first matching row.
- **TAG times:** `parse_html` accepts `<glue>[s, e]</glue>`, then a bare `[s, e]`, then falls
  back to (0, 0). It also handles `mm:ss` and `hh:mm:ss` times.
- **QA:** cells that pandas reads as NaN (`""`, `NA`, `None`, …) leave the denominator.
- **VAU-Eval:** the reply must contain a fenced ```` ```json ```` block, and a missing key counts 0.
  The original script crashes on an unparseable reply. Here the request is retried
  (`--vau_judge_retries`, default 2) and remaining failures are counted and reported.
- **Verbatim text:** the QA prompt (no-think), the CLS question and option list, the two
  reasoning prompts, and the judge prompt and system message are verbatim. `selftest_vau --vau_repo`
  checks this by evaluating the official f-strings with `ast`.

**Parity:** with `--vau_repo <VAU-R1 clone> --vau_repo_python <python with pandas<3>`, every
prediction file is also scored by the unmodified official scripts, and the run aborts on any
difference. The self-test does the same on 300 random rows that include:

- `mm:ss` times and bare brackets
- missing rows and NaN cells
- `-1` rows
- free-text classes

### The official CLS script is broken under pandas ≥ 3

`evaluation_cls.py` stores the list of extracted options with `df.at[i, "CLS answer"] = [...]`.

- **pandas 2.x:** the cell keeps the list.
- **pandas 3.x:** columns are string-typed by default, so the cell becomes a string and
  `pred[0]` is its **first character**. Every 20-way prediction is then scored wrong, with no
  error (a perfect run scores 0% CLS accuracy).

The port implements the intended pandas-2 behaviour. `--vau_repo_python` refuses an interpreter
with pandas ≥ 3.

## 3. Audit of the official test annotations (`inspect_vau`)

| | MSAD | UCF-Crime | ECVA |
| --- | --- | --- | --- |
| rows (videos) | 240 | 290 | 438 |
| TAG skipped, GT (0, 0) | 120 | 150 | 0 |
| TAG scored by the official script | 120 | 140 | 438 |
| … of which GT (−1, −1), IoU ≡ 0 | 0 | **16** | **36** |
| TAG scored, valid GT | 120 | 124 | 402 |
| QA rows with Correct Option `-1` (unanswerable) | 0 | **39** | 0 |
| Correct Option holding option text (never matches) | **8** | 0 | 0 |
| Share of Correct Option = `A` | 68% | 51% | 52% |
| GT (0, 0) but class ≠ Normal | 0 | 27 | 0 |
| GT interval length, mean / median / max (s) | 9.8 / 7.2 / 54 | 18.8 / 7.5 / 230 | 133 / 44 / 1634 |

Consequences for the reported numbers:

1. **Two TAG variants are reported.** `evaluation_tag.py` scores (−1, −1) rows, which can only
   give IoU 0, so the main table uses the official script. Published UCF-Crime TAG numbers are
   multiples of 1/124 (for example, a published Qwen2.5-VL-3B row of 15.32 / 6.45 / 3.23 =
   19/124, 8/124, 4/124), which excludes those 16 rows. That evidence comes from a third-party
   table, not the paper itself. The summary therefore also reports **"TAG, valid GT"**: the same
   code with those rows removed.
2. **Abstaining is not allowed.** QA divides by non-NaN predictions and TAG by predicted videos,
   so abstaining would *raise* QA accuracy and drop hard TAG items. HOVER answers every item. A
   missing or undecodable video counts as a wrong answer: QA answer `X`, TAG span (0, 0), and a CLS
   answer with no option.
3. **Answer-key skew.** "Always A" scores 68% on MSAD and 51% on UCF-Crime QA, so it is reported as
   a reference row. With the real UCF-Crime GT, the trivial span [0, 60] s scores TAG mIoU 11.76,
   and "always Normal" scores 42.41% CLS accuracy.
4. **Audited diagnostics** are reported separately and are not official: QA accuracy without the
   `-1` rows (leading letter of the Correct Option), and CLS accuracy without the `-1` rows.

## 4. HOVER-VAU method (one predeclared rule; nothing fitted on VAU-Bench)

The pipeline runs these stages in order, using window w (8 frames at 2 fps = 4 s, official GliDe
stitching geometry):

| Stage | What it does |
| --- | --- |
| E | Generic anomaly profile: s_E(w) = logit P(Yes \| w, q_E) − logit P(Yes \| grey, q_E). The second term is contextual calibration (Zhao et al., ICML 2021). |
| H | Class hypothesis by Bayesian model averaging over the evidence location z (section 4.1). |
| V | Class-conditioned profile s_ĉ(w) with "Is an anomaly of type ĉ visibly occurring?". The generic profile is used if ĉ = Normal. |
| S | K = 1 segmentation (τ = 0, λ = ∞: Kadane), then masked-frame bisection of the boundary windows (HOVER G, unchanged; 0.5 s resolution). |
| QA | Same BMA as H, with P(z \| V) taken from the profile used by S. |
| R | The official description and analysis prompts as two turns of one conversation, on a storyboard of 8 frames from the localised segment ± 50% context. The system message states HOVER's evidence (segment, class, posterior) and that it may be wrong. |

### 4.1 Class hypothesis by Bayesian model averaging

P(z = w | V) = softmax_w s(w),  P(c | V) = Σ_w P(z = w | V) · P(c | w)

This posterior is exact under three assumptions:

- exactly one window carries the evidence, with a uniform prior over windows;
- windows are conditionally independent given z;
- the VLM's log-odds equal the prior log-odds plus log LR_w.

Under these assumptions P(z = w | V) ∝ LR_w, and contextual calibration estimates the prior term,
so s(w) = log LR_w. They are **modelling assumptions**: VLM log-odds are not calibrated
likelihood ratios.

The sum runs over **Occam's window** (Madigan & Raftery, 1994): the fewest windows holding at
least 99% of the posterior mass, capped at 8. The mass actually kept is reported.

### 4.2 Design choices

- **Option posteriors are not contextually calibrated by default** (ablation flag:
  `--vau_option_calibration`). Within one question the model posterior is already the
  Bayes-coherent quantity. Calibrating against a grey image would also remove legitimate text
  priors and push the CLS decision away from "Normal".
- **Contextual calibration is kept for the profiles.** It gives τ = 0 its meaning ("more likely
  than without the pixels").
- **Letter options are matched case-sensitively.** `a` and `I` are words, not options. Yes/No
  matching keeps the case-insensitive HOVER rule.
- **2 fps** is the `qwen_vl_utils` default used by the official TAG inference. `--fps` changes it,
  and any non-predeclared rule is written to its own `rule_<hash>` directory.

Ablations: `--vau_tag_profile generic`, `--vau_option_calibration`, `--vau_reason_context none`
(official prompts on a whole-video storyboard), `--vau_bma_mass`, `--vau_bma_max_windows`.

### 4.3 Statistics (not official, reported separately)

- 95% bootstrap CIs over videos.
- Paired sign-flip randomisation tests between backbones (exact for ≤ 2^G ≤ B, otherwise Monte
  Carlo with (1 + k)/(1 + B)), with Holm correction per metric.
- No pooling across MSAD, UCF-Crime and ECVA.

## 5. Running it

The official evaluators need `pandas<3 beautifulsoup4 markdown scikit-learn` in the
`--vau_repo_python` environment. HOVER itself needs `numpy scipy opencv-python requests`, plus
`huggingface_hub` or `datasets` for `--vau_source hf`.

```bash
# backbones (one vLLM server each, logprobs enabled)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-logprobs 20
vllm serve OpenGVLab/InternVL2_5-8B --port 8002 --max-logprobs 20 --trust-remote-code
vllm serve ByteDance-Seed/UI-TARS-1.5-7B --port 8003 --max-logprobs 20
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8004 --max-logprobs 20

git clone https://github.com/GVCLab/VAU-R1
python -m venv vau_eval && vau_eval/bin/pip install "pandas<3" beautifulsoup4 markdown scikit-learn

python HOVER.py selftest_vau --vau_repo VAU-R1 --vau_repo_python vau_eval/bin/python
python HOVER.py inspect_vau --vau_annotations_dir VAU-R1/annotations --vau_dataset all \
    --vau_video_root /data/vau_videos --vau_work_root /data/hover_vau_work
# smoke test (2 videos; labelled PARTIAL, written under partial_limit2/)
python HOVER.py run_vau --vau_annotations_dir VAU-R1/annotations --vau_video_root /data/vau_videos \
    --vau_work_root /data/hover_vau_work --vau_dataset msad --backbone qwen2.5-vl-7b --limit 2
# full run, official parity on every file, VAU-Eval with deepseek-chat
export DEEPSEEK_API_KEY=...
python HOVER.py run_vau --vau_source hf --vau_video_root /data/vau_videos \
    --vau_work_root /data/hover_vau_work --vau_dataset all --backbone all \
    --vau_repo VAU-R1 --vau_repo_python vau_eval/bin/python --vau_judge \
    --vau_api_bases qwen2.5-vl-7b=http://localhost:8000/v1,internvl2.5-8b=http://localhost:8002/v1,ui-tars-1.5-7b=http://localhost:8003/v1,qwen2.5-vl-3b=http://localhost:8004/v1
# re-score saved runs (no VLM calls) / score official-format CSVs of another method
python HOVER.py evaluate_vau --vau_annotations_dir VAU-R1/annotations --vau_work_root /data/hover_vau_work \
    --vau_dataset all --backbone all
python HOVER.py evaluate_vau --vau_annotations_dir VAU-R1/annotations --vau_work_root /data/hover_vau_work \
    --vau_dataset ucf --vau_pred_csv qa=out_qa.csv,tag=out_tag.csv,cls=out_cls.csv
```

`--vau_video_root` may contain `msad/`, `ucf_crime/` (class sub-folders are fine; the index is
recursive) and `ecva/`. Alternatively pass `--vau_{msad,ucf,ecva}_videos`.

Each run writes these files under `<work>/vau_bench/<test set>/<backbone>/<rule>/<full|partial_limitN>/`:

| File | Contents |
| --- | --- |
| `raw_outputs.json` | Per-video profiles, posteriors, spans and reasoning (GT-free) |
| `manifest.json` | Model, prompts, rule, hashes and versions |
| `official_eval/*.csv` | Prediction files in the official schema |
| `metrics.json` | Official metrics, parity results and diagnostics |

Summaries are written to `<work>/vau_bench/summary/<rule>/<run>/<sets>__<backbones>/results.{md,csv,json}`.

**Cost.** `inspect_vau` prints `vlm_calls_upper_bound_per_backbone` once videos are present. Per
video the bound is 2T + 2 + 2·8 + 8 + 2 calls, with T = ⌈duration · fps / 8⌉. ECVA videos are long
(GT intervals up to 27 min). `--fps 1` halves T; that change is recorded as a non-predeclared rule.

## 6. Scope and limitations

- **No VAU-Bench numbers are produced or claimed here.** The self-test checks mechanics in a
  simulated world: exact recovery of intervals, classes and answers by a perfect presence oracle,
  isolation, GT-freeness of prompts, and official-script parity. Whether a real VLM verifies
  class-conditioned hypotheses better than it detects anomalies is an empirical question that the
  run answers.
- **H depends on the CLS step.** If ĉ is wrong, the class-conditioned profile can be flat. The
  `generic` ablation measures this.
- **Unvalidated HF layout.** The loader for the HF dataset layout has not been checked against the
  live repository (see section 1).
