# HOVER v10 Extension: VAU-Bench

## Implementation brief for Claude Code

**Objective:** Add VAU-Bench as a third, strictly isolated evaluation surface to the existing HOVER v10 implementation in `HOVER.py` / `HOVER_v10.py` (use the actual supplied filename and repository layout). Implement the requested tasks and metrics, add tests, and preserve existing VideoGlitchBench and UAL-Bench behavior.

**Critical project-wide judge rule:** Use **`meta-llama/Llama-3.1-8B-Instruct` as the single judge model across all HOVER backbones and all datasets**, following the existing HOVER configuration. Do not switch VAU reasoning evaluation to DeepSeek. This makes the judge consistent within our project, but it means our VAU-Eval scores are **not an exact reproduction of the paper's DeepSeek-V3-judged scores**. Label this distinction explicitly in manifests, logs, tables, and any report. Published DeepSeek-scored values may be quoted as reference only; do not describe our Llama-judged scores as directly judge-matched.

## 1. Non-negotiable safety and scope

1. Read the supplied HOVER source and this specification before editing. Reuse existing dataset-isolation, cache, API-client, video preprocessing, scoring, and testing patterns where appropriate.
2. VAU support must be **additive and namespaced**. Do not refactor or change the behavior of existing VideoGlitchBench or UAL-Bench paths.
3. Do not change existing model/backbone definitions or legacy defaults. Add VAU-specific constants, dataclasses, functions, routing, CLI options, and tests only. Make the smallest necessary additive edits to register the new command group and route it safely.
4. Never call legacy `load_records()` or `load_splits()` from a VAU command.
5. Never write VAU outputs, caches, manifests, or logs into VideoGlitchBench data/work directories, UAL-Bench data/work directories, `HOVER_v8_work/`, the GliDe repository, or the source dataset directories.
6. No training, fine-tuning, or hidden adaptation. HOVER remains training-free.
7. Do not scrape or silently substitute other datasets. Load the Hugging Face mirror using `datasets.load_dataset("7xiang/VAU-Bench")`, inspect its real schema/configs/splits, and fail loudly on unsupported/missing data.
8. Do not claim paper parity unless the dataset split, prompts, frame sampling, metric definitions, and judge protocol actually match. Record every deviation.

## 2. Approved backbones and judge

Only these three VLM backbones are allowed for VAU commands:

| CLI key | Required model ID |
|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `internvl2.5-8b` | `OpenGVLab/InternVL2_5-8B` |
| `ui-tars-1.5-7b` | `ByteDance-Seed/UI-TARS-1.5-7B` |

Reject every other backbone or a `--model` override that does not match the selected key. In particular, reject `qwen2.5-vl-3b` and `internvl2.5-4b` for VAU. Use the same refusal style as UAL-Bench.

**One judge for every backbone and dataset:**

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Reuse HOVER's existing judge configuration/client conventions where possible.
- Default temperature: `0.0` for the VAU judge unless the existing shared judge protocol requires a fixed project-wide value; make the chosen value explicit and identical across all runs. Prefer deterministic scoring.
- API endpoint/key must be configurable and use the same endpoint for all backbone × dataset combinations in a comparison.
- Do not use DeepSeek-V3 or `deepseek-chat` as the default or as an automatic fallback.
- Cache judge outputs with a key that includes judge model, judge prompt/version, temperature, ground truth, and prediction. Cache under the VAU work root only.
- Because the paper uses DeepSeek-V3 for VAU-Eval, mark our Llama-judged VAU-Eval results as **project-consistent judge results, not exact official judge parity**.

## 3. Dataset inspection and split policy

Load `7xiang/VAU-Bench` with the Hugging Face `datasets` library. Add `inspect_vau` to print and save:

- available configs, splits, features, and example schema;
- item/video counts by split and source subset;
- the presence and format of video bytes/paths;
- QA option counts and labels;
- TAG timestamps and normal/anomaly conventions;
- classification labels and reasoning/analysis fields;
- comparison with the paper's reported **929 test videos** and **3,700 temporal annotations**, if those counts apply to the exposed mirror schema.

Do not assume the mirror matches the paper. Save discrepancies to `manifest.json` and show them in result headers. Determine the actual split mapping from inspection; do not guess it. Use the test split for headline evaluation. If the mirror has no unambiguous test split, or cannot map the paper's required subsets (MSAD, UCF-Crime, ECVA) reliably, stop with a clear error rather than silently substitute data. Report MSAD, UCF-Crime, and ECVA separately; never merge them into one number unless a clearly labelled additional aggregate is explicitly requested.

If videos are not embedded or accessible from the dataset records, support `--vau_video_root`. Refuse missing videos by default. Only `--vau_allow_missing_videos` may allow a missing video to be represented as a null prediction, and its metric treatment must be explicitly tested and documented.

## 4. Isolation, output paths, and cache keys

Add a VAU-specific work-root resolver parallel to UAL's resolver. Default to a new sibling work directory such as `HOVER_v10_vau_work`; also support `--vau_work_root`.

All VAU writes must remain under:

```text
<work_root>/vau_bench/
  inspect/
  <task>/<backbone>/<split>/
    raw_outputs.json
    metrics.json
    manifest.json
  <task>/<backbone>/cache/
  cache/judge/<judge_model>/
  windows/<preprocessing_fingerprint>/<video_key>/
  summary/<task>/<backbone_set>/
    results.md
    results.csv
    results.json
```

Add a path-containment guard and test every write path. Use a separate `VAUKeyMixin` (or equivalent) and a unique namespace. Cache keys must include dataset/repository and revision, split, task, prompt/adapter version, HOVER version, preprocessing fingerprint, backbone/model, scoring parameters, generation parameters, and judge parameters where relevant. Never reuse UAL or VideoGlitchBench cache entries.

Record the complete command line, dataset repo/revision, split mapping/counts, task, model IDs, judge model/endpoint identifier (never secrets), prompt versions, frame-sampling parameters, generation settings, missing-video policy, anomaly margin, cache statistics, and log-probability truncations in the manifest.

## 5. Task data structures and leakage prevention

Add VAU-specific immutable records parallel to UAL's data structures:

```python
@dataclass(frozen=True)
class VAUQuery:
    task: str
    item_id: str
    video_id: str
    video_file: str
    question: str
    options: tuple = ()
    meta: tuple = ()

@dataclass(frozen=True)
class VAUGroundTruth:
    item_id: str
    task: str
    answer: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    rationale: str | None = None
    anomaly_type: str | None = None
    source: str = ""
```

Adapt typing syntax to the project's supported Python version. **Never place ground-truth labels, timestamps, rationales, or answer strings in `VAUQuery` or any inference prompt.** Add self-tests that inspect all prompts sent to the mock VLM and verify no GT leakage.

Normalize each task without guessing fields. Preserve raw records for audit. For QA, require exactly four options if that is what the actual mirror/paper schema specifies; preserve option order and raise a schema error if the option count differs. For TAG, use the paper's normal convention `[-1.0, -1.0]` only after confirming it matches the actual dataset. For classification, validate the paper's 19 anomaly types plus `Normal` against the real schema before using a fixed list. Unknown labels must fail loudly, not be silently remapped.

## 6. Tasks and inference

Implement four tasks: `qa`, `tag`, `reason`, and `cls`, plus `all`.

### 6.1 Shared V-stage presence prompt

Add VAU-only prompt constants and a single source of truth. Use the domain-neutral presence prompt below for calibrated per-window verification:

```text
You are a video analyst. You are given one stitched window of consecutive video frames, each labelled '#<frame index>' in temporal order (left to right, top row then bottom row), and a query about one specific anomaly. Decide whether that anomaly is visibly occurring in this window. Being plausible is not enough; it must be observable.
```

User template:

```text
Query: {query}
Is the anomaly described by this query visibly occurring in this window? Answer with exactly one word: Yes or No.
```

Keep the original task-instruction strings in a `VAU_TASK_PROMPTS` constant for traceability. Before claiming they are verbatim from the paper, verify them against the paper itself. Do not silently claim a prompt is paper-exact if it was adapted for HOVER.

### 6.2 Temporal Anomaly Grounding (TAG)

- Use the HOVER verification and segmentation primitives through VAU-namespaced wrappers; do not change UAL functions.
- Use a fixed, predeclared, training-free setting: `tau = 0`, `lambda = 1e9` (one segment), and frame-level boundary refinement, if compatible with the existing implementation.
- Add `--vau_anomaly_margin` (default `0.0`) and freeze its value before the evaluation run. A normal prediction must be emitted only under a deterministic, documented rule using the full-video calibrated evidence and this margin. Do not let a best-segment routine force an anomaly on every normal video.
- If evidence does not pass the normal/anomaly decision rule, output `Normal` with `[-1.0, -1.0]`; otherwise output `Anomaly` and a refined interval.
- Preserve timestamps in seconds and verify predictions against decoded video duration.

### 6.3 Multiple-choice QA

For each of the four options, score the candidate with the VAU presence verifier. Let `score(option) = max_t profile(option, t)` over valid windows. Select the option with the highest score; break ties by original option order. No GT answer may affect scoring or tie-breaking.

Report accuracy as a percentage. If the benchmark requires both “without think” and “with think” conditions, implement them as separate prompt configurations and report them separately. Do not label a condition “with think” unless the exact paper instruction and answer extraction are implemented and verified. Record that HOVER itself is training-free in both conditions.

### 6.4 Anomaly classification

Use the verified fixed label set: 19 anomaly types plus `Normal`, only after confirming the actual mirror schema agrees with the paper. Score each candidate label using the same QA scoring rule. Report:

- **Binary accuracy:** normal versus anomaly;
- **Multiclass accuracy:** exact label match.

If the paper requires with/without-think results, compute each separately with independently cached prompt configurations. Do not collapse unknown labels into a convenient class.

### 6.5 Anomaly reasoning

Generate one concise, evidence-grounded explanation using up to `--vau_reason_max_windows` (default 6) windows selected only from model-derived evidence. The output should contain a factual description and analysis, with no timestamps unless the task explicitly requires them. Use a strict JSON schema and robust parser; retain malformed raw outputs for audit.

Evaluate with the **same project-wide Llama-3.1-8B-Instruct judge** for every backbone and dataset. Implement the five VAU-Eval dimensions `CLS`, `KM`, `FLU`, `INF`, and `FAC`, plus `Total`, only after obtaining and verifying the exact rubric/prompt from the paper. Do not fabricate the rubric text. If exact rubric text is not present in the provided materials, leave a clearly marked implementation TODO or require a supplied prompt file rather than inventing a supposedly verbatim prompt. Score parse failures as zero only if that behavior is explicitly chosen and documented; count and report them.

## 7. Frame sampling and generation

Do not reuse VideoGlitchBench's 4 FPS / 8-frame setup by default. Use VAU-specific sampling maps and log them.

Initial requested policy to validate against the paper and supported video APIs:

- Qwen2.5-VL-7B-Instruct: 1 FPS;
- UI-TARS-1.5-7B: 1 FPS;
- InternVL2.5-8B: 16 uniformly sampled frames per video.

Use the existing stitched-grid/window machinery only where compatible. Make window construction deterministic and preserve a mapping from sampled frames to source-video timestamps. Avoid treating a 16-frame whole-video budget as automatically equivalent to 16 frames per window; implement and document the intended semantics carefully.

Use greedy generation (`temperature=0`) for scoring tasks. For reasoning, use a fixed maximum token count (default 512) and deterministic decoding where supported. Record all effective server-side and client-side generation settings. Do not claim the settings match the paper until verified.

## 8. Metrics and paper comparison

Implement separate, tested metric functions in the VAU namespace. Never call the UAL official evaluator for VAU because its null conventions and metrics are dataset-specific.

### 8.1 TAG metrics

For anomaly videos, calculate temporal IoU:

`intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))`

`union = (gt_end - gt_start) + (pred_end - pred_start) - intersection`

`IoU = intersection / union` when the union is positive; otherwise follow a documented, tested edge-case convention.

For normal videos, a correct `Normal` prediction receives IoU 0 for the localization mean under the supplied extension plan's convention; an incorrect anomaly prediction also receives IoU 0. Confirm the official evaluator's exact normal/null convention before calling this an exact port.

- `mIoU = 100 × mean(IoU over all test videos)`;
- `R@k = 100 × count(IoU >= k) / N`, for `k ∈ {0.3, 0.5, 0.7}`, where `N` is all evaluated test videos.

Report two decimal places. Do not silently omit null predictions from the denominator. Add hand-computed tests including normal videos, missing-video nulls, zero-length/invalid intervals, and threshold-boundary cases. **Before claiming official metric parity, compare these definitions with the paper's released evaluator.**

### 8.2 QA and classification metrics

- Accuracy: `100 × correct / N`.
- Binary classification accuracy: compare whether prediction and GT are `Normal`.
- Multiclass classification accuracy: exact label match.
- Report each required prompting condition separately. Do not mix subsets or denominators.

### 8.3 Reasoning metrics

Report mean `CLS`, `KM`, `FLU`, `INF`, `FAC`, and `Total`, to two decimals, by dataset subset. Because the judge differs from the paper, label these as **Llama-judged project results** and not direct numerical replications of DeepSeek-judged VAU-Eval.

### 8.4 Published baselines and supplementary statistics

Include published base-model numbers only when the exact backbone/task/subset/condition can be matched and sourced. Label them `Published (paper, base model)`. Do not reproduce or imply reproduction of SFT/RFT rows. Keep official headline metrics separate from supplementary video-cluster bootstrap CIs and paired permutation tests. Mark supplementary statistics as “not reported by the paper”.

## 9. CLI additions

Add an additive VAU command group:

- `inspect_vau`
- `run_vau`
- `evaluate_vau`
- `selftest_vau`

Add VAU-specific flags, following the existing naming conventions:

- `--vau_repo` (default `7xiang/VAU-Bench`)
- `--vau_cache_dir`
- `--vau_task {qa,tag,reason,cls,all}`
- `--vau_split {test}`
- `--vau_work_root`
- `--vau_video_root`
- `--vau_allow_missing_videos`
- `--vau_api_bases` (`backbone=url,...`, parallel to UAL)
- `--vau_judge_api_base`
- `--vau_judge_model` (default `meta-llama/Llama-3.1-8B-Instruct`; reject any other model by default unless an explicit, documented override is deliberately supported)
- `--vau_judge_temperature` (fixed project-wide default; use consistently)
- `--vau_judge_max_tokens` (default 1024)
- `--vau_anomaly_margin` (default `0.0`)
- `--vau_reason_max_windows` (default `6`)
- `--vau_official_repo` (optional path for evaluator parity checks)

Follow the existing `main_ual` routing pattern. VAU commands must refuse incompatible datasets/methods and must never fall through to legacy commands. Legacy/UAL guards may receive only the smallest additive dataset rejection needed to prevent accidental routing to VAU. Preserve all pre-existing CLI choices and behavior.

## 10. Summary outputs

Write `results.md`, `results.csv`, and `results.json` under the VAU work root, grouped by task, backbone, and source subset. Each Markdown summary must include:

1. HOVER and adapter versions, exact backbone and judge model IDs, date, dataset revision, split counts, prompt versions, frame sampling, generation settings, anomaly margin, and any data/schema discrepancies.
2. A clear statement: “HOVER is training-free; published SFT/RFT rows are reference values only and were not reproduced.”
3. TAG: `mIoU`, `R@0.3`, `R@0.5`, `R@0.7`.
4. QA: accuracy for each supported think condition.
5. Classification: binary and multiclass accuracy for each supported think condition.
6. Reasoning: `CLS`, `KM`, `FLU`, `INF`, `FAC`, `Total`, explicitly labelled as Llama-judged project results.
7. Published base-model rows only where comparisons are genuinely aligned; explain differences in judge and protocol.
8. Supplementary CIs, permutation tests, call counts, parse failures, logprob truncation rates, missing-video counts, and GT timestamps outside decoded video duration.

Never report a subset that could not be mapped to the correct benchmark subset. Omit it with a clear explanation rather than substitute another dataset.

## 11. Required tests

Implement `selftest_vau` using synthetic fixtures and mocked VLM/judge clients; it must not require downloading VAU-Bench or contacting a remote API.

- **Isolation:** all writes remain under the VAU work root; VideoGlitchBench and UAL work roots remain untouched.
- **Legacy-loader guard:** monkeypatch `load_records` and `load_splits` to raise; VAU commands must still work.
- **Backbone guard:** only the three approved backbone keys and exact model IDs are accepted.
- **Judge consistency:** all backbone × dataset combinations use `meta-llama/Llama-3.1-8B-Instruct`; no DeepSeek fallback. Verify cache keys include judge model and prompt version.
- **Prompt leakage:** GT answers, timestamps, and rationales never appear in VLM prompts.
- **TAG metrics:** hand-computed expected mIoU and recalls, including normal and null cases, agree to numerical tolerance.
- **QA/classification:** accuracy and binary/multiclass accuracy match brute-force fixtures.
- **Reasoning judge:** valid JSON parses correctly; malformed output is counted and handled according to the documented policy.
- **Schema validation:** wrong option counts, unknown labels, missing fields, and ambiguous split mappings fail with actionable errors.
- **Frame sampling:** per-backbone sampling settings match the configured maps and appear in manifests.
- **Path guard:** attempts to write outside `<work_root>/vau_bench/` are rejected.
- **Regression:** run existing `selftest` and `selftest_ual` unchanged and verify they still pass. No existing test, dataset, cache, or output may be rewritten as a side effect.

## 12. Suggested implementation order

1. Inspect the existing HOVER and UAL code; identify reusable primitives and current CLI architecture.
2. Add VAU data inspection and schema validation first; test against the actual HF mirror before implementing inference.
3. Add namespaced records, prompts, preprocessing, cache keys, and V-stage scoring.
4. Implement TAG, QA, classification, and reasoning independently.
5. Implement and test each metric function.
6. Add CLI routing, path guards, manifests, and summary writers.
7. Run `selftest_vau`, then the existing `selftest` and `selftest_ual`.
8. Run `inspect_vau` before any model inference. Review the schema/count report and only then run a one-item smoke test for each approved backbone.
9. Run full evaluation only after schema, isolation, sampling, and metric checks pass.

## 13. Acceptance checklist

- [ ] VAU code is additive and isolated; existing VideoGlitchBench and UAL behavior is unchanged.
- [ ] Only Qwen2.5-VL-7B-Instruct, InternVL2.5-8B, and UI-TARS-1.5-7B are accepted.
- [ ] **Every dataset and backbone uses the same judge: `meta-llama/Llama-3.1-8B-Instruct`.**
- [ ] Results accurately disclose that the judge differs from the paper's DeepSeek-V3 judge.
- [ ] Dataset schema, split, and subset mappings were inspected rather than assumed.
- [ ] No GT leakage into inference prompts.
- [ ] Frame sampling and timestamp mapping are deterministic and documented.
- [ ] TAG, QA, classification, and reasoning metrics have synthetic unit tests.
- [ ] Every write/cache stays inside the VAU work root.
- [ ] `selftest_vau`, existing `selftest`, and `selftest_ual` pass.
- [ ] Summary tables distinguish our results from published values and do not overclaim paper parity.
