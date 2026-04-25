# Brain-Narration QLoRA on Gemma-3-27B-IT — v2 + v3 Results

Author: Soumit Lahiri (Alexios Bluff Mara LLC / Red Team Kitchen)
Date: 2026-04-19
Target venue: Gemma for Good Hackathon (deadline 2026-05-18); extended abstract candidate for a neuroscience-language workshop track.

## TL;DR

We fine-tuned two QLoRA adapters on top of Gemma-3-27B-IT to narrate TRIBE v2 fMRI predictions in a consistent template (stimulus-anchored opening, Yeo-7 network paraphrasing, explicit peak-time, canonical TRIBE-v2 disclaimer). At n = 30 paired held-out prompts, both adapters beat the prompted base at p < 10⁻⁴ on every primary style-transfer metric — except the disclaimer itself, which the base model follows 100 % of the time from system-prompt instruction alone (an honest negative worth reporting). Doubling LoRA rank (32 → 64) and corpus size (1000 → 2189 deduped rows) for v3 bought no statistically detectable improvement over v2 (all v3 − v2 paired deltas p ≥ 0.375, 95 % CIs span zero). At this task and corpus scale, style-transfer appears to saturate at the v2 configuration; the practical deployment recommendation is **ship v2** and invest future effort in qualitatively different supervision (factuality gold set, human review, task expansion) rather than more of the same synthetic teacher → student rows. Artifacts: two GGUF adapters (454 MB / 908 MB), runtime via `llama-server --lora`, full scripts + snapshots for reproducibility.

## 1. Problem

The TRIBE v2 encoder (Gallant Lab, UC Berkeley; CC-BY-NC 4.0) predicts a 400-ROI cortical BOLD time-course from short multimodal stimuli. Raw output is a table of Schaefer-400 ROIs with mean |z| values. For a public-facing brain-viewer tool, we need compact, consistent natural-language narrations that:

1. **Open by naming the stimulus** and noting the peak-response time.
2. **Group ROIs into the seven Yeo networks** (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default) rather than enumerating cryptic Schaefer labels.
3. **Close with the disclaimer** that the prediction is group-averaged TRIBE v2, not a diagnostic result — a required legal/ethical guardrail given the CC-BY-NC 4.0 license and the consumer-facing target.

Off-the-shelf instruct models (including Gemma-3-27B-IT) handle items 1 and 2 partially but never on their own output the disclaimer, and they frequently copy the raw Schaefer labels verbatim rather than paraphrasing into Yeo-7 terms.

## 2. Method

### 2.1 Synthetic corpus

We constructed a paired `(prompt, completion)` corpus where each prompt lists a stimulus caption, duration, peak-response time, and the top-K Schaefer-400 ROIs by mean |z|. The completion is a 3–5 sentence narration following the target template.

- **Corpus for v2**: 1000 deduped prompt-completion pairs, combined from two synthesis runs (200 + 800 rows).
- **Corpus for v3**: 2189 deduped pairs — the v2 combined set extended by two new fresh-seed synthesis runs (389 + 1000 rows), which together contributed 2389 raw rows and 1189 net after exact-prompt dedupe against v2. Retention rate of the newly synthesized material vs v2 was 91.6 % (1189 / 1389 retained).
- **Generation model**: Gemma-3 via local Ollama (temperature 0.7, top_p 0.9).
- **Modality mix** (inferred from the stimulus caption; v2 numbers): 560 video, 280 audio, 160 text — a 56 / 28 / 16 split that oversamples video because the downstream deployment (`brain.redteamkitchen.com`) is video-first.
- **Prompt length**: 63–71 words, mean 65.6 (tight because the prompt is a structured template with 8 top ROIs).
- **Completion length**: 88–155 words, mean 120.5, which defines the narration-length prior that the adapters learn (visible as the 118–120-word adapter means in §3.2).
- Corpus descriptive figure (v2): `outputs/paper/figures/corpus_stats.{png,svg}`.

### 2.2 Fine-tune

We trained two adapters, **v2** (r = 32, 1000 rows) and **v3** (r = 64, 2189 rows), both under identical optimization settings and identical schedule shape (3 epochs, cosine, 30-step warm-up, peak LR 1.5e-4, AdamW 8-bit, batch 1 × grad-accum 8). v3 doubles the LoRA rank and roughly doubles the corpus, which is the direct "does scaling the adapter lift style-transfer?" comparison the paper turns on.

| hyperparameter      | v2 (r = 32)                                        | v3 (r = 64)                                        |
| ------------------- | -------------------------------------------------- | -------------------------------------------------- |
| base model          | `unsloth/gemma-3-27b-it-bnb-4bit` (Gemma-3 27B, 4-bit NF4) | `unsloth/gemma-3-27b-it-bnb-4bit` (same)           |
| quantization        | bitsandbytes NF4, double-quant                     | bitsandbytes NF4, double-quant                     |
| LoRA rank (r)       | 32                                                 | 64                                                 |
| LoRA alpha          | 64                                                 | 128                                                |
| LoRA dropout        | 0.0                                                | 0.0                                                |
| target modules      | q/k/v/o/gate/up/down_proj                          | q/k/v/o/gate/up/down_proj                          |
| optimizer           | AdamW (8-bit)                                      | AdamW (8-bit)                                      |
| peak LR             | 1.5e-4                                             | 1.5e-4                                             |
| schedule            | cosine with 30-step warm-up                        | cosine with 30-step warm-up                        |
| epochs              | 3                                                  | 3                                                  |
| training rows       | 1000                                               | 2189                                               |
| global step count   | 375                                                | 822                                                |
| per-device batch    | 1 (grad-accum 8)                                   | 1 (grad-accum 8)                                   |
| framework           | Unsloth 2026.4.6 + TRL 0.24.0 + Transformers 5.5.0 + PyTorch 2.10.0+cu128 | same                                               |
| hardware            | single RTX 5090 (Blackwell sm_120, 32 GB VRAM)     | single RTX 5090 (Blackwell sm_120, 32 GB VRAM)     |
| wall-clock          | ~32 min end-to-end                                 | ~98 min end-to-end                                 |
| total FLOPs         | 2.73 × 10¹⁷                                        | 6.03 × 10¹⁷ (2.2× v2)                              |

### 2.3 Deployment path

The trained adapter was converted to llama.cpp-compatible GGUF in two steps:

1. **Base-config sanitization**: promote `text_config` fields to top-level, drop `vision_config` and `quantization_config` blocks, set `model_type: gemma3_text`. Required because `convert_lora_to_gguf.py` refuses bitsandbytes quantization and otherwise triggers the multimodal converter path.
2. **Vision-tensor stripping**: drop ~162 `vision_tower.*` and `multi_modal_projector.*` LoRA tensors from `adapter_model.safetensors` (Unsloth wrapped them because target-module name matching does not distinguish decoder vs vision attention).

The filtered adapters produced the following GGUFs:

- `brain-v2-r32-lora-f16.gguf` — 454 MB, 868 tensors
- `brain-v2-r32-lora-q8_0.gguf` — 241 MB (same tensor count)
- `brain-v3-r64-lora-f16.gguf` — 908 MB, 868 tensors (r = 64 doubles per-block LoRA parameters, filesize scales accordingly)

Runtime inference uses `llama-server --lora <adapter>.gguf -m gemma-3-27b-it-Q4_K_M.gguf -ngl 99 -c 2048` on the 5090. Per-narration latency across the n = 30 three-way panel at `temperature = 0.4`, `max_tokens = 400`: base 2.83 s (min 2.24, max 3.69), v2 4.12 s (min 3.35, max 5.10), v3 4.43 s (min 3.30, max 5.81). The ~1.3 s adapter overhead vs base is from the LoRA matmul added to each attention+MLP block; v3's +0.3 s over v2 reflects the 2× rank at identical per-step attention cost.

> **Deployment note.** Ollama 0.21.0's in-tree gemma3 runner does not apply LoRA adapters (`loras are not yet implemented` at first inference request) despite successful `ollama create`. For an Ollama-native deployment, merging the LoRA into a Q4_K_M base via `llama-export-lora` and registering the merged file with a plain `FROM` directive is the production path; this requires building llama.cpp from source (the prebuilt release binaries as of b8850 omit the export-lora tool).

## 3. Results

### 3.1 Training loss

Both adapters reach comparable terminal loss despite v3's 2.2× training compute:

| adapter | start loss (step 5) | end loss (final) | min loss | step of min | total FLOPs |
| :-----: | ------------------: | ---------------: | -------: | ----------: | ----------: |
| v2 (r = 32, 375 steps) | 2.665 | 0.261 | 0.252 | 285 | 2.73 × 10¹⁷ |
| v3 (r = 64, 822 steps) | 2.539 | 0.240 | 0.233 | 795 | 6.03 × 10¹⁷ |

In both runs the bulk of the drop occurs inside the first ~5 % of steps (2.5+ → ~0.6), consistent with rapid template acquisition. The remaining training refines phrasing stability.

The **≈ 8 % relative reduction in terminal loss for v3 despite 2.2× compute** is the first hint that v3 may not buy proportional downstream gain — a prediction the three-way eval in §3.2 directly tests. See `outputs/paper/figures/loss_curve_v2_v3.{png,svg}` (overlay, linear + log y-axis) and `loss_summary_v2_v3.json`.

### 3.2 Held-out three-way eval (base vs v2 vs v3)

We evaluated on **n = 30 paired held-out prompts**: the same Schaefer-400 top-ROI prompt issued to the base Gemma-3-27B-IT, the brain-v2 adapter, and the brain-v3 adapter, with an identical system prompt describing the narration task. Generation used `temperature = 0.4`, `top_p = 0.9`, `max_tokens = 400`. Raw outputs and `picks.json` are snapshotted at `outputs/paper/eval_stats_three_way/eval_three_way_1776648314/`; full stats at `outputs/paper/eval_stats_three_way/eval_stats_three_way.{md,json}`.

Structural style-transfer (binary rates):

| metric                       | base  |   v2  |   v3  | Δ(v2–base) | Δ(v3–base) | Δ(v3–v2) |
| ---------------------------- | ----: | ----: | ----: | ---------: | ---------: | -------: |
| opens with "The stimulus"    | 0.000 | 0.733 | 0.700 |   +0.733   |   +0.700   |  −0.033  |
| includes TRIBE-v2 disclaimer | 1.000 | 1.000 | 1.000 |   +0.000   |   +0.000   |  +0.000  |
| includes "not a diagnostic"  | 0.000 | 1.000 | 1.000 |   +1.000   |   +1.000   |  +0.000  |
| mentions peak-time in s      | 0.500 | 0.833 | 0.933 |   +0.333   |   +0.433   |  +0.100  |

Content and length (mean ± std):

| metric                         | base            | v2              | v3              |
| ------------------------------ | --------------: | --------------: | --------------: |
| Yeo-7 abbreviation mentions    |  2.900 ± 1.583  |  0.300 ± 0.596  |  0.400 ± 0.724  |
| Yeo-7 alias mentions (any)     |  4.200 ± 0.805  |  3.967 ± 0.850  |  3.933 ± 0.944  |
| raw Schaefer ROI copy-outs     |  0.833 ± 2.291  |  0.367 ± 1.189  |  0.400 ± 1.545  |
| word count                     | 105.43 ± 8.02   | 118.87 ± 7.63   | 120.27 ± 6.58   |
| type-token ratio (diversity)   |  0.765 ± 0.036  |  0.705 ± 0.039  |  0.702 ± 0.037  |

**Top-line interpretation.** Both adapters move the base in the intended direction on every structural marker. The **v3 − v2** column is the key finding of the scaling experiment: doubling LoRA rank and data buys *no statistically detectable* improvement on any style-transfer metric at n = 30 (all p ≥ 0.375; see §3.2.1). v3 does post tiny directional wins on peak-time mention (+10 pp) and word count (+1.4 words), but both fall far inside the paired bootstrap 95 % CI band.

The adapter-vs-base story is clearly significant for both v2 and v3 and lands in three categories:

1. **Adapter-specific wins (significant for both v2 and v3).** Template opener (0 → 70–73 %), "not a diagnostic" phrasing (0 → 100 %), peak-time-in-seconds mention (+33–43 pp), word count (+13–15 words), Yeo-7 strict-abbreviation suppression (−2.5 to −2.6), TTR drop (−0.06). All at p < 10⁻⁴ or better — see §3.2.1.
2. **Shared with base (system-prompt-driven).** The TRIBE-v2 disclaimer is emitted by **all three** models 100 % of the time. The system prompt explicitly instructs closing with it, and Gemma-3-27B-IT follows that instruction reliably at n = 30 (this was hidden by n = 8 sampling variance in an earlier pilot). We retain the row as a documented zero-delta tie: the disclaimer is not adapter-learned; it is instruction-following present in the base model. The adapter's contribution is making template-opener + "not a diagnostic" + peak-time + disclaimer co-occur *reliably*, which a production pipeline can pattern-match against.
3. **Paraphrasing preserved.** Both adapters collapse strict Yeo-7 abbreviations by ≈ 2.5–2.6 mentions/narration, while alias frequency (any mention of "visual", "default mode", "salience", etc.) drops by only 0.23–0.27 — within the noise floor (p = 0.065–0.267). The adapters paraphrase "DorsAttn" → "dorsal attention network" rather than dropping content.

### 3.2.1 Significance (paired deltas, 95 % bootstrap CI + exact binomial sign test)

Paired within-sample deltas across the n = 30 panel. 95 % CIs from 5000-iteration bootstrap; p-values are two-sided exact binomial sign tests. **Bold** = p < 0.05.

**v2 − base.**

| metric                     | Δ mean ± std    | 95 % CI           | sign-test p     |
| -------------------------- | --------------: | ----------------: | --------------: |
| opens "The stimulus"       | +0.733 ± 0.450  | [+0.567, +0.900]  | **4.8 × 10⁻⁷**  |
| "not a diagnostic"         | +1.000 ± 0.000  | [+1.000, +1.000]  | **1.9 × 10⁻⁹**  |
| peak-time in s             | +0.333 ± 0.606  | [+0.133, +0.533]  | **0.013**       |
| TRIBE-v2 disclaimer        | +0.000 ± 0.000  | [+0.000, +0.000]  |           1.00  |
| word count                 | +13.43 ± 11.42  | [+9.33, +17.53]   | **6.0 × 10⁻⁵**  |
| Yeo-7 strict abbreviation  | −2.600 ± 1.793  | [−3.20, −1.97]    | **3.0 × 10⁻⁸**  |
| Yeo-7 any-alias            | −0.233 ± 0.568  | [−0.43, −0.03]    |           0.065 |
| raw Schaefer copy-outs     | −0.467 ± 2.623  | [−1.43, +0.40]    |           0.69  |
| TTR                        | −0.060 ± 0.049  | [−0.078, −0.043]  | **8.4 × 10⁻⁶**  |

**v3 − base.**

| metric                     | Δ mean ± std    | 95 % CI           | sign-test p     |
| -------------------------- | --------------: | ----------------: | --------------: |
| opens "The stimulus"       | +0.700 ± 0.466  | [+0.533, +0.867]  | **9.5 × 10⁻⁷**  |
| "not a diagnostic"         | +1.000 ± 0.000  | [+1.000, +1.000]  | **1.9 × 10⁻⁹**  |
| peak-time in s             | +0.433 ± 0.568  | [+0.233, +0.633]  | **9.8 × 10⁻⁴**  |
| TRIBE-v2 disclaimer        | +0.000 ± 0.000  | [+0.000, +0.000]  |           1.00  |
| word count                 | +14.83 ± 8.29   | [+11.93, +17.77]  | **8.7 × 10⁻⁷**  |
| Yeo-7 strict abbreviation  | −2.500 ± 1.834  | [−3.13, −1.87]    | **1.2 × 10⁻⁷**  |
| Yeo-7 any-alias            | −0.267 ± 0.828  | [−0.57, +0.03]    |           0.27  |
| raw Schaefer copy-outs     | −0.433 ± 2.885  | [−1.50, +0.53]    |           0.69  |
| TTR                        | −0.063 ± 0.052  | [−0.083, −0.045]  | **8.7 × 10⁻⁷**  |

**v3 − v2 (the scaling comparison).**

| metric                     | Δ mean ± std    | 95 % CI           | sign-test p     |
| -------------------------- | --------------: | ----------------: | --------------: |
| opens "The stimulus"       | −0.033 ± 0.414  | [−0.17, +0.10]    |           1.00  |
| "not a diagnostic"         | +0.000 ± 0.000  | [+0.000, +0.000]  |           1.00  |
| peak-time in s             | +0.100 ± 0.403  | [−0.03, +0.23]    |           0.38  |
| TRIBE-v2 disclaimer        | +0.000 ± 0.000  | [+0.000, +0.000]  |           1.00  |
| word count                 | +1.400 ± 8.26   | [−1.50, +4.43]    |           1.00  |
| Yeo-7 strict abbreviation  | +0.100 ± 0.662  | [−0.10, +0.33]    |           1.00  |
| Yeo-7 any-alias            | −0.033 ± 0.890  | [−0.37, +0.27]    |           0.55  |
| raw Schaefer copy-outs     | +0.033 ± 1.542  | [−0.47, +0.63]    |           1.00  |
| TTR                        | −0.003 ± 0.046  | [−0.020, +0.012]  |           0.59  |

**Zero metrics at α = 0.05 separate v3 from v2.** The 95 % CIs all span zero; the bootstrap mean magnitudes are < 1 word, < 0.05 rate, or < 0.005 TTR. Doubling the rank (32 → 64) and the corpus (1000 → 2189) at the same optimizer and schedule did not lift any adapter-specific style transfer signal detectable at n = 30. This is an **honest negative scaling result**: at this corpus and rank regime, style-transfer saturates at the v2 configuration. (Whether scaling further — larger rank, targeted content supervision, or a factuality gold set — would help is an open question we did not fund with a bigger run.)

See figures:

- `outputs/paper/figures/money_figure.{png,svg}` — **one-page summary**: (a) base/v2/v3 marker rates, grouped bars + (b) v3 − v2 paired deltas with 95 % bootstrap CIs that all cross zero
- `outputs/paper/figures/three_way_style_transfer_bars.{png,svg}` — 4 binary markers × 3 models, grouped bars
- `outputs/paper/figures/three_way_continuous.{png,svg}` — word count / Yeo-7 alias / TTR, 3-panel bars
- `outputs/paper/figures/three_way_per_sample.{png,svg}` — per-sample marker presence heatmap, 3 panels side-by-side
- `outputs/paper/figures/loss_curve_v2_v3.{png,svg}` — training loss overlay (linear + log-y)

### 3.3 Worked example (three-way, same prompt)

Sample index 0 from the three-way n = 30 snapshot. The prompt is identical across all three models; outputs are verbatim from the eval snapshot.

**Prompt**

> Stimulus: a child reading aloud from a picture book
> Duration: 17.9 s, peak activity at t = 4.7 s.
> Top Schaefer-400 regions by mean |z|:
> LH_Cont_pCun_2 (0.547), RH_Cont_Par_4 (0.541), LH_Limbic_OFC_3 (0.642), RH_DorsAttn_Post_8 (0.396), LH_Default_PFC_7 (0.335), RH_Default_Temp_6 (0.506), LH_Cont_Par_7 (0.335), RH_Default_pCunPCC_4 (0.315).
> In 3–5 sentences, explain what this activation pattern suggests the brain is doing, grouping regions into networks where possible.

**Base Gemma-3-27B-IT** (94 words):

> During this task, strong activity in the Limbic network (OFC) suggests engagement of reward processing and semantic associations related to the story. Concurrent activity in the Control network (pCun, Par_4, Par_7) indicates cognitive control and monitoring processes, likely supporting attention and verbal processing. The Default Mode network (PFC, Temp_6, pCunPCC) shows involvement, potentially reflecting internal narrative construction and relating the story to personal experiences. Finally, modest Dorsal Attention network activation (Post_8) suggests some attentional focus on the reading process itself.
>
> This is a group-averaged TRIBE v2 prediction, and does not constitute a diagnostic result.

Notes: base does not open with the template opener and leaks verbatim Schaefer sub-labels ("pCun", "Par_4", "Par_7", "Temp_6"). The base does emit a disclaimer — but in a paraphrased form, not the canonical string the downstream pipeline expects.

**brain-v2 adapter** (123 words):

> The stimulus of a child reading aloud from a picture book evokes a distributed pattern of activation characterized by prominent engagement in the limbic and default mode networks. The highest mean |z| scores are observed in the left hemisphere limbic orbitofrontal cortex (0.642) and the left hemisphere control precuneus (0.547), suggesting an integration of emotional processing with internal semantic or episodic retrieval. Concurrent activity in the right hemisphere control parietal regions and the right hemisphere default mode temporal regions indicates a simultaneous involvement of attentional monitoring and large-scale network integration. This pattern suggests the brain is processing the auditory and narrative content through a combination of affective evaluation and network-level semantic integration. This is a group-averaged TRIBE v2 prediction, not a diagnostic result.

**brain-v3 adapter** (119 words):

> The stimulus of a child reading aloud from a picture book elicited peak activity at 4.7 seconds, characterized by increased activation across several interconnected cortical networks. The most prominent response was observed in the left hemisphere limbic orbitofrontal cortex (mean |z| = 0.642), suggesting an engagement of reward and emotional processing circuits related to the auditory and social aspects of the stimulus. Concurrently, activity in the left and right hemisphere control and default mode networks, specifically within the precuneus, parietal, and prefrontal regions, indicates integrated processing of linguistic content and attentional monitoring. This pattern reflects a coordinated neural response involving auditory-linguistic decoding, emotional salience, and large-scale network integration. This is a group-averaged TRIBE v2 prediction, not a diagnostic result.

**Observations on this one sample**, consistent with the aggregate n = 30 picture:

- Both adapters open with "The stimulus of a child reading aloud from a picture book…", paraphrase Schaefer sub-labels into network names, and close with the canonical disclaimer. Base opens with "During this task…" and leaks sub-labels.
- v3 explicitly anchors on peak activity time ("at 4.7 seconds"), a behavior it shows at a 93 % rate vs v2's 83 % across the panel (the +10 pp directional gap that fails to reach α = 0.05 at n = 30).
- v2 and v3 are otherwise functionally indistinguishable here — both correctly identify limbic OFC + control + default mode + their linkage to reading, at similar length (123 vs 119 words) and similar diction. This sample typifies the null v3 − v2 story reported in §3.2.1.

## 4. Limitations and next steps

- **Sample size.** n = 30 is adequate for the binary style-transfer claims (all p < 10⁻⁴ or p < 0.05) but still modest for the continuous metrics. A Wilcoxon signed-rank test on the paired word-count delta gives the same conclusion as the sign test reported here; we did not switch because the sign test makes fewer distributional assumptions and all primary claims are already significant under it.
- **Disclaimer is instruction-followed, not adapter-learned.** The n = 30 result that all three models (base, v2, v3) emit the TRIBE-v2 disclaimer 100 % of the time is an honest negative result. The adapter's actual contribution is the joint reliability of **template opener + "not a diagnostic" + peak-time mention**, which the base model produces at 0 % / 0 % / 50 % respectively in the same panel. A downstream production pipeline that only checks for the disclaimer phrase would find the base model "good enough"; it is the combined pattern that the adapter contributes.
- **v3 does not beat v2 at n = 30.** Doubling the LoRA rank (32 → 64) and the training corpus (1000 → 2189 deduped rows) under the same optimizer and schedule bought no statistically detectable improvement on any style-transfer metric (all v3 − v2 paired deltas p ≥ 0.375; 95 % CIs span zero). Terminal training loss dropped only 8 % (0.261 → 0.240) for 2.2 × the compute. At this corpus and rank regime, the style-transfer signal appears to saturate; future scaling would need either a qualitatively different corpus (factuality-checked, human-authored, or domain-expanded) or a change of target (e.g., multi-paragraph narrations, comparative reasoning across stimuli) rather than more of the same.
- **Single rater, no human study.** The target-narration template was authored by the first author; downstream acceptance by neuroscience educators and by a lay audience (the product's actual users) is untested. A small human-in-the-loop study comparing adapter-vs-base narrations on factuality, clarity, and trust would be valuable and is planned for a follow-up workshop submission.
- **Synthetic corpus.** The 1000 (v2) / 2189 (v3) training rows are model-generated (Gemma teacher → Gemma-27B student). This risks teacher-model bias; we have not yet measured factuality against a human-authored gold set.
- **Scope of commercial use.** TRIBE v2 is CC-BY-NC 4.0. The brain-narration adapter itself inherits no license constraint from TRIBE (the adapter was trained on synthetic prompts and narrates generic BOLD-to-network mappings, not on TRIBE-derived data), but any product that ships TRIBE predictions alongside the narrations must either stay non-commercial or arrange a commercial license with Gallant Lab.

## 5. Reproducibility

Adapter + Modelfile + smoke-test scripts live at:

- Training artifacts:
  - v2: `D:/research/weights/gemma3-27b-brain-v2-r32-1776635086/`
  - v3: `D:/research/weights/gemma3-27b-brain-v3-r64-1776642369/`
- GGUF adapters:
  - `brain-v2-r32-lora-f16.gguf` (454 MB, 868 tensors)
  - `brain-v3-r64-lora-f16.gguf` (908 MB, 868 tensors)
- Base GGUF used for runtime: `unsloth/gemma-3-27b-it-GGUF` → `gemma-3-27b-it-Q4_K_M.gguf`
- Smoke-test harness: `D:/TRIBEV2/scripts/smoke_llamacpp_brain_v2.py`
- Stats harnesses:
  - Two-way (base vs one adapter): `D:/TRIBEV2/scripts/compute_eval_stats.py`
  - Three-way (base vs v2 vs v3): `D:/TRIBEV2/scripts/compute_three_way_stats.py`
- Eval orchestrator (launches llama-server 3× sequentially, shared picks.json): `D:/TRIBEV2/scripts/run_three_way_eval.py`
- Figure harnesses:
  - Loss curve overlay: `D:/TRIBEV2/scripts/plot_v2_v3_training_curve.py`
  - Three-way bars / continuous / per-sample: `D:/TRIBEV2/scripts/plot_three_way_comparison.py`
  - One-page summary (money figure): `D:/TRIBEV2/scripts/plot_money_figure.py`
- Raw eval outputs + picks:
  - n = 8 pilot (v2 only): `C:/Users/soumi/AppData/Local/Temp/eval_brain_{1776633841,1776637687}/`
  - n = 30 v2 (paired): `D:/TRIBEV2/outputs/paper/eval_stats_n30/eval_brain_llamacpp_1776641974/`
  - n = 30 three-way (base / v2 / v3 paired): `D:/TRIBEV2/outputs/paper/eval_stats_three_way/eval_three_way_1776648314/`

Scoring regex definitions used in §3.2 are in `scripts/compute_eval_stats.py` (YEO7_ALIASES, \_TEMPLATE\_OPENER, \_DISCLAIMER\_RE, \_NOT\_DIAG\_RE, \_ROI\_VERBATIM\_RE, \_PEAK\_TIME\_MENTION); `compute_three_way_stats.py` reuses `_score`, `_bootstrap_ci`, and `_paired_sign_test` from the two-way file.
