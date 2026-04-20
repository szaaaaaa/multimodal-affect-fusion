# Experiment Protocol — Semi-Automatic Research Workflow

## Overview

Semi-automatic research loop for multimodal stress recognition (AMuCS dataset).
Claude Code prepares experiments and analyzes results; user triggers training on Colab.

---

## 1. Task Definition

- **Task**: Arousal State (low/mid/high) + Trend (down/stable/up) multitask sequence classification
- **Labels**: `arousal_state_trend_seq.json` (per-participant z-score quantile discretization)
- **Primary metric**: `val_f1_mean` (mean of state and trend Macro-F1)
- **Secondary metrics**: `val_macro_f1_state`, `val_macro_f1_trend`, `val_balanced_acc_state`, `val_balanced_acc_trend`

---

## 2. Data

### Paths

| Resource | Colab Path | Local Path (Drive sync) |
|---|---|---|
| Aligned features | `/content/drive/MyDrive/AmuCS_experiment/features/aligned` | `G:\我的云端硬盘\AmuCS_experiment\features\aligned` |
| Raw features | `/content/drive/MyDrive/AmuCS_experiment/features` | `G:\我的云端硬盘\AmuCS_experiment\features` |
| Labels | `/content/drive/MyDrive/AmuCS_experiment/labels` | `G:\我的云端硬盘\AmuCS_experiment\labels` |
| Splits | `/content/drive/MyDrive/AmuCS_experiment/splits` | `G:\我的云端硬盘\AmuCS_experiment\splits` |
| Results | `/content/drive/MyDrive/AmuCS_experiment/runs` | `G:\我的云端硬盘\AmuCS_experiment\runs` |

### Modalities

| Modality | Feature dir | Feature dim | Aligned dir |
|---|---|---|---|
| video | `video_clip` | 768 (CLIP ViT-L/14) | `aligned/video_clip` |
| km | `km` | 25 | `aligned/km` |
| telem | `telem` | 109 | `aligned/telem` |

### Modality Combinations (7)

| Type | Combinations |
|---|---|
| Single | video, km, telem |
| Dual | video+km, video+telem, km+telem |
| Triple | video+km+telem |

### Data Splits (2)

| Mode | Description |
|---|---|
| `cross_subject` | Train/val/test by session (different participants) |
| `within_subject` | Same session, temporal 60%/20%/20% |

---

## 3. Experiment Scale

Each experiment round:

```
1 method × 7 modality combinations × 3 seeds × 2 splits = 42 runs
```

Baseline phase: 6 methods × 42 = **252 runs**

---

## 4. Baseline Methods (Frozen)

| Name | Registration | File | Status |
|---|---|---|---|
| EFT | `eft` | `src/models/fusions/eft.py` | Frozen |
| MFT | `mft` | `src/models/fusions/mft.py` | Frozen |
| LFT | `lft` | `src/models/fusions/lft.py` | Frozen |
| Late | `late` | `src/models/fusions/late.py` | Frozen |
| CMA | `cma` | `src/models/fusions/cma.py` | Frozen |
| Gated | `gated` | `src/models/fusions/gated.py` | Frozen |

These files must NEVER be modified after baseline experiments begin.

---

## 5. Training Configuration (Frozen)

All experiments share the same hyperparameters defined in `configs/sweeps/full_ablation.yaml`:

| Parameter | Value |
|---|---|
| d_model | 512 |
| optimizer | AdamW (lr=5e-5, weight_decay=0.01) |
| scheduler | cosine (warmup_epochs=3) |
| grad_clip | 1.0 |
| batch_size | 256 |
| epochs | 40 |
| early_stopping | patience=5, metric=val_f1_mean, mode=max |
| loss | multitask_ce_seq_masked (label_smoothing=0.1) |
| seq_len | 600 |
| stride | 300 |
| seeds | [0, 1, 2] |

These hyperparameters must NOT be changed by Claude Code. Only the user can modify them.

---

## 6. File Permissions

### Writable (Claude Code can create/modify)

| Path | Action | Purpose |
|---|---|---|
| `src/models/fusions/*.py` | New files only | New fusion methods |
| `src/models/encoders/**/*.py` | New files only | New encoders |
| `src/models/fusions/__init__.py` | Modify | Register new modules |
| `src/models/encoders/**/__init__.py` | Modify | Register new modules |
| `configs/sweeps/*.yaml` | New files only | New experiment sweeps |
| `configs/experiments/*.yaml` | Create/modify | Single experiment configs |
| `docs/experiment_results/` | Create/modify | Analysis reports |
| `run_*.ipynb` | Create/modify | Colab notebooks |

### Read-Only (Claude Code must NOT modify)

| Path | Reason |
|---|---|
| `src/core/*` | Frozen infrastructure (runner, types, registry) |
| `src/data/*` | Data pipeline |
| `src/losses/*` | Loss functions |
| `src/metrics/*` | Evaluation metrics |
| `src/models/fusions/eft.py` | Frozen baseline |
| `src/models/fusions/mft.py` | Frozen baseline |
| `src/models/fusions/lft.py` | Frozen baseline |
| `src/models/fusions/late.py` | Frozen baseline |
| `src/models/fusions/cma.py` | Frozen baseline |
| `src/models/fusions/gated.py` | Frozen baseline |
| `configs/sweeps/full_ablation.yaml` | Baseline config (append-only for new tasks) |
| Data splits and labels | Fixed by user |

---

## 7. Decision Criteria

### Effective Improvement

An improvement method is considered **effective** if:

1. Find the **best baseline result**: across all 6 baselines × 7 modality combinations, identify the configuration (method + modality combo) with the highest `val_f1_mean` (3-seed mean)
2. Compare the new method's result **on the same modality combination** against that best baseline
3. If the new method's 3-seed mean `val_f1_mean` > best baseline's 3-seed mean → **effective**

No hard threshold. Report 3-seed mean +/- std for both.

### Reporting

For every experiment round, produce a results table that enables:

- **Cross-method comparison**: same modality combination, different methods (which fusion is best for video+km?)
- **Cross-modality comparison**: same method, different modality combinations (does adding telem help EFT?)

The overall evaluation across all 7 combinations is included as report data but does NOT determine whether an improvement is effective.

---

## 8. Failure Protocol

When an improvement method does NOT outperform the best baseline:

1. **Document** in `docs/experiment_results/{method_name}_report.md`:
   - Method description and hypothesis
   - Results table (vs baseline, all modality combinations)
   - Analysis of why it failed
2. **Preserve** all code (new files stay, not deleted)
3. **Report to user** with analysis and recommendation (try variant / abandon / next direction)
4. **User decides** the next step

---

## 9. Workflow

### Phase 1: Baseline

1. Run 6 baselines (252 runs) via `run_baseline.ipynb`
2. Analyze results, produce `docs/experiment_results/baseline_summary.md`
3. Identify best baseline per modality combination

### Phase 2: Improvement (repeat per direction)

1. Claude Code analyzes baseline results, proposes improvement priority order
2. **User confirms** priority order
3. For each improvement direction:
   - Claude Code implements the method (new files + new sweep config)
   - Claude Code prepares Colab notebook
   - **User triggers** training on Colab
   - Results sync to Drive
   - Claude Code analyzes results, writes report
   - Claude Code reports to user with recommendation
   - **User decides** next step

### Planned Improvement Directions

(Priority to be determined after baseline analysis)

| Direction | Target Layer | Description | Source |
|---|---|---|---|
| A: Bottleneck Token Fusion | Fusion | Learnable bottleneck tokens as cross-modal mediator | `docs/proposed_methods.md` |
| B: Temporal Encoder Enhancement | Encoder | Conv1D / local attention before fusion | `docs/proposed_methods.md` |
| C: Label Smoothing | Labels | Gaussian-smoothed soft labels | `docs/proposed_methods.md` |
| D: Task-Specific Heads | Head | Separate token streams for state vs trend | `docs/proposed_methods.md` |
| E: Modality Reliability Weighting | Fusion | Adaptive per-modality confidence gating | `docs/proposed_methods.md` |
| F: Progressive Fusion | Fusion | Hierarchical fusion with staged integration | `docs/proposed_methods.md` |

---

## 10. Results Directory Structure

```
G:\我的云端硬盘\AmuCS_experiment\runs\
├── baselines/
│   ├── cross_subject/
│   │   ├── eft_state_trend_3seed/
│   │   │   ├── single_video/
│   │   │   │   ├── 2026-...__seed0/
│   │   │   │   │   ├── config.yaml
│   │   │   │   │   └── metrics.json
│   │   │   │   ├── ...seed1/
│   │   │   │   └── ...seed2/
│   │   │   ├── dual_video_km/
│   │   │   ├── ...
│   │   │   ├── results.tsv
│   │   │   └── results_summary.csv
│   │   ├── mft_state_trend_3seed/
│   │   ├── lft_state_trend_3seed/
│   │   ├── late_state_trend_3seed/
│   │   ├── cma_state_trend_3seed/
│   │   └── gated_state_trend_3seed/
│   └── within_subject/
│       └── (same structure)
├── improvement_A_bottleneck/
│   └── (same structure)
├── improvement_B_temporal/
│   └── ...
└── ...
```

---

## 11. Analysis Report Template

Each experiment report (`docs/experiment_results/{name}_report.md`) should contain:

1. **Method**: Name, registration key, core idea, what bottleneck it addresses
2. **Configuration**: Key parameters, what differs from baseline
3. **Results Table**: All modality combinations × 2 splits, 3-seed mean +/- std
4. **Comparison**: vs best baseline (same modality combo)
5. **Conclusion**: Effective or not, why, next recommendation
