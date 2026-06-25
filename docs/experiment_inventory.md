# 实验运行清单

> 更新日期：2026-04-24
> 数据来源：直接数 `G:/我的云端硬盘/AmuCS_experiment/runs/` 下每个 run 目录
> 目的：记录当前实际做完的实验范围、缺口，作为毕设写作时的 fact-check 参考

---

## 一、一张总表

| 代号 | 实验组 | 路径 | 设计矩阵 | 完成状态 | Run 数 |
|---|---|---|---|---|---|
| A | 3-class state+trend baseline sweep | `archive/baselines/` | 6 fusion × 7 modality × 2 split × 3 seed | ✅ 完整（含重跑）| 252 设计 / 356 目录 |
| B | EFT + C/D/F 独立消融 | `archive/eft_CDF_ablation/` | 4 variant × 3 modality × 2 split × 3 seed | ✅ 完整 | 72 |
| C | 任务公式化对比（experiment1）| `archive/experiment1/` | 7 task × 7 modality × 1 split × 3 seed | ⚠️ 只有 1 个 split | 147 |
| D | Path C state-only + event/60Hz | `pathC_full_state_only/` | 9 variant × 4 modality × 2 split × 3 seed | ✅ 完整 | 216 |
| E | Path B CMA state+trend | `archive/pathB_cma_state_trend/` | CMA only × 4 modality × 2 split × 3 seed | ⚠️ 只有 CMA | 25 |
| F | Path B state-only pilot | `archive/pathB_state_only_pilot/` | 1 modality × 5 seed | ⚠️ Pilot | 5 |
| G | Trajectory regression pilot | `trajectory_pilot/` | EFT only, 部分 modality | ⚠️ 严重不完整 | 16 |
| H | Dumb baseline（非神经 ceiling）| `dumb_baseline/results.csv` | 3 model × 3 seed | ✅ 完整 | 9 |
| I | Phase 0 事件锁相诊断 | `../phase0/phase0_stats.json` | 统计诊断（非 run）| ✅ 完成 | — |
| J | Early LFT 探索 | `archive/final_lft_*/` | 前期 pilot | 🗑️ 归档，不用 | ~9 |

**神经实验 run 目录合计：~750+（含重跑则 ~850+）**

---

## 二、详细分解

### A. 3-class state+trend baseline sweep（主线）

设计：6 fusion (CMA/EFT/Gated/Late/LFT/MFT) × 7 modality × 2 split × 3 seed = 252 runs
| split | 设计 | 实际 run 目录 |
|---|---|---|
| cross_subject | 126 | 208（部分 cell 有重跑）|
| within_subject | 126 | 148 |

对应分析：`docs/baseline_results_analysis.md`（522 行硬分析）、`AmuCS_experiment/实验结果汇总.md`

### B. EFT + C/D/F 独立消融

设计：4 variant (EFT+C / EFT+D / EFT+F / EFT+CDF) × 3 modality (single_video / dual_video_telem / triple) × 2 split × 3 seed = 72 runs
cross_subject: 36 / within_subject: 36

### C. 任务公式化对比 — experiment1

每 task 跑 7 modality × 3 seed = 21 runs。**只在 1 个 split 上跑了**。

| task 子目录 | 任务类型 | runs |
|---|---|---|
| `arousal_3cls_3seed` | 3-class 单任务 | 21 |
| `arousal_3trend_3seed` | 3-class trend 单任务 | 21 |
| `arousal_reg_trend_multitask_3seed` | **连续 arousal 回归 + trend multitask** | 21 |
| `cma_arousal_reg_trend_multitask_3seed` | CMA 版连续回归 | 21 |
| `cma_state_trend_multitask_3seed` | CMA 版 3-class | 21 |
| `perparticipant_z_3seed` | per-participant z-score 归一化 | 21 |
| `state_trend_multitask_3seed` | 3-class state+trend multitask | 21 |

### D. Path C state-only + event/60Hz

9 variant（cma / eft / eft_C / eft_CD / eft_D / gated / late / lft / mft）× 4 modality（dual_km_event_telem_60hz / dual_video_km_event / dual_video_telem_60hz / triple_video_km_event_telem_60hz）× 2 split × 3 seed = 216 runs
cross_subject: 108 / within_subject: 108

### E. Path B CMA state+trend

CMA × 4 modality × 2 split × 3 seed = 24 runs 设计 / 25 目录（triple cross 有 4 seed）

| split | modality | seed 数 |
|---|---|---|
| cross | dual_km_event_telem_60hz | 3 |
| cross | dual_video_km_event | 3 |
| cross | dual_video_telem_60hz | 3 |
| cross | triple_video_km_event_telem_60hz | 4 |
| within | dual_km_event_telem_60hz | 3 |
| within | dual_video_km_event | 3 |
| within | dual_video_telem_60hz | 3 |
| within | triple_video_km_event_telem_60hz | 3 |

### F. Path B state-only pilot

`cma_pathB_state_only_3seed/triple_video_km_event_telem_60hz/` — 5 seed（pilot 性质）

### G. Trajectory regression pilot — 严重不完整

| 路径 | modality | seed 数 |
|---|---|---|
| cross/eft_trajectory_pilot_3seed | single_video | 5 |
| cross/eft_trajectory_pilot_3seed | dual_video_telem | 4 |
| cross/eft_trajectory_pilot_3seed | triple_video_km_telem | 3 |
| cross/eft_trajectory_pilot_uniform_3seed | single_video | 1 |
| within/eft_trajectory_pilot_within_short_3seed | single_video | 1 |
| within/eft_trajectory_pilot_within_single_3seed | single_video | 1 |
| within/eft_trajectory_pilot_within_stem_3seed | single_video | 1 |

只有 EFT，大量 cell 缺失。对应 notebook：`notebooks/run_trajectory_*.ipynb`

### H. Dumb baseline（非神经 ceiling）

`runs/dumb_baseline/results.csv`：LogReg / RandomForest / XGBoost × 3 seed = 9 runs
- XGBoost: test_macro_f1 = 0.4416（跨 seed 稳定）
- 这个数字对应 memory 里的 "dumb baseline matches neural fusion" finding

### I. Phase 0 事件锁相诊断

`AmuCS_experiment/phase0/phase0_stats.json`（params: pre=5s / post=15s / 500 shuffle）：
- `deathVictim`: 1641 events, peak +0.272σ @ +7.2s, p<0.001
- `deathAttacker`: 1846 events, peak +0.083σ @ +0.2s, p=0.002

对应图：`phase0/phase0_response.pdf`
对应 pivot 决策文档：`docs/2026-04-23-pivot-to-event-locked-future-regression.md`

---

## 三、明确的缺口（按优先级）

| # | 缺口 | 优先级 | 理由 |
|---|---|---|---|
| 1 | **事件锁相 future ranktrace regression（完全未跑）** | 🔴 P0 | 当前 bug-blocked；是 2026-04-23 pivot 的主任务；没这个，pivot 只是论文里的"设想"而非 empirical finding |
| 2 | trajectory_pilot within_subject 每个 variant 只 1 run | 🟡 P1 | 没达到 3 seed 统计门槛 |
| 3 | trajectory_pilot 其他 fusion（只测 EFT）| 🟡 P2 | 缺 CMA/Gated 对照 |
| 4 | Path B 其他 fusion（只测 CMA）| 🟡 P2 | 没有非 CMA 的 Path B 对照 |
| 5 | 连续 arousal regression 在 within_subject 的复现 | 🟡 P2 | experiment1 只有 1 split |
| 6 | EFT CDF 叠到 CMA / Gated（`baseline_results_analysis.md` 阶段 2）| 🟢 P3 | ~48 runs，写作期放不下 |

---

## 四、对毕设写作的含义

**已完成的 5 条完整主轴**（A / B / C / D / E）足够支撑毕设论文，无需跑新实验即可写：

1. **Ch4 主线**：baseline sweep (A) + ablation (B) + dumb baseline (H) → "fusion 架构的系统评估 + ceiling 揭示"
2. **Ch4 支线**：task formulation 对比 (C) → "为什么选 state+trend multitask / 为什么不选连续回归"
3. **Ch4 数据处理支线**：Path B (E) + Path C (D) → "原始事件流 vs 统计特征 vs state-only 的输入设计空间"
4. **Ch4 pivot 动机**：Phase 0 诊断 (I) → "事件锁相证据推翻 3-class 范式"

**只要把缺口 #1 修完跑完**（event-locked future ranktrace regression），就多一条 "pivot 的 empirical closure" 主轴，pass → merit → distinction 往上推。

**缺口 #2-#6 写成 Limitations / Future Work 诚实声明**，不影响毕业。

---

## 五、维护约定

- 跑新实验后更新此文件的表格
- 只写实际 ls 数出来的数字，不估不猜（遵循 memory `feedback_no_fabricated_results`）
- 与 `baseline_results_analysis.md` 和 `AmuCS_experiment/实验结果汇总.md` 互为交叉引用
