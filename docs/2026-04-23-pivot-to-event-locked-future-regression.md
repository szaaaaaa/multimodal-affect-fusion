# 2026-04-23 · 范式切换：从 3-class 压力分类到未来轨迹回归

**状态**：已决策，待实施
**影响范围**：Ch4 任务定义、Ch5 闭环接口、评估框架
**起草者**：ziang（与 Claude 协作）

本文档记录本次会话中从质疑当前实验范式到锁定新任务定义的完整推理链条，供毕业论文 Ch4 撰写、后续实施和与导师对齐使用。

---

## 1. 起点：对现有 3-class 任务的质疑

当前实验做的是 **stress_level 三分类（LOW / MID / HIGH）+ stress_trend 三分类（DOWN / FLAT / UP）**，标签由 `scripts/build_stress_labels.py` 从连续 ranktrace 生成：

- **stress_level**：EMA 平滑后，按 **per-session 的 q20/q80 分位数**切片
- **stress_trend**：前后窗均值差分 + per-session τ = 0.25·std 阈值

252-run baseline 显示 cross-subject state 天花板 ≈ 0.44（macro-F1），within-subject state 显著更差。

**质疑触发点**：为什么被试内反而比跨被试表现更差？

---

## 2. 诊断：标签生成与切分的不兼容

### 2.1 within-subject state collapse 的机制

- `build_stress_labels.py:65-66` 里 `q20 / q80` 是用**整段** `y_smooth` 算的
- ranktrace 在一场游戏内存在**单调漂移**（紧张感随 session 累积）
- `within_subject.json` 配合 `temporal_split_ratios` 把**同一 stem 按时间比例切**（前 60% / 中 20% / 后 20%）
- 结果：训练段只有 LOW+MID，测试段只有 MID+HIGH
- 模型训练时没见过 HIGH 的输入分布 → F1 直接崩塌

这不是"任务更难"，而是**标签阈值和切分方式的错配**。

### 2.2 两个正交的问题

| 问题 | 机制 | 修法维度 |
|---|---|---|
| 阈值错位 | q20/q80 在整段算，train 段只有 LOW+MID | 标签侧 |
| 分布漂移 | 前后段 y 分布不重合 | 切分侧 |

---

## 3. 数据结构真相（避免走错切分路线）

本节记录了一次**方法论错误和纠正**，对后续研究者极其重要。

### 3.1 一次错误的假设

最初以为：
- stem `S<X>_P<Y>` 里 `P<Y>` 是某个常驻玩家
- `steamID` 是真实玩家身份
- 当前 cross-subject split 按 stem 切会泄露，因为同一 steamID 出现在 train/test 两边
- 建议改为 **steamID 级 LOSO**

**ziang 的关键质疑**："你确定同一 steamID 就是同一个玩家吗？"

### 3.2 验证数据

通过 `gamePlayerInfo.csv` 和 `AMuCS_prequestionnaire_results.csv` 交叉验证：

| 检查项 | 结果 |
|---|---|
| 主力 steamID `...554076` 对应的 `player_name` | 始终是 `sims-P1`，坐 slot 0 达 53/57 次 |
| `AMuCS_prequestionnaire_results.csv` 的 `player_id` unique 值 | 只有 P1, P2, P3, P4（座位号） |
| 同一 `player_id=P1` 下的 **年龄** | 8 种不同值（18, 19, 21, 22, 24, 26, 30, 33） |
| 同一 `player_id=P1` 下的 **CSGO rank** | 5 种不同级别 |
| Session 编号 vs 问卷 group 编号 | **完全对应**，Sessions NOT in groups = [23, 34]，Groups NOT in sessions = [] |
| 每个 `(group, player_id)` 问卷提交次数 | 231/233 只提交一次 |

### 3.3 真相

- **`steamID` = 实验室 4 台工位 PC 的共享 Steam 账号**（`sims-P1` / `sims-P2` / `sims-P3` / `sims-P4`），不是人
- **session = group**，71 场实验各对应一个 group
- **每 stem `S<X>_P<Y>` = 一个独立被试**，一生打一场
- **总被试数 ≈ 233 个 unique (group, player_id)**

### 3.4 对切分的最终结论

| 原先判断 | 修正后 |
|---|---|
| 按 stem 切不是真 cross-subject | ❌ 错。**按 stem 切就是真 cross-subject** |
| 应按 steamID 做 LOSO | ❌ 错。按 steamID 切等于按**座位**切，会引入座位线索泄露 |
| AMuCS 可做 multi-session within-player | ❌ 错。每人只打一场，within-player multi-session 根本不存在 |
| `within_subject.json` 是错的 | ⚠️ 语义应改名为 **within-session**（同一人同场的时间切） |

### 3.5 当前 split 文件评估

| 文件 | 同 session 跨 split 泄露 | 判定 |
|---|---|---|
| `session_tvt.json` | 0 | ✅ 正确的 cross-subject |
| `multimodal_split_seq_session_tvt.json` | 0 | ✅ 正确的 cross-subject |
| `multimodal_split_seq.json` | **12** | ❌ 应停用 |

**行动项**：停用 `multimodal_split_seq.json`，统一使用带 `session_tvt` 字样的两个文件。

---

## 4. 范式质疑：macro-F1 是否适合 ranktrace？

### 4.1 黄金类比（ziang 提出）

> 压力预测类似黄金价格：遭遇黑天鹅事件（伊朗冲突）黄金会随预期波动。压力也应由玩家被击杀、交火等事件决定。

这个类比揭示了当前范式的三处结构性错误：

1. **输入端把事件压扁了**：telem 109 维是 per-window 统计聚合，击杀/死亡的**时间戳和顺序丢了**
2. **标签端把响应曲线当稳态分布切**：q20/q80 假设 HIGH 占 20%，但事件响应是**脉冲 + τ 衰减**
3. **评估端用 macro-F1**：奖励"稳态命中率"，惩罚"早 2 秒预报峰值"

### 4.2 对 252-run 天花板的重新解读

macro-F1 ≈ 0.44 的天花板**不是任务本质极限**，而是"非事件感知 baseline"的极限。模型可能通过学习 MID baseline 分布拿到 0.44，对事件毫无理解——而在事件响应上**完全失败**。

---

## 5. AMuCS 数据的事件可用性核查

为判断事件驱动路径是否可行，核查了 telem / km / video 的粒度和事件结构：

| 模态 | 原始粒度 | 事件可用性 |
|---|---|---|
| ranktrace | ~1.4 Hz（每 session ~260 点） | 事后 annotation，已经是响应信号 |
| gameInt | **64 Hz** per-tick | ✅ `deathVictim` / `deathAttacker` / `damage` / `combat` 均有 per-tick 字段 |
| gameMrk | 每 session 5-20 条 | ❌ 只有元数据，**不是 round 事件流** |
| KM raw | 8 Hz 按键 + 1 kHz 鼠标位置 | ✅ 原始事件时间戳在 `keyboard.csv` / `mousebuttons.csv` |
| Video | 30 Hz 帧 | ✅ 屏幕视频 |

**round 字段不可用**：每 session 的 `round` 只有 1-3 个值，其中一个占 600+ 秒。AMuCS 是长 session（11-14 分钟）模式，不是标准 MR12/MR15 多 round。leave-one-round-out 不可行。

**死亡事件天然可用**：每 session 20-35 次死亡，gap 中位数 18 秒，范围 5-85 秒。是天然的事件边界。

---

## 6. Phase 0：事件锁相响应诊断

### 6.1 设计目标

一张图回答：**AMuCS 的 ranktrace 在游戏事件发生后真的会系统性地变化吗？**

如果答案是"会" → 事件驱动路径成立
如果答案是"不会" → ranktrace 和事件关联弱，要另起叙事

### 6.2 方法

脚本：`scripts/phase0_event_response.py`

- **事件**：`gameInt.csv` 里 `deathVictim != -1` / `deathAttacker != -1` 的绝对时间戳
- **孤立过滤**：前后 ±5 秒无**其他类型**事件
- **Ranktrace**：`ranktrace.csv` 绝对时间 → 5 Hz 均匀网格（线性插值）
- **Per-event detrend**：减 pre-event 5s baseline 均值
- **归一化**：除以 **session-level std**（不是 per-event baseline std，否则会因 baseline 平直段爆炸）
- **Shuffled control**：每 stem 内随机采样同等数量时刻，重复 500 次，取 2.5 / 97.5 百分位做 CI 带

### 6.3 结果

| 事件类型 | n 事件 | 峰值幅度 | 峰值延迟 | shuffle p |
|---|---|---|---|---|
| deathVictim | 1641 | **+0.27 σ** | +7.2s | **<0.001** |
| deathAttacker | 1846 | **-0.26 σ** (谷底) | +5s (谷) | 0.002 |

图：`AmuCS_experiment/phase0/phase0_response.pdf`

### 6.4 形态解读

**deathVictim（被杀）**：事件前 2 秒开始上升 → 事件时 +0.12σ → +7.2s 达峰 +0.27σ → 缓慢衰减。S 形响应。

**deathAttacker（击杀）**：事件瞬间 +0.08σ 小尖峰 → 之后急剧下降到 -0.26σ（压力释放）→ 5s 达谷 → 缓慢恢复。

### 6.5 三个关键发现

1. **响应双向且对称**：死亡 +0.27σ vs 击杀 -0.26σ，幅度几乎对称 → ranktrace 测的是完整的"压力-放松"维度
2. **死亡事件在 t=-2s 有预响应**：玩家（或事后回放的标注者）意识到"要死了"比实际死亡早 2 秒。**这说明事件是可提前预测的**
3. **时间尺度清晰**：死亡响应 10-12 秒，击杀响应 8-10 秒 → 未来 10 秒预测窗口足够

### 6.6 结论

事件驱动假设**完全成立**。可以进入 Phase 1（任务定义 + 评估实现）。

---

## 7. 任务定义选择

### 7.1 候选方案

| 方案 | 描述 | 对 Ch5 预警能力 | 信息保真 | 与 Phase 0 对齐 |
|---|---|---|---|---|
| A | 当前时刻连续回归 | ❌ 只能事后确认 | 中 | 只用当前点 |
| B | 事件响应二分类 | ⚠️ 有但粗糙 | 低 | 丢弃响应形态 |
| C | 残差回归 | ❌ 同 A | 中 | 同 A |
| D | 重定义 3-class 标签 | ⚠️ 分类粒度粗 | 低 | 丢弃幅度 |
| **E** | **未来 N 秒轨迹回归** | **✅ 天然预测未来** | **高** | **完整利用响应曲线** |

### 7.2 为什么选 E（未来轨迹回归）

回归 Ch5 部署场景：LLM 需要 **在压力峰值到来之前**干预，才能有效调整难度。

- Phase 0 已证明死亡事件有 2 秒预响应窗口
- 如果任务本身是"预测未来 10 秒的压力曲线"，LLM 每次读到的是**未来 10s 的压力预测**，可以看出上升斜率、峰值位置、峰值幅度
- 其他四个方案都无法直接提供"未来轨迹"给 LLM

### 7.3 LLM 提示词对比

```
A（单点回归）:   "当前压力: 0.65"
                LLM 只能反应当下

E（未来轨迹）:  "未来 10s 压力曲线: 0.65 → 上升至 0.92 @ +4.2s → 回落"
                LLM 能在 4 秒前就开始预防性干预
```

---

## 8. 最终任务定义

### 8.1 Spec

```
输入：  t 时刻之前 20s 的多模态窗口 (video + KM + telem)，100 帧 @ 5Hz
输出：  51 维向量 [ŷ(t), ŷ(t+0.2s), ..., ŷ(t+10s)]
GT：    t 到 t+10s 的 ranktrace 序列（5Hz，51 点）
Loss：  MSE（主）；CCC 可作补充
```

### 8.2 评估（三级）

1. **传统连续回归指标**：整段 MSE / CCC，与文献可比
2. **事件锁相四指标**（Phase 1 实现）：
   - Peak-F1 @ ±2s
   - Lead time（median ± IQR）
   - Amplitude error
   - Event-triggered correlation（稳健性交叉检验）
3. **Lead time 分层**：按预测 horizon 分层报告（+1s / +3s / +5s / +10s）

### 8.3 已确认参数

| 参数 | 值 |
|---|---|
| 未来预测窗口 | 10 秒（51 点 @ 5Hz） |
| 输入上下文窗口 | 20 秒（100 点 @ 5Hz） |
| Loss | MSE |
| 归一化参考 | session-level σ |
| 事件锁相 pre / post | 5s / 15s |
| 事件类型起步 | `deathVictim`, `deathAttacker` |
| Shuffle trials | 500 |

---

## 9. 对论文章节的影响

### Ch4 新结构

1. **§4.1** 旧范式 baseline（252-run，3-class macro-F1，ceiling 0.44）
2. **§4.2** 诊断：Phase 0 事件锁相响应图（killer figure）
3. **§4.3** 新范式：未来轨迹回归 + 事件锁相评估
4. **§4.4** 新范式下的 baseline 比较（CMA / Gated / LFT / XGBoost dumb baseline）
5. **§4.5** Lead time 分析（对 Ch5 的桥梁）

叙事升级：从 "又比较了几个 fusion" → "方法论诊断 + 新评估范式 + 新 baseline"。

### Ch5 闭环

- 基于 Ch4 最优模型的 **未来 10s 预测**作为 LLM 状态输入
- LLM 据此决策难度调整
- 部署协议：每 0.5-1 秒调用一次模型，输出 51 点轨迹 → LLM 读取

### Ch6

记忆里已定的 RL-simulated player eval 不受影响。

---

## 10. 实施路径

| Phase | 工作 | 预算 | 产出 |
|---|---|---|---|
| 0 | 事件锁相诊断（本次已完成） | 0.5 天 | `phase0_response.pdf` ✅ |
| 1 | 事件锁相 evaluator 实现 | 1-2 周 | `src/metrics/event_locked.py` + pipeline |
| 2 | 切换任务到未来轨迹回归 | 1 周 | `RegressionHead(out_dim=51)` + 新 config |
| 3 | 重跑 3-5 个代表模型 | 1-2 周 | 新 baseline 表 |
| 4 | 论文 Ch4 写作 | 2 周 | Ch4 草稿 |

总计 ~5-7 周，距 8 月 deadline 仍有 buffer。

---

## 11. 和旧实验的可比性

- **不删 252-run 结果**，保留作为"旧范式 baseline"
- 新模型的 retrospective 单点输出（只取 `ŷ(t)`）可以 argmax 后算 macro-F1，和旧 0.44 直接比较
- Ch4.1 讲旧范式数字，Ch4.3-4 讲新范式数字，4.5 讨论"为什么旧指标选错了模型"

---

## 12. 已停用 / 已废弃

- **停用** `splits/multimodal_split_seq.json`（12 session 泄露）
- **语义重命名** `splits/within_subject.json` → 叙事上明确为 **within-session**（不是真 within-subject，AMuCS 不支持 multi-session within-player）
- **废弃** 按 steamID 切分的想法（曾误认为是 cross-player，实为 cross-seat）
- **废弃** 基于分位数的 state 标签作为主任务（保留用于旧范式对照）

---

## 13. 关于方法论的一个元教训

在本次讨论中，曾一度建议按 steamID 做 LOSO 作为"真 cross-subject"。这是**基于命名 heuristic 推断字段语义**的错误——`steamID` 听起来像个人身份标识，但在 AMuCS 的实验设计里它是实验室工位账号。

**教训**：涉及数据切分 / 身份映射时，必须通过 documentation 和跨字段 consistency check 验证字段真实语义，不能用命名直觉推断。已记入 `memory/feedback_verify_data_semantics.md`。

---

## 14. 下一步立即行动项

1. ✅ **本文档**已归档到 `docs/2026-04-23-pivot-to-event-locked-future-regression.md`
2. ⏭️ **约导师 meeting**：把 Phase 0 响应图 + 本文档发给导师，对齐 Ch4 调整
3. ⏭️ **导师 OK 后**：进 Phase 1 实现 `src/metrics/event_locked.py`
4. ⏭️ **停用** `multimodal_split_seq.json` 的所有引用（grep 一下 configs）
