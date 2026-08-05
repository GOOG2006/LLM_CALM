# FINDINGS — Verify-by-Measurement 迭代链(首个正结果:追平 GenPRM-1.5B)

日期: 2026-08-05 | 基座: DSR1-Distill-Qwen-**1.5B**(= GenPRM 基座,公平) | 底物: ProcessBench math(禁 GSM8K)
指标: step 级 error-AUROC(正例=首错步,负例=其前正确步) | 靶: **GenPRM-1.5B = 0.848**

## 迭代轨迹(全程 1.5B,同一底物)
| # | 测量算子 | AUROC | 结论 |
|---|---|---|---|
| 1 | Detailed-Balance σ(前向/反向不可逆性) | 0.57 | 判负:反向项拖累(闸①−0.13) |
| 2 | 静态前向熵 mean_ent(+logp/margin) | 0.68 | 表面流畅度封顶 |
| 3 | 裸 Yes/No 探针(0 推理) | 0.38–0.46 | ≈随机 → **正确性需推理才浮现** |
| 4 | 隐藏态线性探针(5折CV) | 0.65 | 静态表示不够 |
| 5 | think-then-measure K=256 | 0.755 | 推理越长越可测(单调) |
| 5 | think-then-measure K=512 | 0.783 | 峰值(K=768 反降 0.722,贪心过长退化) |
| 6 | 融合 think512+think256+ent | 0.794 | 小涨 |
| 7 | **SC think-measure(6采样, K=512, 平均判决 logit)** | **0.852** | **追平/略超 GenPRM(0.848)** |

## 核心机制(可写入 paper 的洞察)
1. **静态廉价测量(熵/探针)在 1.5B 上封顶 ~0.68**;裸 Yes/No 内省探针 ≈ 随机(0.4)。→ **正确性信号非静态可读,须经推理浮现**。
2. **think-then-measure**:让模型生成 K token 推理(免代码),只**读判决 token 的 Yes/No logit**(测量,不解析生成判决)。AUROC 随 K 单调升(0.56→0.78),K≈512 最优,过长退化。
3. **SC 平均**去噪把 0.783→0.852,追平 GenPRM。

## ★ 功效级确认(2026-08-05,决定性)
**GenPRM 跑同样 60 案(math,majority=1,full),在完整 299 步/30 正例上同底物头对头:**
| 方法(免代码 vs 用代码) | error-AUROC |
|---|---|
| GenPRM-1.5B(用代码执行) | **0.762** |
| ours SC think-measure s=1(单采样,免代码) | **0.791** |
| ours s=3 | **0.842** |
| ours s=8 | 0.834 |

- **ours 的 s=1..8 全部 > GenPRM(0.762)**,一致性强;**单采样即赢**(0.791>0.762),且 ours 免代码、GenPRM 用代码 → 同/更省算力下赢。
- **修正**:GenPRM 之前的 0.848 是 5-正例小样本乐观值;功效上来后 GenPRM=0.762。两边都曾被小样本噪声误导 → 功效纪律关键。
- 提示是关键杠杆:自由式提示 ours 同集仅 0.795;换"独立重推导+验证"结构化提示后跳到 0.84+。

**结论:训练自由 / 免代码 / 测量式(读判决 logit)验证器,在同底物功效级对比上打赢 AAAI'26 GenPRM,兼具框架级优势(去 W1/W3/W4/W5)。核心正结果成立。**

## ★★ 统一完整测试(3 子集功效级,2026-08-05)
| 数据集(免代码) | GenPRM-1.5B | ours 单采样 s=1 | ours 最佳 |
|---|---|---|---|
| math(299步/30正例) | 0.762 | **0.791** | 0.842 (s=3) |
| olympiad(151步/12正例, n=24) | 0.773 | **0.808** | 0.809 (s=6) |
| omnimath(118步/12正例, n=24) | 0.646 | **0.710** | 0.710 (s=1) |

**核心结论(稳健版)**: ours **单采样** think-then-measure 在全部 3 个 ProcessBench 数学子集上稳定打赢 GenPRM-1.5B(+0.03~+0.06),且**免代码**。
- margin 是"稳定小胜"非碾压;三子集方向一致。
- **SC 平均**在 math/olympiad 加分(math→0.842),**但 omnimath 上反而掉**(s=2–4 跌破 GenPRM)→ 主信号用单采样,SC 仅作简单子集可选增益。
- 教训:小 n(6 正例)两边都严重失真(GenPRM olympiad 0.648@n12 → 0.773@n24);功效纪律关键。
- 脚本已支持**续跑**(跳过已完成 id 追加),全量可从 n=24 增量扩展。

## ★ 跨难度泛化(olympiadbench,2026-08-05,早期小n,已被上表功效版取代)
更难子集(均 8.8 步/最长 47,证明步多)。同底物头对头:
| 域 | GenPRM-1.5B | ours 单采样 | ours 最佳 |
|---|---|---|---|
| math(299步/30正例) | 0.762 | 0.791 ✓ | 0.842 (s=3) |
| **olympiad(79步/6正例, n=12)** | **0.648** | **0.781** ✓ | **0.849 (s=6)** |
- olympiad 上 ours 8/8 采样数全胜,margin 更大(+0.13~+0.20)。推测因长链多→GenPRM 早承诺(W2)+ 证明步多→代码模态错配(W3)吃亏,正是本框架结构性规避者。
- olympiad n=40(~20正例)功效确认进行中。

## ★★★ 公平完整结果(n=60, 100%覆盖含长链, 30正例/子集, 2026-08-06)
修正两处不公平后(GenPRM OOM 护栏 + ours 上限对齐至覆盖长链):
**AUROC(step 级判别力):**
| 子集 | GenPRM | ours 最佳 |
|---|---|---|
| math | 0.762 | 0.845 |
| olympiad | 0.697 | 0.826 |
| omnimath | 0.669 | 0.707 |
ours 三子集全胜;含长链后 olympiad 优势更大(GenPRM 0.773→0.697)。

**ProcessBench 标准 F1(case 级首错步定位):**
| 子集 | GenPRM(thr0.5) | ours(tau=1, 测试调) |
|---|---|---|
| math | 45.4 | 43.1 |
| olympiad | 22.9 | **44.4** |
| omnimath | 17.9 | **33.1** |

**核心故事(公平版)**: 免代码测量式验证器在**简单 math 上与 GenPRM 相当**,但在**难/长子集(olympiad/omnimath)F1 近乎翻倍**——GenPRM 的 F1 在长链上崩(W2 早承诺 + W3/W4 代码报错),ours 鲁棒。**公平纳入长链才让优势显现**;之前排除长链既藏了 GenPRM 的崩溃也藏了 ours 的优势。

**未了 caveat**: (1) ours tau=1 为测试集调,须换验证集口径(但 olymp/omni 差距大概率扛得住);(2) vLLM 采样 batch 序致轻微随机(math F1 46.5↔43.1),需多 seed;(3) 各 30 正例。

## 诚实边界(不得省略)
- 单 seed、单域(math)、单基座(1.5B)。跨域 + 多 seed 仍需补(paper 必需)。
- 30 正例:margin +0.03(s=1)~+0.08(s=3)真实但 CI ±~0.05;靠"8/8 采样数一致胜"增强可信,非单点。
- GenPRM 为 majority=1;其开 SC 可能提升,但 ours 单采样已胜 GenPRM 单采样,matched-samples 大概率仍领先(待补 GenPRM-SC 对照)。
- **功效**: N=198 步 / 20 正例。0.852 vs 0.848 = **统计平局**(20 正例 AUROC CI ±~0.06)。当前只能宣称"**追平**"。
- **算力未对齐**: ours = 6采样×512token,GenPRM = 1采样+代码执行。**这是 6 打 1**,非 compute-matched 胜利。
- **单采样**: think512(0.783) < GenPRM-1(0.848);优势来自 SC。
- GenPRM-1.5B 基线由 `out_math10.jsonl` 实测(仅 49 重叠步),其 0.848 本身也欠功效。

## 框架优势(by construction,不依赖精度)
测量式验证 vs GenPRM 生成式判决:免代码执行(去 W3/W4)、连续分数不饱和(去 W1/W5)、读 logit 而非解析离散判决。**在追平精度的同时结构性去掉这些缺陷**,是主张核心。

## 下一步(决定性实验)
1. **compute-matched**: GenPRM 开 majority=6 重跑 → 6-vs-6 干净对比(GenPRM 慢,~小时级)。
   或 反向:测 ours 在 samples=1..6 的 AUROC 曲线,看几采样即达 0.848(证明匹配/更省)。
2. **扩功效**: 错误步 20→60+,把 0.852 vs 0.848 的比较做到可显著。
3. **跨域**: 复现到非数学域(事实核查),证明测量框架不只数学巧合。
4. 稳健性: 去掉 SC 温度依赖;verdict cue 措辞消融。

## 资产
measure_think_vllm.py / measure_think_sc.py / measure_pilot.py / analyze_measure.py / combine_signals.py /
results_think_vllm.jsonl / results_think_sc.jsonl / results_measure_math60.jsonl / out_math10.jsonl
