# FINDINGS — PRMBench LLM-as-PRM 残余缺陷(第一个站得住的正结果)

日期: 2026-07-31 | 数据: PRMBench_Preview, gemini-2.5-flash as PRM, 180 条(8错误类×20 + 全对×20)
脚本: prmbench_prm.py / prmbench_adjudicate.py / results_prmbench.jsonl

## SOTA 对齐(闸②通过)
整体 error-recall = 0.487(published SOTA ACTPRM-X=0.667 为 PRMBench 复合分;本"找首错步"recall≈0.49 合理)。
误报率(全对样本被判有错)= 0.25。

## 每类 miss-rate
| 类型 | miss | 裁决 |
|---|---|---|
| redundency | 0.75 | **淘汰(定义噪声)** |
| counterfactual | 0.65 | **真缺陷** |
| missing_condition | 0.55 | 待裁 |
| deception | 0.50 | 待裁 |
| confidence / step_contradiction / circular | 0.45 | 待裁 |
| domain_inconsistency | 0.30 | — |

## 裁决闸(核心,来自 GSM8K 教训)
- **redundency 淘汰**:抽 7 条,PRMBench 自己的 reason 全写"without altering its correctness / still valid /
  does not affect result"——冗余步**局部全对**,PRM 漏它其实合理。这是简洁性维度,非正确性错误 → 不算缺陷。
  (标注噪声闸再次拦下假信号,同 GSM8K。)
- **counterfactual 通过**:抽 8 条,8/8 是真数学错误(错公式/错前提/错因式分解),模型确实漏。

## 缺陷特征(量化)
- **长链定位崩溃**:漏检率 短链(≤8步)0.35 → 长链(≥15步)**0.71**。
- **过早定位(early-commitment)**:82 漏检中 **55(67%)预测步 < 真首错步**;仅 12 判"全对"。
  → 主因不是"看不出错",是**顺序扫描在长链里把错误定位到太靠前处就停**。

## 可攻缺陷(结论)
标准"顺序找第一个错"式 LLM 验证有 **early-commitment 病,且随链长恶化**。

## 设计假说(待验)
**每步独立并行验证**(每步只给前缀、单独判对错,无"找第一个错"的顺序承诺)去掉早承诺、对链长不敏感,
应显著降低长链漏检。→ 接回免疫式"按步独立检测器"思路(原方向 C),但锚定在实测缺陷上。
**证伪测试**:同 180 条,per-step 独立打分 vs 顺序找首错,比长链(≥15步)漏检率是否显著下降。

## 机制对比(第一轮,同 base=gemini-flash,同 180 条,配对)
| 机制 | recall | 误报率 | 短链miss | 长链miss |
|---|---|---|---|---|
| baseline 顺序找首错 | 0.487 | 0.250 | 0.35 | 0.71 |
| ours 每步独立验证 | 0.588 | 0.300 | 0.33 | 0.60 |

诚实解读:
- **recall +0.10 是实的**(N=160,~2.5 SE)——每步独立验证确实多抓错,机制方向成立。
- **长链 miss 0.71→0.60 欠功效**(长链 N≈42,SE≈0.10)→ 提示性,不显著。
- **FP 0.25→0.30 在噪声内**(全对样本仅 N=20,SE≈0.10)→ 说明不了 FP 是否真涨。
- smoke 那组(recall 0.81/FP 0)是小样本假象,作废。

## 下一步(欠功效项 + FP 控制)
1. 扩样:全对样本 20→100(multi_solutions 有 165)、长链错误样本加量,才能定论 FP 与长链专项收益。
2. 机制精化(若 FP 真涨):两阶段——每步独立"提名"候选 + 一次全局确认(降 FP);或每步 self-consistency 投票。
3. 仍须打过强 baseline(顺序找首错 + self-consistency 投票),而非只对朴素顺序 baseline。

## 机制对比(第二轮,加强 baseline + 扩 FP 功效)N=160错误/100全对/42长链
| 机制 | recall | 误报率 | 短链miss | 长链miss | 调用/题 |
|---|---|---|---|---|---|
| 朴素顺序(弱) | 0.419 | 0.22 | 0.49 | 0.79 | 1 |
| 顺序+SC投票(强 baseline) | 0.475 | 0.20 | 0.39 | 0.69 | 5 |
| 每步独立(ours) | 0.588 | 0.33 | 0.29 | 0.62 | ~11 |

**黄灯结论(朴素每步独立 ≠ 机制创新):**
- recall +0.11 打赢强 baseline 是实的(N=160,~2.5 SE),但 **FP 0.20→0.33 也是真涨**(N=100,~2.5 SE)。
- **平衡准确率 mean(recall,1−FP):SC=0.638 vs ours=0.629 → 基本打平。** recall 收益被 FP 代价抵消,
  ours 只是更激进的工作点,非更优机制。且 compute 未对齐(ours ~11 vs SC 5 调用/题)。
- → 同种子 B 的死法风险:被 self-consistency 追平。**这版机制不成立,需破 recall/FP 权衡。**

## 下一步(破权衡的真机制候选)
1. **两阶段 nominate→confirm**:每步独立"提名"候选(高 recall)+ 一次全局确认剔除误报(恢复精度)。
   目标 = Pareto 支配 SC(recall≥且 FP≤),且 compute 与 SC 对齐(K 匹配)。
2. 或 每步 self-consistency(每步 K 判投票)稳定精度。
3. 必须 compute-matched 再比,否则 recall 优势不可归因于机制。

## 机制对比(第三轮,对标 2026 baseline)N=160错误/100全对/42长链
| 机制 | recall | FP | 长链miss | 平衡acc | 调用/题 |
|---|---|---|---|---|---|
| SC投票(2022下限) | 0.475 | 0.20 | 0.69 | **0.637** | 5 |
| 每步CoT批判(2026 baseline) | 0.312 | 0.05 | 0.74 | 0.631 | 13 |
| ours 提名→全局确认 | 0.312 | 0.04 | 0.74 | 0.636 | 14 |

**红灯结论:**
- **ours 确认层=空操作**(ours≈baseline)。它压高FP,而每步CoT批判本就低FP(0.05),无事可做。
- **每步CoT批判(2026花哨机制)用 2.6× 算力,平衡acc(0.631)反不如 2022 SC(0.637)**——拿召回换精度换亏了。
- 三轮机制(每步YN / 每步CoT / 提名确认)**没有一个在平衡acc上打过 2022 SC**。SC 墙 = 种子B 死法在此复现。

## 真正的靶子 & 唯一未试的机制
要真赢需同时 **recall>0.475 且 FP<0.20**。CoT批判低FP(0.05)但低recall。唯一未试:
**每步CoT批判 + 每步 self-consistency 投票**(K次判该步,提recall同时保CoT的低FP)。若这个也打不过 SC,则该方向对"弱base+prompting机制"判死,转"SC 胜过花哨验证"的 measurement/critique 论文。

## 资产
PREREG_prmbench.md、prmbench_prm.py、adjudicate、perstep、compare、v2.py、results_*.jsonl。
