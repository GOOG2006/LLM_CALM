# Pilot 预注册 — Idea 1: Detailed-Balance Verifier(推理转移的"熵产/不可逆性")

> Gate 1(存在性)。承接 IDEA_REPORT.md 领头种子。**在跑全量前钉死存在性问题、判死阈值、防自欺闸。**
> 预注册时间: 2026-08-05。**主 2026 baseline = GenPRM(AAAI 2026,官方代码 RyanLiu112/GenPRM,服务器已有)**,ProcessBench-7B F1 80.5。
> 基座: `DeepSeek-R1-Distill-Qwen`(= GenPRM 的基座,归因干净:同基座下 GenPRM 训生成式验证器 vs 本文零训练熵产规则)。
> **分阶段**:Gate 1 存在性先在 **1.5B**(DSR1-1.5B + GenPRM-1.5B,均已有,零下载)拿信号;过闸再下 DSR1-Distill-Qwen-7B 做规模对比 vs GenPRM-7B。
> 次要 baseline:Qwen2.5-Math-PRM-7B(监督 SOTA)、FreePRM(training-free,有代码)、self-consistency。
> 后端: vLLM `prompt_logprobs` teacher-forcing,远程 V100-32G / GenPRM 环境。

## 理论根
非平衡热力学 detailed balance / Crooks 涨落定理:物理过程的可逆性由前向/反向路径概率比刻画。类比:
**正确推理转移前后可行性平衡;错误步是"前向流畅的合理化(rationalization)",但从目标/后续反推时难以自洽** → 反向可行性低 → 沿该转移的"熵产"偏高。

对 ProcessBench 每步 `s_t`(问题 Q、前缀 `s_<t`、后续 `s_>t`、最终答案 A):
- 前向可行性 `logp_f = (1/|s_t|) log P(s_t | Q, s_<t)`(标准 teacher-forced,= 负 perplexity)。
- 反向可行性 `logp_b = (1/|s_t|) log P(s_t | Q, A, s_>t)`(把答案+后续步前置,重构该步)。
- **熵产分** `σ_t = logp_f − logp_b`。

## 存在性问题(唯一要回答的)
> 在 ProcessBench 标注步上,`σ_t` 能否分开"错误步"与"正确步"?
> 用一个数衡量:**AUROC( σ_t 作为"该步为错误步"的预测子 )**,step 级别计算。

若不可分(AUROC≈0.5)→ 熵产无判别力,当场判死。

## 判死阈值(预注册,不得事后改)
- **σ AUROC ≥ 0.65,且过下列全部闸 → 底物存在**,进 Gate 2(建 ProcessBench 验证器 + compute-matched 对比 Qwen2.5-Math-PRM-7B / FreePRM / SC)。
- **σ AUROC < 0.55 → 判死。**
- **0.55 ≤ σ AUROC < 0.65 → 灰区**:换 ProcessBench 更难子集(OlympiadBench/Omni-MATH)+ 加样复测;仍 <0.65 判死。

## 防自欺闸(每条结论过闸,任一不过即降级)

1. **方向归因闸(核心,对标动力学校对的闸①)**:σ 的判别力是否只是普通 perplexity(前向项)换皮?
   分别报 **AUROC[仅前向 logp_f] / AUROC[仅反向 logp_b] / AUROC[σ=差]**。
   **若 σ 相对"仅前向 logp_f"增益 < 0.03 AUROC → 反向项无独立价值,降级为"perplexity 验证器",非 detailed-balance 方法创新。**
   *——这一刀决定"不可逆性/反向"这条跨学科桥是否真有东西,是全 pilot 最关键的。*
2. **自洽但错闸(Gate 0,对标全项目跨切威胁)**:在"自洽但错"子集(base 对该题多采样答案一致却错)上,σ 是否仍有判别力?
   **若 σ 在自洽错子集 AUROC 掉到 <0.55 → 撞第四次墙,判死。**
3. **SC 正交闸**:σ 是否只是 self-consistency 换皮?对照 = 每步 K 次重采样的分歧率(per-step SC disagreement)。σ 需相对该对照有增益,或至少在 SC 抓不到的子集上有效。
4. **位置/长度混淆闸**:ProcessBench 错误偏后;σ 的判别力须在按步位置 & 步长分层后**层内存活**,且打过平凡"位置先验"(总猜最后/中间步)基线。

## 样本与分层
- 底物:**ProcessBench 的 MATH + OlympiadBench + Omni-MATH 子集**(竞赛级、gold 可信)。**禁用 GSM8K 子集**(dead-4:残余错=标注噪声)。
- step 级两桶:错误步(每案标注首错步)与正确步(首错步之前的步);各桶 ≥ ~100 条,不足则加案例。
- 自洽但错子集(闸②):对每题用 base 多采样(K=6)整解、答案众数与 gold 比,取"答案自洽却错"的案例单列。

## Compute(Gate 1)
- 每步:1 前向 teacher-forced pass + 1 反向 pass(均 vLLM prompt_logprobs,批量)。
- 闸③ SC 对照:每步 K=5 重采样(仅在过了闸①后才补跑,省算力)。
- 预算 < 2 GPU-hr / 数百案例。

## 产出
- `pilot/detbal_pilot.py` — 存在性测量(前向/反向 logprob + σ + 分桶 AUROC + 三闸)。
- `pilot/results_detbal.jsonl` — 每步 {case_id, subset, step_idx, is_error, chain_len, logp_f, logp_b, sigma, self_consistent_wrong}。
- `pilot/FINDINGS_detbal.md` — 三个 AUROC(前向/反向/σ) + 闸①增益 + 闸②自洽错 AUROC + 判死/过闸决策。

## 状态
- [x] 预注册本文件。
- [ ] 下载基座 Qwen2.5-Math-7B-Instruct(进行中,hf-mirror)。
- [ ] 写 detbal_pilot.py,ProcessBench 非-GSM8K 子集加载 + 前向/反向 logprob。
- [ ] Round 1 测 σ AUROC + 闸①(方向归因)——**这是第一刀,增益<0.03 直接判死**。
- [ ] (过闸①)补闸②自洽错 + 闸③ SC 对照 + 闸④分层。
- [ ] FINDINGS + 决策。
