# PREREG — PRMBench 上 LLM-as-PRM 残余缺陷分析(用户方法论落地)

> 方法(见 [[user-research-method]]):锚定顶会公用数据集 → 攻 SOTA 残余错误 → 据缺陷设计。
> 数据集 = **PRMBench**(ACL'25,2026 PRM 论文公用榜,SOTA ACTPRM-X=0.667≈GPT-4o)。
> 后端 = gemini-flash(GPT-4o 在此持平开源 SOTA,故 API 版即代表 SOTA 水平)。预注册 2026-07-31。

## 数据
`hitsmy/PRMBench_Preview`,N=6216。8 类注入错误各 ~757 + multi_solutions 165(全对样本,测误报)。
每条:question + modified_process(步列表) + error_steps(错误步,1-indexed) + classification + reason。
错误为**合成注入 + 人工校验**(比 GSM8K gold 干净,但代表性有限,故仍设标注噪声闸)。

## 任务(LLM-as-PRM)
给 question + 编号步骤,模型输出**第一个错误步号**,全对输出 0。
- 错误样本(error_steps 非空):hit = 预测步 ∈ error_steps;miss = 漏检(残余失败)。
- 全对样本(multi_solutions):flag 任何错 = 误报(false positive)。

## 存在性问题(唯一要回答的)
复现 SOTA 水平(整体 recall ≈ 0.5–0.7)后,**是否存在某个错误类型,LLM-as-PRM 系统性漏检、
且漏检案例经裁决确为真错(非标注噪声)?** —— 这就是可攻的"缺陷"。

## 指标
- 每类 miss-rate(漏检率)= 1 − 该类 recall;全对样本 false-positive-rate。
- 整体 recall(对齐 SOTA 的 sanity check:若远低于 0.5,先查 harness/prompt bug 再谈)。

## 防自欺闸
1. **标注噪声闸(核心,来自 GSM8K 教训)**:从漏检率最高的类抽 ≥12 条,人工读注入错误+模型判断,
   确认 ≥70% 是**真错且模型确实漏了**(非"注入的'错'其实合理/PRMBench 标注可疑")。<70% → 该类不算真缺陷。
2. **SOTA 对齐闸**:整体 recall 须落在 published SOTA 区间(~0.5–0.7)。太低=harness 问题,结论作废。
3. **baseline 闸**:定位的缺陷,须是简单 baseline(如"让模型逐步 verify")也栽的地方,才值得设计。

## 判死/过闸阈值(预注册)
- **某类 miss-rate ≥ 0.40 且 裁决 ≥70% 真错 → 找到可攻缺陷**,进入方案设计(据该类结构)。
- **所有类 miss-rate < 0.40,或最差类裁决 <70% 真错(多为噪声)→ 无集中缺陷**,回退换数据集/角度。
- 灰区:加大采样或换 GPT-4o 复测。

## 采样(pilot)
分层:8 错误类 × 20 + multi_solutions × 20 = 180 条(seed 固定)。每条 1 次 LLM-PRM 调用。
够看出类间 miss-rate 差异;定位后对最差类扩样 + 裁决。

## 产出
`prmbench_prm.py`、`results_prmbench.jsonl`(每条 {idx, classification, error_steps, pred_step, hit, n_steps})、
每类 miss-rate 表 + 最差类裁决 + 缺陷结论。

## 状态
- [ ] smoke(gemini-flash,格式/解析)。
- [ ] pilot 180 条 → 每类 miss-rate + 整体 recall(对齐闸)。
- [ ] 最差类扩样 + 标注噪声裁决。
- [ ] 缺陷结论 → 方案设计 or 回退。
