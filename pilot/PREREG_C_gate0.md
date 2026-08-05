# Gate 0 预注册 — 方向 C(免疫负选择 / 正交信号)存在性测

> 承接种子 B 判死(见 FINDINGS.md)。B 的两条硬结论:
> (1) self-consistency 是强 baseline(AUROC 0.84–0.89);(2) "自洽但错"跨三域,是结构墙。
> 方向 C 唯一的立身之本 = 能对**自洽子集**(self-consistency 够不着的地方)有判别力。
> **不先过这一关,C 和 B/方向B 同死。** 预注册: 2026-07-31。

## 存在性问题(唯一要回答的)
在 base 模型**自洽**的样本(k 次采样答案全一致 → self-consistency 无信息)里,
存在**任何正交信号**能把 base-对 与 base-错 分开吗?第一个候选信号 = **跨模型分歧**:
独立 checker 模型对同题的答案,与 base 答案分歧多大。

> **Gate 0 指标 = AUROC( 跨模型分歧 → base 答错 ),只在自洽子集上算。**

## 判死阈值(预注册,不得事后改)
- **自洽子集 AUROC ≥ 0.65(且子集内错答 ≥ 15)→ Gate 0 过**:正交信号存在 → 方向 C 活,进 Gate 1。
- **< 0.55 → 判死**:连跨模型都抓不住自洽-错 → 输出侧验证对"自信-错"的这一大类近乎不可能(至少廉价版)。
- **0.55–0.65 或 子集错答 <15 → 灰区**:加 N,或等 OpenAI 配额用 GPT 当真·异家族 checker 复测。

## 防自欺闸
1. **checker 能力混淆闸(核心)**:若 checker 远强于 base,"分歧"退化成"checker 更对"→ 变成
   "直接用更强模型",非正交信号。**今天用 Gemini 内部对(flash-lite × flash)只算探针**;若过闸,
   Gate 1 **必须**用**同档不同家族**(GPT×Gemini 平级)复测,证明信号来自"不同",非"更强"。
2. **自洽定义闸**:主定义 = base 的 k 次采样答案**全一致**(disagree=0)。附报 disagree≤0.2 的近自洽子集
   以补功效,但判死以严格子集为准(够 15 错答时)。
3. **判分闸**:GSM8K 数值判分(gemini-2.5-flash-lite 已验证格式规整,见 run_gemini)。

## 设计
- base = gemini-2.5-flash-lite,k_base=6(定自洽 + 定对错)。checker = gemini-2.5-flash,k_check=4。
- cross_disagree = checker 的 k_check 个答案里 ≠ base 多数答案的比例(连续分歧度)。
- 每题存 {id, gold, base_ans, base_disagree, correct, cross_disagree},便于明天并入 GPT checker。
- N=500(seed 0 抽样,与 B 的题池可对齐);预期自洽-错 ~10–15,不足则判为灰区加 N。

## 产出
- `gate0_crossmodel.py`、`results_gate0.jsonl`、Gate 0 AUROC(全集 + 严格自洽 + 近自洽)+ 判决。

## 状态
- [ ] 跑 Gemini 内部对(探针)→ 自洽子集 AUROC → 过/死/灰。
- [ ] (过闸)等 OpenAI 配额,GPT×Gemini 异家族复测(过闸①能力混淆闸)。
