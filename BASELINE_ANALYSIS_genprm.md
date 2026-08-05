# 基线深度解剖 — GenPRM(AAAI 2026)缺陷清单

**目的**: 在设计方法前,先解剖要打的 2026 SOTA baseline(GenPRM, AAAI 2026, ProcessBench-7B F1 80.5),用源码 + 真实轨迹 + 已有 findings 找出 ≥5 个具体缺陷,让方法**由缺陷驱动**而非先入为主套类比。
**日期**: 2026-08-05 | 代码: `GenPRM/src/prm_evaluation/genprm_inference.py`、`pilot/genprm_run.py` | 轨迹: `pb_gsm8k_out/`

## GenPRM 机制(一句话)
逐步验证:对第 i 步,prompt=[系统+问题+完整解答] → 强制生成 `<analyze>`(该段分析)→ `<verify>` 用 **python 代码**查错 → 执行代码 → `<output> \boxed{Yes/No}`;reward = softmax(P(Yes), P(No)) 取自那两个 token 的 logprob;temperature=0.6,majority_num 多采样取均值。评测(genprm_run.py)**顺序扫描、首个 reward<thr 即停**。

---

## 缺陷清单(7 条,证据分级标注)

### W1. 过自信、饱和的每步判决 —— 在流畅链上漏掉真实错误 【轨迹证据】
- **证据**: `pb_gsm8k_out/gsm8k_000` 轨迹:gold 首错步=step1、`final_answer_correct=false`(确有错),GenPRM `value=[1.0,1.0,1.0,1.0]` —— **四步全判满分,整案漏检**。
- **机制**: reward 只由 `\boxed{}` 里 Yes/No 两个 token 的 logprob 做 softmax(`get_reward_score`)。模型生成流畅时对每步都给 P(Yes)≈1 → 分数饱和在 1.0,失去分辨率。错误若"读起来顺"(自洽但错)就被判满分。
- **量化(需 math/olympiad 重跑)**: 分数分布直方图;在 `final_answer_correct=false` 的案例中"所有步 ≥thr 即完全漏检"的占比;分数在 [0.9,1.0] 的饱和比例。

### W2. 早承诺 —— 顺序阈值截断把定位推向过前 【源码 + 已有 findings】
- **证据**: `genprm_run.py:52` `if reward < thr: pred = i; break`——首个低分步即定为首错并停。
- **交叉印证**: 本项目 `pilot/FINDINGS_prmbench.md`:顺序"找首错"验证 67% 漏检把位置预测在**真首错之前**,长链(≥15步)漏检 0.71 vs 短链 0.35。
- **量化**: math/olympiad 上 预测首错步 − 真首错步 的偏差分布;漏检率 vs 链长曲线。

### W3. 代码验证的模态错配 —— 非计算型步骤上退化为无依据判断 【源码证据】
- **证据**: `verify_template="Let's use python code to find any potential error:\n```python\n"`——**核心查错手段是写 python 代码**。
- **问题**: 竞赛题(olympiadbench)大量步骤是证明/不等式/几何/组合**推理**,无可执行计算。代码要么写不出、要么与错误无关,verdict 退回纯 LLM 主观判断(且 W1 过自信)。
- **量化**: 计算型 vs 概念型步骤的 recall;math.json vs olympiadbench.json 的 F1 落差(预期 olympiad 显著更差)。

### W4. 有状态 CodeExecutor 的命名空间泄漏 —— 真实正确性 bug 【源码证据】
- **证据**: `CodeExecutor.__init__: self.namespace = {}` 只建一次;`execute: exec(code_block, self.namespace)`;`genprm_run.py` 全程复用**同一个 `ce` 实例**。
- **后果**: 变量跨步、**跨题**残留。后一步/后一题的查错代码可能引用到前面残留的 `x/n/result` 等变量而"跑通"或输出错值 → 误导 verdict。这是可复现的正确性缺陷,非调参问题。
- **量化**: 统计 exec 代码块中"引用了本块未定义、却在此前块定义过的变量名"的次数;构造隔离 vs 共享 namespace 的 verdict 翻转率。

### W5. 判决格式脆弱 —— 静默退回中性 0.5 【源码证据】
- **证据**: `get_reward_score`: `re.search(r'(Yes|No)\}', text)`,未命中 → `return 0.5`(WARNING);另有 max_tokens 耗尽时的**降级路径**强行截断补 `<output>`,产生仓促 verdict。
- **问题**: 1.5B / 长步骤上格式遵循率下降 → 大量"非判决"被当成 0.5 边界值(阈值 0.5 时约等于随机),污染指标且不可归因。
- **量化**: 0.5 回退率;与链长/步长的相关;剔除 0.5 案例前后 F1 变化。

### W6. 验证自身的随机自不一致 —— temp 0.6、默认 majority=1 【源码证据】
- **证据**: `TEMPERATURE=0.6`;`genprm_run.py` 默认 `--majority 1`。
- **问题**: 同一步不同采样得到不同 Yes/No。单样本 verdict 高方差;要稳定需堆 majority(=堆算力)。
- **量化**: 每步 K 采样的 verdict 翻转率;majority 1→4→8 时 F1 与算力曲线。

### W7. 长链上的算力–精度反比 【源码 + 量化】
- **证据**: 每步全套 CoT+code(max_tokens 2048)× majority × n_steps;而精度最差恰在长链(W2)。
- **量化**: tokens/案例 vs 链长;精度 vs 链长——证明"最花算力处精度最低"。

---

## 缺陷 → 方法设计约束(由缺陷倒推,而非套类比)

一个能打 GenPRM 的验证器,应针对其**可量化的失效**,满足:
1. **抗饱和/不过自信**(打 W1):产出**连续、有分辨率**的每步信号,而非 Yes/No softmax。
2. **无顺序截断、可并行**(打 W2):每步独立打分,不"首个低分即停"→ 去早承诺。
3. **免代码、跨模态通用**(打 W3):在概念/证明步上同样有效,不依赖可执行计算。
4. **确定性、低方差**(打 W6):不靠随机采样,单次即稳。
5. **廉价、随链长温和增长**(打 W7):远低于每步 2048-token 的生成成本。

**候选主方法(Detailed-Balance / 熵产,见 IDEA_REPORT.md Idea 1)恰好命中 1–5**:
- 每步 σ = logp_f − logp_b 是**连续标量**(抗 W1 饱和);
- 每步独立、可**一次性并行**打全部步(去 W2 早承诺);
- 纯 **logprob**、不写代码(免 W3 模态错配);
- teacher-forcing **确定性**(去 W6 方差);
- 每步仅 2 次前/反向打分,**远低于生成式 CoT**(缓 W7)。

> **仍需过 Gate 0**(对"自洽但错"子集的判别力 + compute-matched 打赢):W1 的"自洽但错满分漏检"正是 σ 要证明能抓的子集。存在性 pilot(PREREG_detbal.md)先在 1.5B 上测 σ AUROC 与方向归因闸①。

## 框架级贡献(顶会主张)

**共同病根**: W1–W7 都源于 GenPRM 的单一设计选择——**把验证当作生成任务**(逐步生成 critique + 离散 Yes/No + 代码执行)。

**范式切换**: **Verify-by-Measurement(测量式验证)**——不生成判决,而是从基座模型对整条推理轨迹的**自身概率分布**读出连续、确定、跨模态、全步联合的内在量。此范式 **by construction** 结构性消灭 6–7 个缺陷:

| 框架属性 | 结构性解决的缺陷 |
|---|---|
| 连续标量(非 Yes/No softmax) | W1 饱和判决、W5 格式退化 |
| 全步联合、无顺序阈值截断 | W2 早承诺 |
| 纯 logprob、适用任意文本 | W3 模态错配、W4 namespace bug |
| teacher-forcing 确定性 | W6 verdict 方差 |
| 无生成、每步仅 2 次打分 | W7 算力-精度反比 |

**框架 = 一族"轨迹测量算子",统一在一个原理下**:
- 旗舰算子 = **Detailed-Balance 熵产/不可逆性**(σ=logp_f−logp_b),针对"自洽但错"(W1 的漏检子集);
- 可扩展算子 = 每步信念曲率/熵、跨步一致性场(均为 logprob 派生、确定性)。

**诚实边界**: "结构性消灭 6 缺陷"不依赖精度、by construction 成立;**唯一经验问题 = 测量信号精度能否 compute-matched 追平 GenPRM**(Gate 0)。顶会主张 = **SOTA 缺陷分类学(W1–W7 量化)+ 在 6 个结构轴碾压且精度追平的测量式验证框架**。

## 待办(量化证据)
- [ ] 修环境(vLLM 0.7.1 × transformers 5.14.1 tokenizer 冲突)以重跑 GenPRM。
- [ ] GenPRM-1.5B 跑 ProcessBench **math + olympiadbench**(非 GSM8K),落每步 score + 预测。
- [ ] 按 W1–W7 逐项量化(分数饱和 / 早承诺偏差 / math-olympiad 落差 / namespace 泄漏计数 / 0.5 回退率 / verdict 方差 / 算力-精度)。
- [ ] 据量化结果确定主攻的 2–3 个缺陷,定稿方法。
