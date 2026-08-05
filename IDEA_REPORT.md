# Research Idea Report — 跨学科 test-time 验证方法

**方向**: LLM test-time 验证引导方法(training-free / LoRA 级),方法灵感来自化学/物理/心理学/生物的跨学科类比;目标 ACL/NeurIPS 主会;单卡(7B 推理 + LoRA)。
**生成日期**: 2026-08-05
**流程**: 10 生成 → 5 过滤存活 → 3 推荐(+ 2 备选)→ 0 已跑 pilot(待用户确认算力后跑)
**复用资产**: 本项目 `pilot/` 已判死 4 类种子 + 立起 2 条活线(见下)

---

## 治理约束(来自本项目已有 pilot,不可违反)

任何新种子必须先过 **Gate 0**:
> 对"**自洽但错**"(self-consistent-but-wrong,~38% 稳定错盆地)子集有判别力,**且**在 **compute-matched** 下打赢 self-consistency(SC)。否则判死。

**已判死(不再生成):**
1. **动力学校对 Kinetic Proofreading** — 跨两模型家族判死;检查点前缀锚回原路径,严格劣于 SC。
2. **整个"扰动看输出变不变"类**(含涨落-耗散)— 对稳定错盆地无底物。
3. **朴素每步独立验证** — 被 compute-matched SC 在平衡准确率上追平(三轮机制无一胜)。
4. **GSM8K 作底物** — 残余"自信错"7/7 是标注噪声,非模型错。

**两条活线(正密度):**
- **LIVE-1 早承诺病**:顺序"找首错"长链漏检 0.71(短链 0.35),67% 漏检定位早于真首错 = 心理学**锚定偏差**。
- **LIVE-2 共识审计 gold**:"自洽+跨模型一致+与gold不符" = 基准标注错误探测器(pilot 命中 7/7)。

**设计原则(从死法反推):** 新种子的主观测量必须**与"重采样看答案是否一致"正交**——否则不是 SC 就是已死类。要找的是**盆地内部结构信号**(basin 形状/曲率、不可逆性、多尺度不变性、约束阻挫、premise 耦合),而非"是否离开盆地"。

---

## Landscape Summary(2025–2026)

test-time 验证已从"生成即答案"转向"生成+验证"两段式,但主流验证器仍绕不开三条老路:(1)基于答案一致性的聚合(SC / 加权投票,如 ACL'26 SCOPE);(2)prefix-confidence 选择(2507.18122);(3)归一化自评置信度(2603.06604 *Know When You're Wrong*)。

对"自洽但错"的正面攻击刚刚兴起且**恰是最拥挤的雷区**:`ARBITER`(2605.26172)研究主导盆地失效、何时该选非多数盆地;`Reasoning on the Manifold`(2604.16565)提出正确解在高密度流形、错误序列 off-manifold drift(但针对 diffusion LM)。任何"盆地重排/流形一致性"想法都会被审稿人并到这两篇——**必须精确差异化或避开**。

一个关键正向信号:已有工作观测到"**低 token 熵但几何复杂度高**"是"自信错"的签名——**证明存在与答案一致性正交的可用信号**,这正是本报告推荐种子的立足点。已有热力学尝试(2606.19404 用注意力 Laplacian 当 Hamiltonian 做幻觉检测;2603.18940 熵轨迹形状)证明"热力学量做验证"这条路可发,但**都没做前向/反向不可逆性**——留下缺口。

反向/结构化纠错方向正在快速填满:`NCV`(2510.02816 node-wise 一致性、低成本长链纠错)、`Bidirectional PRM`(2508.01682)、`LEDOM`(2507.01335 反向语言模型)。"反向验证"这个词已被占,想做必须换更锐的切入(如"锚定去偏 + 只验割点")。

---

## 推荐种子(排序)

### ⭐ Idea 1(领头): Detailed-Balance Verifier — 推理路径的"熵产/不可逆性"
- **跨学科机制**: 非平衡热力学 **detailed balance / Crooks 涨落定理**。真实物理过程的可逆性由前向/反向路径概率比刻画;错误推理常是"**前向易编、反向难证**"的合理化(rationalization),沿答案关键转移的不可逆性(熵产)应偏高。
- **精确映射**: 对答案关键转移 `x_t → x_{t+1}`,用同一 7B 计算前向可行性 `p_f(x_{t+1} | x_t, Q)` 与反向可行性 `p_b(x_t | x_{t+1}, Q, A)`(teacher-forcing 强制打分),验证分数 = Σ log(p_f / p_b) = 该链的"熵产"。
- **如何过 Gate 0**: 观测量是**转移本身的不可逆性**,不是"答案是否移动"——对稳定错盆地天然有底物。前向合理化在反向重构时需要复活被省略的约束,故即使答案永不变也可分。
- **为什么它最适合领头**:
  - **pilot 最干净**:纯 logits、teacher-forcing、**无采样方差、无脆弱图解析**,信号确定性最高。
  - **精确吃满算力画像**:反向条件打分需要**全 logit 访问**,API 模型给不了 → 正是单卡 7B 开权重的用武之地(顺带成为"为何用本地 7B"的合理性)。
  - **理论章节漂亮**:Crooks 涨落定理类比,NeurIPS 审稿人吃这套。
- **最小 pilot(<1 GPU-hr)**: 100 自洽对 + 100 自洽错(**非 GSM8K**,用 MATH-hard / 竞赛题,gold 可信),每题 teacher-force 3 个关键转移的前向/反向 logprob。**成功阈值**: AUROC ≥ 0.68 在稳定错子集,或与 SC 融合后平衡准确率 +1;否则判死。
- **Novelty**: **8/10**。最近邻 `Reasoning on the Manifold`(全序列流形一致性、diffusion LM)+ 2606.19404(注意力 Laplacian 热力学)。差异:本方法是**自回归 LLM 的转移级不可逆性(前向/反向 logprob 比)**,不是全序列流形也不是谱 Hamiltonian。查新未见直接撞。
- **Risk**: MED。主风险=反向可行性判别力不足(pilot 会立刻证伪)。
- **Reviewer 会关心**: 从 logits 出的干净标量 + 一段有物理根的理论 + 直击"自信错"。

### ⭐ Idea 2: Reverse-Replay / Bottleneck Verifier — 给顺序验证去锚定
- **跨学科机制**: 海马**逆向回放(reverse replay)/ 优先扫描(prioritized sweeping)** + 心理学**锚定去偏**。不按 `1→n` 顺序扫,而是从答案反向、随机访问,只验"割点"(cut vertex)。
- **精确映射**: 从单条链抽 claim DAG,找连接 premises→answer 的割顶点,只对这些瓶颈做上下游一致性核查(反向/随机顺序,消除顺序承诺)。
- **如何过 Gate 0**: 稳定错盆地常挂在一个隐藏桥接 claim 上——直接验桥,不需要答案多样性;并且**正面命中 LIVE-1 实测的早承诺/锚定病**。
- **最小 pilot(<1.5 GPU-hr)**: 200 长链样本,1 链 + 4 割点核查 vs token-matched SC-4。**成功阈值**: 早于真首错的漏检从 67% → <45%,且长链切片平衡准确率 +2;否则判死。
- **Novelty**: **6/10(需精确卡位)**。空间拥挤:`NCV`(node-wise 一致性纠错)、`Bidirectional PRM`。差异化必须打**"锚定去偏 + 只验割点(非全节点顺序打分)"**这一具体贡献,并在实验里直接对比 NCV。
- **Risk**: MED(工程 LOW,novelty MED-HIGH)。
- **Reviewer 会关心**: "对一个**实测偏差**的原理化修复" 故事干净、算力低。

### ⭐ Idea 3: RG-Verify — 多尺度不变性验证(粗粒化后局部重合成)
- **跨学科机制**: 重整化群 **RG 粗粒化**。正确推理在粗粒化下应稳定;错误推理压缩再局部重合成时会暴露"尺度缺陷"。
- **精确映射**: 把链压成 3–4 个宏步,只重新生成相邻宏态间的答案关键桥接步,度量原链与重合成的局部不变量之间的"scale defect"。
- **如何过 Gate 0**: 稳定错盆地可对全链重采样稳定,但**缺乏有效的多尺度骨架**;粗粒化剥掉掩盖错误的语言脚手架。
- **最小 pilot(<1.5 GPU-hr)**: 150 长链,1 链 + 1 摘要 + 2 桥接重合成 vs SC-4。**成功阈值**: scale-defect AUROC ≥ 0.65 在自洽错子集,整体平衡准确率 +1.5;否则判死。
- **Novelty**: **8/10**。LLM 验证里未见直接撞(RG-for-ML 存在但非推理验证)。
- **Risk**: MED(信号确定性低于 Idea 1)。
- **Reviewer 会关心**: 新鲜的物理迁移,若增益集中在长链尤其有说服力。

---

## 备选(不进首轮预算,信号出现后再启)

- **Conservation Ledger(Noether 不变量/化学计量守恒)**: 建"账本"(实体/单位/极性/资源),每步须守恒或显式转化。**最便宜的 pilot、最可解释、跨域可移植** → 定位为**最快证伪实验**,可先跑一枪探底。Novelty 6/10(CSP-verification 已存在,靠"守恒审计"框架区分)。
- **Type-2 Foil Resolution(元认知二型信号检测)**: 对高介数步生成 minimal-pair foil,测模型区分原步 vs 近误步的**局部判别 margin**(非答案是否翻转,故不属 dead-2)。Novelty 7/10,Risk MED-HIGH。
- **Detailed-Balance + Reverse-Replay 融合**: 若两者各自出信号,融合成 3–5 标量特征喂一个 4-bit LoRA 融合头,是自然的"paper 版"升级路径。

---

## 已淘汰 / 降级(存档,省未来时间)

| 种子 | 原因 |
|---|---|
| Kinetic Proofreading | 本项目 pilot 已判死(跨两模型),严格劣于 SC |
| 涨落-耗散 / 扰动观测类 | 对稳定错盆地无底物(dead-2) |
| 朴素每步独立验证 | 被 compute-matched SC 追平(dead-3) |
| Surface-Tension Basin Reranking | 最贴 `ARBITER`,易被审稿人并入"盆地启发式";Codex 亦建议除非 #4/#5 先出信号否则不投预算 |
| Allosteric Grounding(mask premise 看 late-step logprob 掉多少) | Risk HIGH;虽观测量正交(耦合强度非答案翻转),但依赖脆弱 premise 抽取 |
| Predictive-Coding Residual Audit | 与自评置信度方法(2603.06604)边界模糊,降级 |
| Frustrated-Loop(自旋玻璃阻挫) | 机制可能塌缩为"换名的 CSP 违背计数"(Eidoku 2512.20664 已做),且依赖 7B 脆弱约束解析 |
| 任何 GSM8K 上的验证评测 | 残余错=标注噪声(dead-4);底物必须换 MATH-hard/竞赛 |

---

## 建议执行顺序

1. **先跑 Idea 1(Detailed-Balance)pilot** — 信号最确定、pilot 最干净、novelty 最高。同一批数据顺带跑 **Conservation Ledger** 作最快证伪对照(几乎零额外成本)。
2. Idea 1 出信号 → 直接进 Idea 2(Reverse-Replay)长链切片实验,并对 `NCV` 精确卡位。
3. Idea 1 判死 → 转 Idea 3(RG-Verify)。
4. 任一站住 → `/novelty-check` 深查 → `/research-review` 外部批判 → 实现 → `/run-experiment` → `/auto-review-loop`。

**统一评测底物**: MATH-hard / 竞赛题(gold 可信),**禁用 GSM8K**。**统一强 baseline**: compute-matched SC(先 SC-4,paper 版 SC-8/16)+ 至少一个 2026 顶会验证器(如 SCOPE 或 NCV)。

## 基线与复现锚点(2026-08-05 定)

**远程环境**: AutoDL, Tesla V100-PCIE-32GB(≤7B 免训练可跑);`/root/autodl-tmp` 已清理出 25G 空闲(删了 6 个已判死 judge-SFT 线的可重生 merged_* 模型,共 20G)。

**决策:不复现无代码论文。** 直接竞品 ARBITER(2605.26172)、NCV(2510.02816)**均未开源**,且 ARBITER 只在 GSM8K 评测——复现=从零重实现,对比不可信,放弃。

**评测锚点 = ProcessBench + 有代码/权重的 SOTA 基线:**
| 资产 | 角色 | 可得性 |
|---|---|---|
| ProcessBench(QwenLM 官方) | 评测集(竞赛数学,人工 gold,找首错步) | 官方代码,服务器已有 |
| Qwen2.5-Math-PRM-7B | 监督 SOTA 基线(有公开榜单) | 官方 HF 权重,V100 可跑 |
| FreePRM(2506.03570) | training-free SOTA(同哲学,最佳直接竞品) | 有代码,ProcessBench F1 53.0 |
| GenPRM + self-consistency | 生成式 PRM + 免训练下限 | 服务器已有 |

**Idea 1 与 ProcessBench 的契合**: Detailed-Balance 对每个答案关键转移给"熵产"分 → 天然产生**每步**错误信号 → ProcessBench 找首错步 = 取熵产最大的步。评测口径与 Qwen2.5-Math-PRM-7B / FreePRM 完全对齐,可 compute-matched 直比。成功判据:在 ProcessBench(尤其长链切片)F1 追平/超过发布的 7B SOTA PRM,同时免训练。

**基座决策(由 baseline 定,保证归因干净)**: Qwen2.5-Math-PRM-7B 从 `Qwen2.5-Math-7B-Instruct` 微调而来。因此我的免训练验证器**用同一基座 `Qwen2.5-Math-7B-Instruct`**——主对比 = 同一 checkpoint 下 "Qwen 训练的 PRM 头" vs "本文零训练熵产打分规则",任何差异归因于方法而非基座。这也是更强的 claim:免训练追平训练过的 SOTA PRM。跨域(多域交叉验证)为**次要迁移实验**,不得反过来决定主对比基座。FreePRM(有代码,github.com/sunlin-ai/FreePRM)作次要 training-free 对比。

## 下一步
- [x] 确认远程 GPU(V100-32G)+ 清理空间(释放 20G)
- [x] 定评测锚点(ProcessBench + Qwen2.5-Math-PRM-7B + FreePRM,均有代码/权重)
- [ ] 选并下载 Idea 1 的 7B 基座(反向 logprob 需全 logit 访问)
- [ ] 预注册 Idea 1 的 Gate 1 pilot(仿 `pilot/PILOT_PREREG.md`:存在性问题 + 判死阈值 + 防自欺闸)
- [ ] 跑 pilot(ProcessBench 子集),落 `pilot/FINDINGS_detbal.md`
