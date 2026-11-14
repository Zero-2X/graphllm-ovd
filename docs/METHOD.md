# GraphLLM v2 算法详情

## 图、候选和原始证据

节点向量为 $x_i=[\operatorname{L2Norm}(u_i);\operatorname{L2Norm}(v_i)]\in\mathbb{R}^{768}$。$u_i$ 是 256 维 DINO decoder 最后一层 query，$v_i$ 是 512 维裁片 CLIP。供体 $d$ 的具体类别取其最高原生类别分。供体原生分至少 0.42，CLIP 对该类至少 0.17 且距裁片最佳具体类至多 0.045。接收框与供体需满足 IoU 小于 0.45、CLIP 余弦大于 0.45。接收框对提名类的 CLIP 分至少 0.15，且距自己的最佳具体类至多 0.075。

图先按外观、供体分和接收框适配度保留最多两条具体类边，再按 $c_1$ 分组。执行槽是同 query 的第一个原输出且原分小于 0.35。排除相同原类、同 query 自引用和原列表里已存在的同 query 提名类。动作内去重供体，同一动作中第二个供体与第一个供体的 IoU 达到 0.50 时移除第二条。

`edge12` 顺序为 CLIP 余弦、供体原生分、供体对提名的 CLIP 分、供体跨层框稳定性、归一化长宽比差、父类标记、接收框交叠比例、供体框交叠比例、相对中心 x/y、对数宽高比例。具体类路径中的父类标记为 0。`native5` 是原类与提名类的 DINO logits、相应 CLIP 分及接收框跨层稳定性。`native16` 另含两类各四项检测覆盖证据，以及整框裁片和原生注意力裁片的类别支持与分差。固定风险头使用其原统计量和预处理。 [图构建](../experiments/hierarchical_instance_v1/cache.py) · [候选与边](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/native_bridge.py) · [固定配置](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/config.json)

## 接收框、供体和空参考的读出

DINO 的 Swin stage3 对应 encoder memory 的 level 1，名义 stride 16。在预测框内取 3×3 相对网格，只用有效特征位置通过双线性插值读出 9 个 256 维向量。坐标先转换到有效图像区域，按 valid ratios 获得正确的带 padding 坐标。用同一采样算子获取的有效掩码权重进行归一化。

Qwen 隐藏宽度为 896。接收框前缀包含 14 个有效 token。

| 位置 | 内容 |
| --- | --- |
| 0–1 | 原类和提名类文本嵌入与角色 |
| 2 | 接收框全局投影 |
| 3 | 接收框 native5 投影 |
| 4–12 | 9 个接收框局部投影及网格位置 |
| 13 | 接收端 BASE 读出 |

BASE 的末层隐藏状态经 MLP 产生 $z^0=(z_{c_0},z_{c_1},z_{other})$。每条供体独立追加 12 个 token，依次是供体全局和类别、9 个供体局部、edge12、REL 读出。空参考保持相同的序列模板、位置及提名类文本，供体视觉输入和 edge12 置零。两种后缀都读位置 25。因果掩码使 BASE 无法读取后缀。实现将 BASE 读取的 12 个未来位置屏蔽，使 BF16 共用同一序列长度和运算形状。 [局部特征与张量](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/native_bridge.py) · [实际读出](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/model.py)

## 关系核验和视觉消息

关系头读取每条边的真实 REL 状态，输出 `support`、`contrast`、`invalid_ref` 三类。由真实和空参考的 REL 状态得到局部查询 $q_d$、$q_\varnothing$。接收框网格的键含局部视觉和相对位置，值 $v_j$ 只用接收框视觉。令 $\beta_{dj}$ 为在九个真实位置归一化后的注意力，则

$$
m_d = \sum_j (\beta_{dj}-\beta_{\varnothing j}) v_j.
$$

比较查询在真实位置和 NULL 位置上的分数，得到非 NULL 质量 $s_d$。实际边权为 $g_d=s_d(1-\operatorname{stopgrad}(p_{d,invalid}))$。使用原有效供体数 $D$ 归一化，$M=\sum_d g_d m_d/D$。接收框语义改变量来自一个有自身条件的有符号头，$z^f=z^0+\operatorname{center}(F(M,h_{BASE}))$。该头满足 $F(0,h)=0$，且输出投影从零初始化。供体读取变化影响接收框证据，独立的关系概率只用于内部参考采用量。 [消息实现](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/model.py)

## 标注与三阶段优化

语义标签描述接收框。相关类别标注覆盖且无 crowd 冲突时，目标所属类由与 GT 的 IoU 核验。命中原类、提名类或第三类需要对应 IoU 至少 0.50，竞争类的证据至多 0.30。关系标签描述供体来源及对象关系。确认同一实例重复参考或错误声明时标为无效；确认不同实例且供体声明正确时，接收框同类为支持，接收框属于别类为对比。核验不足的语义、关系和收益字段保留 ignore。

A 训练三态语义并冻结最佳检查点。B 训练语义、关系、坏边损失三项等权；全部为坏边时，约束消息和语义回退，自身语义标签不驱动融合改类。混合好坏边时，只约束坏边的消息及影响，删除贡献时固定原来的供体数。训练可从原边池给同一动作加一个替换供体视图。分层权重先按类，再按图、动作和视图/边分配。A/B 的最大训练轮数均为 12，按监控损失、耐心 3 选最佳点。

C 输入 147 维特征，包含 native16、风险 logit、$z^f_{c_1}-z^f_{c_0}$、第三类概率、128 维消息 $M$。模型为 147→32→1，每条动作的 full/off 视图各占一半权重。收益正例要求三态核验改对，至少在 IoU 0.50 或 0.75 的最低分档新增一个匹配，且六档均无匹配损失。C 固定 120 步，用已核验动作拟合加权二元交叉熵。 [真值和策略](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/targets.py) · [训练实现](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/train.py)

## 推理与结果核验

在至少有一条实际可用供体且结果有限时，只有 $z^f_{c_1}>\max(z^f_{c_0},z^f_{other})$ 的动作进入排序。每图按收益 logit 取唯一首位，并以规范动作键打破并列。策略集按同一排序检查全局前缀，要求已知损害为 0、六档匹配丢失为 0、未确证至多 11。最终收益分界锁定为 2.6715939045。留出集的未确证动作依然单列。

JS 用单边决策分布的混合熵减去平均熵，只记录在有两个供体的候选上。本版预测、训练和收益不读取该值。消息置零组复用同一候选、供体存在性、权重与分界，令 $M=0$ 后重新执行全部资格、排序和门控。预测阶段只读图像清单，在预测哈希封存后再读取留出标注。 [推理执行](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/predict.py) · [评估](../experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/evaluate.py)

原始设计包含全套损失、监督掩码、配置与检查规则。 [审查后实施方案](design/2026-11-13-receiver-evidence-v2.md)
