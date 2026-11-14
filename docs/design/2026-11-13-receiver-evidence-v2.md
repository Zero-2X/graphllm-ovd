# GraphLLM v2：关系核验与接收端证据补充
## 审查后的完整 Codex 实施规范

**版本：v2.0｜日期：2026-11-13｜状态：方法设计已审查；实际仓库待实现、待训练、待验证**

**交付目标：**保留 GraphLLM 主线，将“相似供体投票／分歧门控”改为“核验参考关系 → 供体引导接收端局部证据读取 → 有符号消息修正 → 判断检测集合中的实际收益”。先搭建完整链路，再完成一次受控训练和少量验证。

**证据边界：**本文依据原流程文档、上一版实施方案及用户提供的 2,048 图汇总结果编写；没有检查用户实际 Windows 仓库、逐动作日志或 GPU 训练结果。文中的新结构、损失与参数是本轮固定的设计，不是已验证最优配置。审查能排除已发现的逻辑矛盾，不能保证没有任何未知问题，也不能保证 AP 提升。

**版本关系：**本文是新的 v2 规范。与上一版冲突时，v2 的公式、监督、训练资格及执行规则优先；旧版本作为不可改写的历史基线保存，不能把两版规则拼接使用。

---

## 0. 直接交给 Codex 的任务

> 请读取仓库 AGENTS.md、本文及原流程文档，定位实际源码、权重、数据清单、冻结策略和缓存。按照本文完成实际实现，不只给建议。
>
> 顺序固定为：**盘点并锁定配置 → 完整工程搭建 → 必要正确性检查 → BUILD_COMPLETE → self 训练并冻结 → 关系与消息训练并冻结 → 收益头拟合并冻结 → 一次 policy 锁定 → 最多四臂的一次比较 → 交付。**
>
> 在 BUILD_COMPLETE 前，必须把输入、标签、模型、全部损失、冻结阶段、缓存、策略选择、无 GT 推理和独立评估器接通。允许小批量形状／梯度检查；不得先试一部分效果，再依据测试结果持续添加模块。
>
> 常规代码修改、路径适配、必要检查、本文明确的训练和评估可以直接执行，无需逐步请求确认。缺少权重、标注核验规则、数据或必要运行条件时，完成不依赖缺项的构建，报告具体阻塞，不伪造结果、不暗换方法。
>
> 本轮只实现一个 v2，使用 seed=42，依次训练三个阶段，只锁定一次最终策略，最多四个评估臂。不要追加种子、阈值网格、大规模消融、更多数据集或无上限续训。结果无提升也按停止规则交付。
>
> 本文给出的模块名是建议接口，历史文件名只是定位线索。优先复用实际仓库的公共实现，交付真实路径和可执行命令。

### 0.1 必须保持的边界

| 项目 | 本轮要求 |
| --- | --- |
| 主线 | 实例图提名跨实例参考；冻结 Qwen 的 GraphLLM 负责关系读取与条件化证据查询 |
| 检测 | 冻结 Grounding DINO，沿用原生尺度、全词表与单次原生前向 |
| 视觉 | 冻结原 CLIP；局部信息来自同次 DINO 的空间特征，不新增视觉主干 |
| 语言模型 | 冻结 Qwen2.5-0.5B，隐藏维 896；训练适配器、查询与小型任务头 |
| 候选 | 同图、原具体类候选规则；每 receiver 最多两条具体类边 |
| 执行 | 每图最多改变 D₀ 中一个既有槽的类别 |
| 输出 | 框、检测分数、槽号、列表长度、其他槽位保持不变；不重跑 NMS |
| 不执行 | 无合格动作则返回原 D₀ |
| 排除 | 未输出 query 补回、改框、检测重打分、多跳图、局部 DINO 重跑、LLM 生成文字决策、额外骨干、图拉普拉斯损失 |

“单次前向”约束部署路径：一个图像不能因 receiver/donor 或裁片数量而重复运行 DINO。补建缺失的空间缓存可以对每图做一次完整原生特征提取，应记录这项预处理成本；不得宣称旧缓存已经包含未经核对的新特征。

---

## 1. 已知结果能够说明什么

### 1.1 用户提供的历史结果

| 方法 | 执行改类 | 确认改对 | 原类被改错 | 未确证 | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生 Grounding DINO | 0 | 0 | 0 | 0 | 34.7067 | 46.8166 | 37.6022 |
| 旧门控 | 50 | 29 | 4 | 15 | 34.7077 | 46.8162 | 37.6039 |
| v1，混合熵开启 | 71 | 38 | 3 | 29 | 34.7075 | 46.8158 | 37.6048 |
| v1，混合熵关闭 | 71 | 38 | 3 | 29 | 34.7075 | 46.8158 | 37.6048 |

上述数字来自用户汇总，本文没有重新计算：

- v1 比旧门控多确认改对 9 次、少原类误改 1 次；执行总数增加 21 次，未确证总数增加 14 次。具体新增、消失和替换动作的组成仍须逐动作配对核验。
- JS 改变了 7 个候选的语义资格，却没有改变任何最终动作；这轮没有证明混合熵产生检测收益。
- v1 相对 D₀ 的 AP 增量为 **0.0008 个百分点**；不能写成 0.0008 的 0–1 AP 增量。
- 38 次确认改对中，9 次同时形成新增且保住已有匹配；相对于原报告 135 张机会图，约为 6.7%。这个分母需要用原日志核对。
- 最佳语义权重在第 5 epoch，未通过旧资格检查。它属于探索性历史结果；仅凭 epoch 数无法判断是否真正优化失败。
- “原类被改错”不是全部有害行为。还须核对其他原匹配损失。
- 旧门控的 29+4+15=48，未覆盖 50 次全部动作；v1 的 38+3+29=70，未覆盖 71 次。可能存在另列中性状态，必须读日志对账，不能直接认定缺的动作安全。

### 1.2 本轮工作假设

最值得修复的是：**模型是否真的学习了“某个 receiver 应从某份参考中获得何种增量”，以及这个增量是否转化为检测集合收益。**

训练数量增加可能帮助学习，但不能自动解决输入没有局部信息、标签含义混杂、损失目标冲突、消息被公式限制、收益排序不响应消息等问题。当前汇总结果也不足以在这些原因中确定唯一根因。

本轮据此设计一条可验证的改进路径，而不是承诺“扩数据就会好”或“换一个熵公式就会好”。

---

## 2. 保留的流形假设与准确的方法主张

继续使用以下表述：

> 在任务相关的实例表征空间中，对象范围相容、供体可靠且关系适合类别信息传递的局部邻域，具有一定的语义平滑性。我们利用这一局部结构寻找跨实例证据，由 GraphLLM 判断关系的适用性并生成条件化消息，同时保留接收实例自身证据，以限制错误邻居和类别边界处的负迁移。

这里是**带条件的局部语义先验**。它支持寻找参考，不意味着所有相似框同类，不证明真实数据满足特定可恢复流形，也不要求图上所有预测一致。流形正则研究只提供背景，不提供本方案的有效性定理。[R1]

v2 中各部分的含义：

| 用户目标 | 操作定义 |
| --- | --- |
| 核验供体 | 检查其原生声明是否与可核验对象相符，并训练来源／关系类别 |
| 核验关系 | 区分同类参考、对比参考和本任务不允许的参考 |
| receiver 缺少的证据 | 当前 self 判断未充分利用、但仍存在于 receiver 冻结局部特征中的辨别信息 |
| 供体补证 | 供体改变 GraphLLM 的局部查询，重新读取 receiver 的视觉 value，形成相对同模板 NULL 读取的增量 |
| 保留自身证据 | 先训练并完全冻结 self；最终 logits 为 self 加有符号修正 |
| 检测收益 | 对固定框、固定分数、固定槽的实际改类进行集合回放与 AP 评估 |

**“补充”不等于凭空恢复视觉信息。** 如果目标的区别信息未被检测到、被裁片彻底丢失，或冻结特征已不能表征，供体无法可靠证明 receiver 就具有该属性。v2 允许拒绝或零消息。

GraphLLM 使用 donor 的视觉、原生声明和边信息生成条件查询；最终消息 value 来自 receiver。这个结构加强了接收端依据约束，但不能把 attention、低 NULL 权重或 support 标签直接当作真实因果证据。局部对应工作提供了设计背景，不能直接证明本任务有效。[R2]

---

## 3. 审查后的统一决策

| 已发现或需规避的问题 | v2 的唯一处理 |
| --- | --- |
| 同类被当成“必有帮助” | support 只表示可核验的同类参考，允许零增量 |
| 异类可靠供体被当作坏供体 | 设 contrast，可提供负向证据，不强制回退 |
| 未确证被塞进 other 或 invalid | 语义与关系都保留 ignore，单独掩码 |
| self 与消息分支 receiver 信息不对等 | 两者共享同一完整 receiver 前缀和 3×3 局部特征 |
| self 只是 detach，但共享模块仍更新 | 阶段一后冻结完整前缀、读出头、统计量与输入规范 |
| donor 读出减 BASE，混入角色和读出深度差异 | 相减的是相同 donor 后缀模板的真实／NULL 局部读取；BASE 仅负责固定自身预测 |
| raw GraphLLM hidden 绕过接收端证据 | 禁止直接进入语义残差头或 gain 头 |
| NULL 质量在读出与 gate 中乘两次 | 真实 patch 单独归一化；非 NULL 质量只乘一次 |
| 单独 donor logits 的凸组合及 B=2 限制 | 改为局部特征消息聚合后产生直接有符号残差，不沿用硬幅度截断 |
| 不可靠边与好边共同进入 Qwen，无法隔离坏边 | 每条边独立关系读取；只在消息特征层聚合 |
| 错误供体回退与纠正错误 self 的 CE 冲突 | 全坏 action 禁用 fused CE；混合 action 只约束坏边贡献 |
| 坏边只在语义头不生效，却仍经 gain 生效 | 坏边损失同时约束被 gain 消费的消息本体及语义输出影响 |
| JS 没改变最终动作且缺少方向性 | JS 只诊断，不进入训练损失、资格、幅度或排序 |
| 旧风险 cap 可能继续堵住新版工作点 | 兼容时复用冻结风险 logit 作为 gain 输入，不继承硬 cap |
| gain 只有线性读出、难表达证据和覆盖的交互 | 使用固定小 MLP，输入包含原覆盖证据和实际消息 |
| 早停与完整 cosine／末五轮平台条件冲突 | 统一新训练资格；合法早停不再判流程失败 |
| 消息关闭仅改日志或语义，不改收益路径 | 同时关闭实际消息、语义增量，并重算 gain、资格和最终动作 |

本轮不加入覆盖节点、不做多供体长序列、不加第二套风险模型。对象覆盖继续由既有 native16、收益标签与独立集合回放处理。

---

## 4. 数据身份、候选与输入契约

### 4.1 固定动作身份

~~~python
ActionKey(
    image_id,
    query_id,
    slot_id,          # 原始 D0 的槽号
    original_class,   # canonical c0
    proposed_class,   # canonical c1
)
~~~

所有类别比较使用 Objects365 规范类别 ID。显式保存：

- 检测文本索引 → 规范类别 ID；
- 规范 ID → 原类别文本；
- 已有别名映射；
- 图节点 query ID → 原 D₀ 槽号映射。

不假定官方 ID 连续，不用字符串近似匹配合并类别。映射后 c₀=c₁ 就不是改类动作。供体声明始终是原生预测 `c_d_star=c1`，不能用 GT 重贴标签。

### 4.2 沿用的候选设置

| 项目 | 固定设置 |
| --- | --- |
| DINO | 原生 800/1333 尺度；900 queries；原 365 类及辅助父类提示和 token 映射 |
| D₀ | 具体类分数 ≥0.05；类内 NMS IoU 0.50；最多 100 项 |
| 图节点 | 类无关 NMS IoU 0.82；最多 128 节点 |
| 供体 | 原生提名分 ≥0.42；CLIP 适配 ≥0.17；距裁片最佳类 ≤0.045 |
| receiver–donor | 框 IoU <0.45；CLIP 余弦 >0.45 |
| receiver 对 c₁ | CLIP ≥0.15；距其最佳类 ≤0.075 |
| 改类槽 | query 在 D₀ 中的首个原输出槽，原分 <0.35 |
| 图边上限 | 每 receiver 最多两条具体类边，再按 c₁ 分组成动作 |

原边排序和其他原资格保持一致；不能将“两条边”改成“每个提名类两条边”。

执行前排除无效框、非有限必要特征、同 query 自供体、c₀=c₁，以及 D₀ 其他槽已存在的同一 `(query_id,c1)`。

动作内按原边序去重 donor query；若第二 donor 与已保留 donor 的 IoU≥0.50，舍弃第二条，不从截断外补边。此处只提高可区分性，不建立统计独立性。

父类边不进入本轮改类路径。不同 query 的同类重叠不新增硬拒绝阈值，仍由原覆盖证据与收益决定。

### 4.3 全局与原生证据

节点：

$$
x_i=[\operatorname{L2Norm}(u_i)_{256};
     \operatorname{L2Norm}(v_i)_{512}]\in\mathbb R^{768}.
$$

receiver 公共 5 维：

$$
s_r=[\ell_r(c_0),\ell_r(c_1),f_r(c_0),f_r(c_1),\mathrm{stable}_r].
$$

ℓ 是 DINO 原始 logit；f 是原整裁片 CLIP 相似度；stable 是末层框和前一层参考框 IoU。训练集合的原始动作各计一次拟合均值／标准差，std 下限 0.01，标准化后截到 [-6,6]。不能用 policy/test 统计，也不因训练配对重复而重计统计量。

### 4.4 edge12，顺序固定

| 索引 | 数值 |
| ---: | --- |
| 0 | receiver 与 donor 的 CLIP 余弦 |
| 1 | donor 的原生提名分 p_d(c₁) |
| 2 | donor 的整裁片 CLIP 适配 f_d(c₁) |
| 3 | donor 框稳定性 |
| 4 | min(3, abs(log aspect_r − log aspect_d))/3 |
| 5 | parent_flag=0 |
| 6 | 交集面积 / receiver 面积，即 IoA_R |
| 7 | 交集面积 / donor 面积，即 IoA_D |
| 8 | (cx_d−cx_r)/w_r |
| 9 | (cy_d−cy_r)/h_r |
| 10 | log(w_d/w_r) |
| 11 | log(h_d/h_r) |

统一归一化 xyxy；无效宽高不建边；合法分母下限 1e-6；6–7 截到 [0,1]，8–11 截到 [-3,3]。本版 edge12 不再包含 receiver-only 的 DINO/CLIP 分数或稳定性，因其已在公共前缀。

这些特征提供范围关系线索，不证明 part/whole 或同一对象已经核验成功。

### 4.5 receiver/donor 的局部视觉特征：唯一默认

为每个图节点提取 `local9:[9,256]`，来自**预测框**，不使用 GT 框。

1. 在同次冻结 DINO 前向中，取送入 decoder 的图像 encoder memory；利用 `spatial_shapes` 和 `level_start_index` 还原空间层。
2. 固定选原配置中名义 stride=16 的层，并保存实际层索引与张量来源。通过模型配置确认层身份，不能凭数组下标猜测。若现有实现不暴露该张量，在对应前向增加只读缓存钩子；不要改为另一个视觉模型或临时改层。
3. 在预测框内取 3×3 等距网格中心：
   $$x_j=x_1+(j+1/2)(x_2-x_1)/3,\quad
     y_k=y_1+(k+1/2)(y_2-y_1)/3.$$
4. 使用该层实际的 `valid_ratios`，将相对未填充图像的归一化点映射到填充特征图：
   $$g_x=2x_j\,\mathrm{valid\_ratio}_{x}-1,\quad
     g_y=2y_k\,\mathrm{valid\_ratio}_{y}-1.$$
   按 DINO 现有坐标语义确认预测框归一化基准；不能再叠加一次 padding 比例。
5. `grid_sample` 固定 bilinear、align_corners=False、padding_mode=zeros。对 `feature×valid_mask` 和 `valid_mask` 分别采样，再除以采样到的有效权重，避免边缘 padding 被当成视觉证据；权重须 >1e-6，否则该节点局部特征无效。
6. 九个点按行优先排列；另外保存框内相对位置 `[(j+.5)/3,(k+.5)/3]`，位置只用于 token 和 key，不进入视觉 value。

必须做一次解析坐标检查：矩形 ramp 特征、不同尺寸和 padding 下采样点与预期对应。只需合成数据检查，不做新效果实验。

原文提到“框周边 5×5 编码器特征”，不能未经检查就将它当作本规范的框内 3×3。只有缓存张量来源、空间范围、坐标、层与采样方式都能证明等价时才可复用；否则新增 `local9_predbox_stride16_v2` 缓存。

小框的 9 个点可能高度相似。这是信息限制，不通过给 value 添加位置来伪造视觉差异。

---

## 5. GraphLLM 的完整结构

### 5.1 总体数据通路

~~~mermaid
flowchart TD
    R["receiver 全局、局部和原生证据"] --> B["冻结后的 self 前缀与三态预测"]
    R --> P["逐边 GraphLLM 关系读取"]
    D["实际供体与 edge12"] --> P
    N["同模板 NULL 参考"] --> Q["条件化局部查询差异"]
    P --> Q
    R --> V["receiver 视觉 value"]
    Q --> M["门控局部消息"]
    V --> M
    M --> F["有符号语义修正"]
    B --> F
    M --> G["收益判断"]
    F --> G
    C["原生覆盖与冻结风险输入"] --> G
    G --> A["固定槽位的一次改类"]
~~~

图中不允许增加 `h_REL → gain` 或 `h_REL → 语义残差` 的直接连线。

冻结BASE hidden进入自身头H₀，以及进入残差头的自身条件门控W_b，均是允许的。禁止的是联合REL hidden直接决定语义残差或gain；gain的全部输入严格限于第10.2节。

### 5.2 14-token receiver 前缀

| 位置 | 内容 |
| ---: | --- |
| 0 | Text(c₀)+role_C0 |
| 1 | Text(c₁)+role_C1 |
| 2 | receiver 全局 token |
| 3 | receiver native5 token |
| 4–12 | 9 个 receiver 局部 token，行优先 |
| 13 | BASE 读出 token |

Text(c) 沿用 `specific class: {name}` 的冻结 Qwen 输入嵌入掩码均值，不调用生成接口。所有序列用显式 attention mask 和从 0 连续的 position_ids。

前缀适配器：

- `P_Rglobal: LN(768) → Linear(768,896)`。
- `P_Rlocal: LN(256) → Linear(256,896)`。
- `P_native: Linear(5,896)`。
- `P_Rpos: Linear(2,896,bias=False)`。
- 对 global/local/native 投影输出再做**无可训练仿射项**的 LN(896)，记视觉输出为 φ；乘冻结词嵌入权重标准差 s_E。
- local token 为 `s_E*(phi_local + P_Rpos(xy_relative)) + role_Rlocal`。
- global/native 分别加自己的 role；独立 role 与 BASE 初始化为 N(0,s_E²)。
- 类别文本嵌入保持原值，类别 role 独立可训练。

读取最后一层位置 13 的 `h_base∈R896`；自身头：

$$
z^0=H_0(h_{\rm base})\in\mathbb R^3,\quad
H_0=\mathrm{LN}(896)\to\mathrm{Linear}(896,256)
\to\mathrm{GELU}\to\mathrm{Linear}(256,3).
$$

三态固定按 `[c0,c1,other]` 排列。

阶段一训练前缀及 H₀。阶段一选定最佳权重后，**上述所有模块、角色、BASE、统计量和文本规范完全冻结**。

### 5.3 每条边的 12-token 后缀

接在同一个 receiver 前缀后：

| 位置 | 内容 |
| ---: | --- |
| 14 | donor 全局 token + Text(c₁) + role_Dglobal |
| 15–23 | donor 的 9 个局部 token及相对位置 |
| 24 | edge12 token |
| 25 | REL 读出 token |

`P_Dglobal/P_Dlocal/P_Dpos` 从阶段一对应 receiver 投影复制初始化，随后独立训练，不能共享仍会更新的参数。`P_edge: Linear(12,896)`，其输出用无仿射 LN 后乘 s_E；其余投影缩放与前缀一致。donor role、edge role、REL 独立初始化并训练。

Qwen 为原有 causal decoder。位置 13 不能读取后面的 donor。实际代码必须验证 causal mask、位置及冻结前缀保证 `h_base` 不随 donor 内容改变；不能只依赖“它应该是 causal”。

每条边独立读取一个 26-token 序列，取最后层 `h_d=h[25]`。最多两条边，不让某个坏 donor 通过另一边的上下文影响消息。

### 5.4 同模板 NULL 参考：避免读出差异伪装成供体增量

每个 action 另外读取一次相同的 26-token 序列：

- receiver 前缀完全相同；
- donor global 的**原始输入向量**置零；
- donor local 的 9 个**原始视觉向量**置零；
- edge12 原始向量置零；
- donor 类别文本仍为同一个 c₁；
- 后缀的投影、role、相对网格位置、REL、attention mask 和 position_ids 全部与真实边相同；
- 不加 is_null 标志，不更换读出 token，不用另一套参数。

得到 `h_∅`。这是局部查询的参考，不是新的语义 teacher；固定语义 teacher 始终是 z⁰。

禁止直接以 `h_base` 的局部查询作为相减基准：BASE 和 REL 的位置、上下文及训练状态不同，这会把额外 receiver 读取能力混进供体增量。

NULL 与真实边都经过当前可训练后缀，因此训练阶段不能缓存已经 detach 的 `h_∅`。推理时同一 action 的 NULL 可被其两条边共用。

如果完全忽略 donor 相关输入，真实／NULL 读取应相同，消息应为零。这个对齐消除了一种结构捷径，但不能证明所有 donor 存在性或数据偏差捷径均已消除。[R3]

### 5.5 关系头

$$
p_d^{rel}=\mathrm{softmax}(H_{rel}(h_d)),
$$

其中 `H_rel=LN(896)→Linear(896,128)→GELU→Linear(128,3)`，类别顺序：

1. `support`：同类参考；
2. `contrast`：可信对比参考；
3. `invalid_ref`：本任务不允许的参考。

这不是“增益／无增益”三分类，也不是校准过的真实可靠概率。其标签由第 7 节定义。

### 5.6 供体引导 receiver 读取视觉 value

令 φ_rj 是**冻结 receiver 局部投影输出的纯视觉分量**，不含 role、位置、类别文本或 native5。

设 d=128：

$$
q_d=W_q\mathrm{LN}(h_d),\quad
q_\varnothing=W_q\mathrm{LN}(h_\varnothing),
$$

$$
k_j=W_k\phi_{rj}+W_{pos}\xi_j,\qquad
v_j=W_v\phi_{rj}.
$$

W_q:896→128，W_k/W_v:896→128，W_pos:2→128。四者使用无 bias 的线性层；查询前的 LN 固定无仿射项。位置 ξ 使用同一相对网格。**V 中不能出现位置、类别、供体或原生分数。**

真实 patch 的分数：

$$
a_{dj}=\frac{q_d^\top k_j}{\sqrt{128}},\qquad
a_{\varnothing j}=\frac{q_\varnothing^\top k_j}{\sqrt{128}}.
$$

在 9 个真实 patch 内单独 softmax：

$$
u_d=\sum_j \mathrm{softmax}_j(a_d)v_j,\quad
u_\varnothing=\sum_j \mathrm{softmax}_j(a_\varnothing)v_j,
$$

$$
m_d=u_d-u_\varnothing\in\mathbb R^{128}.
$$

m_d 表示 donor 条件引起的局部读取变化。若两次查询相同，或所有纯视觉 value 相同，m_d 应为零（浮点误差范围内）。

### 5.7 NULL 质量和关系 gate

另设可学习 `k_NULL∈R128` 与标量 b_NULL：

$$
a_{d,NULL}=q_d^\top k_{NULL}/\sqrt{128}+b_{NULL},
$$

$$
s_d=\sigma\left(\operatorname{logsumexp}_{j=1}^9 a_{dj}
-a_{d,NULL}\right),
$$

$$
g_d=s_d\left(1-\operatorname{stopgrad}
p^{rel}_{d,invalid}\right),\qquad t_d=g_dm_d.
$$

k_NULL 正态 std=1/√128；b_NULL 初始化为 ln(9)。s_d 等于“9 个真实位置加一个 NULL”softmax 中的非 NULL 总质量。u_d 已在真实位置内归一化，所以 s_d 只乘一次。

训练中，消息路径对 p_invalid detach，使关系头的专属参数由关系监督学习；共享 h_d 仍可能受到其他损失影响，不能据此声称绝对杜绝门控投机。推理的数值定义不变。

不再乘 `p_d×stable_d`、JS 或额外熵权重；这些来源信息已经作为候选条件和关系输入存在。g_d 是内部软采用量，不解释为独立校准的可信概率。

### 5.8 聚合与直接有符号修正

对于实际存在且通过非 GT 输入检查的 D∈{1,2} 条边：

$$
M=\frac1D\sum_{d=1}^D t_d.
$$

固定分母 D，不按预测 gate 重新归一化。gate 全零时 M=0；单供体的绝对采用量也不会被归一化为 1。

定义：

$$
F(M,h_0)=W_o\left[
\mathrm{GELU}(W_mM)\odot
\sigma(W_b\,\mathrm{LN}(h_0))
\right],
$$

$$
\delta=\operatorname{center}(F(M,h_{\rm base})),\qquad
z^f=z^0+\delta,
$$

其中：

- W_m:128→128，无 bias；
- W_o:128→3，无 bias，**全零初始化**；
- W_b:896→128，可有 bias；输入 h_base 固定；
- center(x)=x−mean(x)，只去除不影响 softmax 的公共常数；
- 不能在消息支路插入带非零偏置的 LN、额外常数通路或 raw h 拼接；
- 聚合 M 后才过非线性头，不平均独立 donor 类别 logits；
- 不再使用 `z_d−z_self`、B=2 截断或旧原生 margin 偏置。

因此 F(0,h₀)=0，显式关闭消息时 zᶠ=z⁰。零初始化保证阶段二起点保持自身预测。

该形式允许正向或负向纠正，解除旧凸组合与固定 margin 上限，但不提供无限纠错能力或安全保证。梯度裁剪、权重衰减与监督控制优化，不等价于旧硬截断。

平均聚合仍可能出现抵消；最多两条边且多数单边，本轮不为此再加一套注意力聚合。一个坏候选即使消息为零，固定 D 仍可能稀释好消息；本版不声称“删除坏候选后系统完全不变”。

### 5.9 主前向参考伪代码

~~~python
def semantic_forward(action, model, message_off=False):
    # action 不得含任何 GT/target/known_harm 字段。
    prefix, phi_r = model.frozen_receiver_prefix(action)
    h_base, z_base = model.read_base(prefix)  # 固定 self

    if not action.donors:
        return SemanticRecord(z_base=z_base, z_final=z_base,
                              message=zeros(128), delta=zeros(3),
                              has_raw_donor=False)

    if message_off:
        # 候选及 raw donor 可执行性仍按同一规则检查。
        return SemanticRecord(z_base=z_base, z_final=z_base,
                              message=zeros(128), delta=zeros(3),
                              has_raw_donor=True)

    h_null = model.read_rel(prefix, donor_inputs="zero_raw_inputs")
    keys, values = model.receiver_keys_values(phi_r)
    q_null = model.query(h_null)
    u_null = real_patch_softmax(q_null, keys) @ values

    applied_messages = []
    edge_records = []
    for donor in action.donors:
        h = model.read_rel(prefix, donor_inputs=donor)
        rel_logits = model.relation_head(h)
        p_rel = rel_logits.float().softmax(-1)
        q = model.query(h)
        real_scores = dot_scores(q, keys).float()
        beta = real_scores.softmax(-1)
        u = beta @ values.float()
        s = sigmoid(logsumexp(real_scores) - model.null_score(q))
        gate = s * (1.0 - p_rel[2].detach())
        m = u - u_null
        t = gate * m
        applied_messages.append(t)
        edge_records.append((rel_logits, s, gate, m, t))

    M = stack(applied_messages).mean(0)
    delta = center(model.correction(M, h_base))
    return SemanticRecord(
        z_base=z_base, z_final=z_base + delta,
        message=M, delta=delta, edge_records=edge_records,
        has_raw_donor=True,
    )
~~~

该伪代码省略 batching 和缓存，只定义计算契约。实际训练中 prefix 固定仍可参与后缀 Qwen 前向；**不能把整个 Qwen 调用包进 no_grad**。所有 gate、softmax、损失、最终排序用 FP32。

---

## 6. 混合熵的处置

本轮 `js_affects_decision=false`，JS 不进入损失、gate、语义资格、gain 特征、图内排序或 cutoff；也不新增开关 JS 的评估臂。

保留诊断时使用实际单边消息而非另一条聚合 donor 路径：

$$
z_d^{single}=z^0+\operatorname{center}(F(t_d,h_{\rm base})),
\quad p_d=\mathrm{softmax}(z_d^{single}),
$$

$$
J=H\!\left(\frac1D\sum_dp_d\right)-\frac1D\sum_dH(p_d).
$$

温度固定 1，D=2 时计算；D=1 时 `js=null, js_available=False`。不能把单供体的缺少分歧观测写成“高度一致”。这里单边诊断不等于完整两边前向，也不用于修正完整前向。

本轮退出 JS 门控的理由是：它对这轮最终动作没有作用，且分歧本身不能回答哪个供体可信、哪种信息对 receiver 缺失。它可以作为观察量保留；不能通过最小化 JS 强迫不同条件下的预测相同。

---

## 7. 标签与监督掩码

### 7.1 必须分开的监督对象

~~~python
semantic_target       # -1 / 0 / 1 / 2，receiver 的 c0/c1/other
relation_target[d]     # -1 / 0 / 1 / 2，support/contrast/invalid_ref
owner_gt_id           # 只用于训练和评估核验；不进入模型
owner_known
coverage_verified
crowd_clear
legacy_utility_known       # 原核验字段，只供迁移核对
legacy_confirmed_correct   # 原语义确认字段，不是旧 action MLP 的二分类 target
utility_known              # v2 统一核验状态
confirmed_correct          # v2 GT语义确认；不是模型预测正确
original_class_harmed
known_harm
new_matches[6]
lost_matches[6]
gain_target           # -1 / 0 / 1
~~~

语义正确、关系相容、动作新增匹配是不同标签。不能由“没有新增”推断 receiver 语义错误，也不能由“同类”推断 donor 必然有帮助。

### 7.2 可核验前置条件

复用原项目的标注覆盖核验器，必须定位并记录它的真实规则：

- 对相关类别，标注覆盖可核验；
- 相关 crowd/ignore 处理与原核验一致，其中已定义的 crowd 对框覆盖须 <0.50；
- 常规非 crowd、非 ignore GT 用于对象和类别 IoU；
- 缺失某类标注不自动表示该类不存在；
- 无法重建覆盖核验规则时，完成标签接口并报告阻塞，不用“都视作完整标注”替代。

`g*` 为最大 IoU 的常规 GT；`u_c` 为与类别 c 的常规 GT 最大 IoU，无该类可核验常规 GT 时按既有核验器处理。

类别可核验和对象身份可核验分开：多个不同 GT 均与该框 IoU≥0.50 时，本版 `owner_known=False`；即使它们同类，也不能据此确定两个 proposal 是不同实例。

### 7.3 receiver 三态语义标签

在覆盖和 crowd 条件通过、最高 IoU 类别归属无冲突时：

| 条件 | 标签 |
| --- | --- |
| g* 类别 c₀，u₀≥0.50，u₁≤0.30 | 0 |
| g* 类别 c₁，u₁≥0.50，u₀≤0.30 | 1 |
| g* 为可确认第三类、IoU≥0.50，且 u₀、u₁≤0.30 | 2 |
| 灰区、双类冲突、crowd、缺标、无可确认物体 | -1 |

同类多个 GT 可以保留类别明确的语义标签，但不能据此获得明确的跨实例身份标签。other 必须有已确认第三类，不把背景或未知当作 other。

receiver 标签不随 donor 更换而改变。

### 7.4 来源与关系标签

供体声明必须保持其真实原生 `c_d_star=c1`。

- **source_valid：** donor 对象身份、覆盖及 crowd 均可核验，IoU≥0.50，归属 GT 类别就是 c₁。
- **source_contradicted：** donor 明确属于另一个类别、IoU≥0.50，且与声明类 c₁ 的最大 IoU≤0.30，覆盖与 crowd 通过。
- 其他为 source_unknown，不制造错误来源标签。

依以下优先级建立关系标签：

| 条件 | relation_target | 含义 |
| --- | --- | --- |
| 双方 owner_known 且明确属于同一 GT 实例，相关核验通过 | invalid_ref | 本任务排除的同实例重复参考 |
| receiver 对象范围可核验，donor source_contradicted | invalid_ref | 原生来源声明被明确证伪 |
| 双方 owner_known、不同 GT、source_valid，receiver semantic=1 | support | 可信同类参考 |
| 双方 owner_known、不同 GT、source_valid，receiver semantic=0 或 2 | contrast | 可信对比参考 |
| 其他 | ignore=-1 | 关系未知 |

这里将明确同 GT 的重复参考列为 invalid_ref，统一解决“不属于跨实例支持”的盲区；**不声称其所有视觉内容错误或永远没有利用价值**。这是本轮任务范围的标签定义。

GT 同对象判定只能进入训练标签或评估分析，不能进入测试候选生成。原生几何与预测关系在测试时仍可能无法识别全部重复对象。

contrast 不被当作错误来源；它可能帮助否定 c₁。support 不被强制提高 c₁，也不被强制产生非零消息。

### 7.5 单动作监督掩码

对实际 D 条边的关系标签：

~~~python
known = rel_target >= 0
valid = (rel_target == SUPPORT) | (rel_target == CONTRAST)
bad = rel_target == INVALID_REF

sem_mask = (
    semantic_target >= 0
    and D > 0
    and known.all()
    and valid.any()
)
rel_mask = known
all_bad = D > 0 and bad.all()
bad_mask = bad.any()

assert not (sem_mask and all_bad)
~~~

| 边组成 | fused CE | 关系 CE | 坏边约束 |
| --- | --- | --- | --- |
| 全部 support/contrast | 开 | 每条已知边 | 无 |
| valid + invalid_ref | 开 | 每条已知边 | 只限制坏边贡献 |
| 全部 invalid_ref | **关** | 每条已知边 | 坏边贡献 + 回到 base |
| 含 unknown | 关 | 仅已知边 | 有明确坏边时只约束该坏边 |
| 无边 | 关 | 无 | 无，语义返回 base |

主前向始终使用预测的 gate；GT 仅决定训练损失和采样。禁止训练时根据 GT 先把坏边消息置零，再在测试时换成预测门控。

对 unknown+bad 的不变性约束可能通过共享参数影响另一条边，但没有给 unknown 伪造语义或关系标签。

### 7.6 一致的损失

阶段一：

$$L_A=\operatorname{BalancedMean}_{y_r\ne-1}
\mathrm{CE}(z^0,y_r).$$

阶段二：

$$L_B=L_{\rm sem}+L_{\rm rel}+L_{\rm bad}.$$

三项系数固定均为 1，不做搜索：

- L_sem：只在 sem_mask 开启的 action 上对 zᶠ 做 receiver 三态 CE。
- L_rel：只对 known 边做关系 CE，按第8.2节的逐边全局权重实现；同一ActionKey、同一关系状态内的边已在权重中均分，不能再额外除以D或先做一次跨边mean。
- L_bad：约束明确无效参考，定义如下。

令 t_j 是已经经过预测 gate 的 128 维消息。去除坏边 j 的**消息贡献**：

$$
M^{(-j)}=M-t_j/D,\quad
z^{(-j)}=z^0+\operatorname{center}(F(M^{(-j)},h_{\rm base})).
$$

保持原 D、其他消息、Qwen 读取、候选、native16 不变；只需多算小残差头，不重新运行 Qwen，不把剩余边重新归一化。

$$
e_j=\frac{\|t_j\|_2^2}{128}
+\frac{\|\operatorname{center}(z^f)
-\operatorname{stopgrad}(\operatorname{center}(z^{(-j)}))\|_2^2}{3}.
$$

第一项约束 gain 也会读取的消息本体；第二项避免小消息被残差头放大后仍造成明显语义变化。`stopgrad(z_without_j)` 是当前比较目标，不是固定真值 teacher。

全部边均为 bad 时另加：

$$e_0=\|\operatorname{center}(z^f-z^0)\|_2^2/3.$$

对存在坏边的 action：

$$
L_{{bad},a}=
\frac{\sum_{j\in B_a}e_j+
\mathbf1[|B_a|=D]e_0}
{|B_a|+\mathbf1[|B_a|=D]}.
$$

全坏 action 不加 fused GT CE，所以即使 self 本身判错，也不会同时要求“凭坏参考改正”和“保持 self”。好坏混合 action 不加整动作 base 回退，允许好消息完成纠错。

这些是**软学习目标**。严格恒等回退只对显式 M=0 成立；不能写成模型对全部真实坏供体已经实现零影响。门控和消息幅度也不分别可辨识，不能仅看 gate 小就断言风险低。

各损失的空有效集合返回有限零；全部损失均无监督的 batch 跳过 optimizer step，避免只有 weight decay 更新。不得附加“support 必须大于某个消息幅度”、最小熵或最小 JS 损失。

---

## 8. 训练样本组织：让关系任务有辨别信息

### 8.1 基础集合和有限的配对补充

阶段一使用原 v2 候选 action 中全部语义可核验样本，每个 ActionKey 一次，self 不读取 donor 特征。

阶段二的基础训练视图是每个 action 的真实完整 donor 集合。另允许**每个原始 ActionKey 最多一个单供体替换视图**：

1. 只从同一 train 图、原始边截断前的候选池寻找；
2. 仍满足原 donor、receiver、外观、空间和可执行性条件，c₁ 不变；
3. donor 保留原生声明、真实图像特征和分数，不用 GT 改贴类别；
4. 优先寻找与原视图已知来源状态不同的参考：原有有效参考时找 invalid_ref，原有 invalid_ref 时找有效参考；原参考均 unknown 时找一个已知参考；
5. 同一优先级按原边序、donor query ID 确定性选择；没有合格替换就不新增；
6. 所有用于选择的 GT 只属于 train；明确标记 `training_pair_extension=True`，不改变测试候选规则。

这样可在同一个 receiver、同一个提名类上看到不同来源，降低只看 receiver 就完成关系任务的机会。

同时统计固定 `(image_id, donor_query_id,c1)` 下是否存在不同的 receiver，分别构成 support 和 contrast。关系损失对三态平衡，确保已有对比样本参与训练；本轮不为凑齐这一统计扩大数据集或伪造配对。

双向条件变化只减少简单单侧捷径。模型仍可能分别识别 receiver/donor 再比较类别；关系准确不自动等于学会细粒度部件对应。

### 8.2 固定的分层权重，避免重复加权

采用预计算权重，不同时使用一套逆频率 sampler 和另一套 class-weight CE。

每一项损失按下列层级归一化：

1. 语义 CE 的有效三态等权；关系 CE 的有效三态等权；坏边损失不区分类别；
2. 每个状态内图像等权；
3. 每图内原始 ActionKey 等权；
4. 同一 ActionKey、同一状态内的有效训练视图／边平均。

对于一个监督原子，权重为：

$$
w=\frac{1}{|S|\,n_{\rm image}(s)\,
n_{\rm action}(i,s)\,n_{\rm atom}(a,s)}.
$$

语义原子是有效 view，关系原子是已知边，坏边原子是第 7.6 节已经在内部归一化的 action-view 损失。坏边项令 |S|=1。每个非空损失项的全部权重和应为 1。

stage B 的训练视图集合取至少存在一项监督的 union，固定 N 个视图；每 epoch 用 seed 确定的随机排列遍历一次。若有效 optimizer batch 有 b 个视图，使用：

$$
\widehat L_B=\frac{N}{b}
\sum_{a\in batch}
\left[w_{\rm sem}(a)\ell_{\rm sem}(a)
+\sum_d w_{\rm rel}(a,d)\ell_{\rm rel}(a,d)
+w_{\rm bad}(a)\ell_{\rm bad}(a)\right].
$$

这是对上述全局加权目标的 minibatch 估计。梯度累积按整个有效 batch 的 b 计算，不能各 micro batch 单独重归一化。monitor 用完整固定权重精确汇总，不依赖 batch 划分。

同样的层级适用于 stage A。阶段 A 每个 ActionKey 只有一个自身视图。缺少某个必须学习的三态时报告数据前置不足，不把未知填进缺失类别；完成其他构建，但不宣称完成该三态模型的有效训练。

### 8.3 明确不加入的监督

- 不要求同类 donor 的每条消息都正向或非零。
- 不把两个框中相同网格位置强制视作同一部件。
- 不把 GT 框当作像素级部件／分割真值。
- 不要求全部接收端预测服从图邻居。
- 不把模型自己的 gain 上升当作 donor 有用的真值。
- 不依据测试成败生成训练对。
- 不对所有样本蒸馏固定 self 概率分布，避免同时惩罚合理纠错。

---

## 9. 三阶段训练与统一资格

### 9.1 数据分工

| 数据 | 本轮用途 |
| --- | --- |
| train+rare_train：历史 6,038 图 | 拟合所有新参数、normalizer 和配对视图 |
| monitor：历史 807 图 | 每阶段监控与最佳检查点选择 |
| policy：历史 871 图 | 模型全部冻结后，只锁一次 gain 前缀 |
| 原 790 图 | 已观察开发数据；不新增一轮调参用途 |
| 已报告的 2,048 图 | 已观察历史测试／开发证据，不能再次称 v2 盲测 |
| 一个新的未观察留出清单 | 若现有数据可用，训练前锁定，最多 2,048 图，一次正式比较 |

实际清单和内容哈希优先于上述历史数字。检查 image ID、文件哈希、重复图像及标注归属；不同清单名称不代表独立数据。

新留出不能因为类别或结果不理想而替换。没有未观察数据时仍可完成构建、训练及一次历史比较，结果明确标为开发验证；不自动下载大规模新数据。

### 9.2 阶段 A：先建立完整自身基线

训练第 5.2 节前缀和自身三态头，DINO、CLIP、Qwen 冻结。载入 monitor 最佳权重后：

- 冻结全部 receiver 投影、位置投影、role、BASE、H₀、统计量与模板；
- 保存 `self_snapshot_hash`；
- 允许预计算 train/monitor 的 h_base、z⁰和纯视觉 φ_r；
- 检查之后的阶段 B optimizer 不包含上述参数。

若复用历史 P_node 初始化，必须逐张量检查语义和尺寸兼容；新前缀结构、局部投影、角色和任务头按本规范初始化。不能复用旧 action head 冒充新的 H₀。

### 9.3 阶段 B：训练关系与消息

固定阶段 A，自主训练：

- donor 后缀投影、role、REL；
- H_rel；
- W_q、W_k、W_v、W_pos；
- k_NULL、b_NULL；
- W_m、W_o、W_b。

W_m 常规随机初始化，W_o 零初始化；不要两者同时为零。b_NULL 初始化为 **ln(9)**：当所有 real score 与未加偏置的 NULL score 均为零时，s=0.5，而不是因 9 个真实位置自然得到 0.9。这是初始化，不是校准承诺。

阶段 B 加载 monitor 最佳权重后冻结上述模块，再生成收益阶段需要的最终语义／消息缓存。

### 9.4 A、B 的统一设置

| 设置 | 固定默认 |
| --- | --- |
| seed | 42 |
| 每阶段最大 epoch | 12 |
| patience | 3，min_delta=0；按完整 epoch monitor loss |
| 初始化监控 | 保存 epoch=0 指标和初始快照 |
| 最佳权重 | monitor 目标最小；精确相同则取较早检查点 |
| 优化器 | AdamW |
| 输入适配器、role、query/K/V、NULL 参数 LR | 1e-4 |
| A 的 H₀；B 的 H_rel、W_m/W_o/W_b LR | 3e-4 |
| weight decay | 0.01；bias、LN、独立 role/read/NULL 参数不衰减 |
| 有效 batch | 64 个 action/view |
| 默认 micro batch | 8；累积至有效 64 |
| 梯度裁剪 | 总 norm 1.0 |
| 调度 | 按最大预算步数，前 5% warmup，其后 cosine；早停可提前结束 |
| 精度 | 硬件支持则 BF16；否则 FP32；关键归一化、损失、gate、排序 FP32 |
| dropout | 新模块不使用；Qwen 固定 eval |

不足 64 的末批按实际 b 归一化，不能丢弃稀少正例。显存不足可减 micro batch 并保持有效 batch；不更换模型或损失。中断可从同一 optimizer、scheduler、RNG 与数据位置恢复，不属于新实验。

### 9.5 必须保留的梯度路径

~~~python
qwen.requires_grad_(False)
qwen.eval()

for p in receiver_prefix.parameters():
    p.requires_grad_(stage == "A")

# 阶段 B 的 donor suffix 是可训练输入。
inputs_embeds = build_sequence(...)
hidden = qwen(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=False,
    output_hidden_states=False,
).last_hidden_state
# 不在这个调用外包 no_grad()/inference_mode()。
loss.backward()
~~~

实际 Qwen wrapper 需按现有 Transformers 接口适配。若使用 CausalLM 包装器，选择返回 decoder last_hidden_state 的正确子模块，不能把词表 logits 当 896 维 hidden。

冻结权重、eval 和关闭 autograd 是三个不同概念。[R4] 外层 model.train() 不得把 Qwen 变回 train，也不能将训练后缀的 hidden-state 缓存当作新前向。

W_o 零初始化时，**第一步**语义梯度通常先更新 W_o，不能要求此刻全部上游语义梯度非零。正确检查是：

1. W_o 首步获得有效梯度；
2. 非零更新后，上游 query/K/V 和 donor adapter 存在有效语义梯度；
3. 关系监督本身可以从第一步训练 H_rel 和后缀；
4. 冻结 self 与 Qwen 参数前后哈希不变。

对于某个恰好零消息样本，梯度可为零；用有差异的合成输入或已核验小批量检查整条路径，不能据单样本下结论。

### 9.6 v2 训练资格

新资格版本固定为 `training_qualification_v2`，区分：

**工程有效完成：**

- 数值、梯度、冻结与缓存来源检查通过；
- 无 train/monitor/policy/test 角色混用；
- 按预先设置的 patience 或预算正常终止；
- 先恢复最佳权重，再冻结、再生成下游缓存；
- 固定输入的加载前后输出在规定精度容差内一致；
- 保存停止原因、初始／最佳／末轮指标。

**学习是否得到支持：**

- 另列 monitor 各损失变化、语义三态结果、关系三态结果和消息作用；
- 初始检查点最优或没有改善时标记“未观察到有效学习”；
- 流程有效完成不等于数学收敛，也不等于检测性能有效。

取消旧的“必须跑完整 cosine，同时允许 early stop，另要求末五轮平台”组合。第 5 epoch 最佳可以合法出现。读取旧日志解释 v1 的原资格失败项，但不要据 v2 新规则追认旧结果为已经验证成功，也不要为此重跑 v1。

---

## 10. 冻结风险输入与新的收益头

### 10.1 原 native16 只能完整兼容复用

原 16 项证据：

- 对 c₀、c₁各四项：receiver 原生类别分、其他 query 最大 IoU×类分支持、其他原输出同类最大 IoU×原分、其他原输出同类最大 IoU；
- 上述 8 项先截到 [0.001,0.999] 再取 logit；
- 整裁片与原注意力聚焦裁片，对两类分别计算 CLIP 相似度及距 365 类最佳相似度的差；跨两视图取均值与绝对差，得到另外 8 项。

**索引顺序以旧特征生产函数和风险模型 manifest 为准，不在 v2 中自行重排。** 旧模型的 16 项顺序、裁片生成、词表、预处理、normalizer 和风险权重必须共同匹配。

冻结复用原 `16→16→1` 风险模型，使用其 **sigmoid 前 logit**，风险 normalizer 仍使用它自己的旧训练统计。

v2 不再把 `risk≤0.220496` 当执行硬门槛；该值只属于历史版本。若特征或旧风险依赖不兼容，完成其他代码并报告缺项，不用新的16项冒用旧权重，也不额外训练一个风险模型。

### 10.2 gain147 输入与网络

$$
x_{\rm gain}=[
\underbrace{\mathrm{native16}}_{16};
\underbrace{\ell_{\rm risk}}_1;
\underbrace{z^f_1-z^f_0}_1;
\underbrace{\mathrm{softmax}(z^f)_2}_1;
\underbrace{M}_{128}]
\in\mathbb R^{147}.
$$

gain：

$$
\ell_{\rm gain}=
\mathrm{Linear}(32,1)\circ\mathrm{GELU}
\circ\mathrm{Linear}(147,32)(\widehat x_{\rm gain}).
$$

无 dropout，常规初始化；最后一层 bias=0。native16 使用旧预处理后、**旧风险标准化之前**的值；新 gain 自己拟合 147 维 normalizer，std≥0.01，标准化后截到 [-6,6]。

禁止输入：

- raw h_REL / donor hidden；
- 独立的 relation 概率、gate、NULL 分；
- JS；
- GT、是否缺标、任何标注派生特征；
- v1 是否执行、历史测试正确与否。

M 是实际进入语义修正的同一份门控消息，不另建“供体摘要”绕路。

### 10.3 收益真值

v2 必须将收益核验与第7.3节的GT语义归属对齐，避免旧标签中来自不同物体的最大IoU或灰区重新进入收益正例。新增v2字段，不修改历史v1记录；也不能从“旧action MLP的正负target”倒推语义确认。

原六档集合回放保持不变。先得到独立可确证的known_harm，再按下列优先级产生v2效用状态。这里所有semantic_target均由GT核验得到，绝不是full/off的预测argmax：

~~~python
gain_target = -1
confirmed_correct = (semantic_target == 1)
utility_reason = None
if known_harm:  # 具有可确证依据的伤害；不能仅由模型风险分判断
    gain_target = 0
elif not legacy_utility_known:
    utility_reason = "legacy_utility_unknown"
elif semantic_target < 0:
    utility_reason = "v2_semantic_unverified"
elif legacy_confirmed_correct != confirmed_correct:
    utility_reason = "legacy_v2_claim_conflict"
else:
    gain_target = int(
        confirmed_correct
        and all(x == 0 for x in lost_matches)
        and (new_matches[0] > 0 or new_matches[3] > 0)
    )
utility_known = (gain_target >= 0)
~~~

明确伤害足以确认gain负例，即使其他信息仍未知；它在执行状态统计中也具有最高优先级。非有害但旧／新核验冲突或缺少新语义确认的动作保留unknown，并记录原因，不能偷偷沿用旧正例。该规则同时用于v2 train/monitor/policy/evaluate；B臂的历史策略、训练标签与已保存结果不被覆盖。新的四臂比较则由同一评估器按同一v2核验规则重新标注固定预测，详见第14.1节。

如果两个核验器本应使用相同IoU、类别映射和标注子集，却出现不可能的逻辑矛盾，先定位缓存、映射或实现错误；只有定义收紧造成的真实灰区才转为unknown。不能用unknown掩盖确定的程序错误。

六档顺序固定：

~~~text
0: IoU .50, score >= .05
1: IoU .50, score >= .15
2: IoU .50, score >= .35
3: IoU .75, score >= .05
4: IoU .75, score >= .15
5: IoU .75, score >= .35
~~~

utility_known=False 的样本不参与收益拟合；不能把未确证当负例。使用所有已核验可执行原候选 action，不只训练语义已经通过的部分，不使用阶段 B 的替换训练视图扩张收益动作池。

### 10.4 full/off 成对拟合，避免关闭消息后输入失配

冻结 A、B 后，每个收益 action 构造两个输入：

- full：实际 M、zᶠ；
- off：M=0、δ=0、zᶠ=z⁰，再重算 margin 和 P_other。

两者的固定框、分数、c₀→c₁动作完全相同，故使用**相同的检测收益标签**。每个 action 的 full/off 各占一半权重。不能只置零 M 却保留 full 的 margin/P_other。

normalizer 仅从 train 已核验收益动作的这两种原始输入拟合，每个 action 两视图等权；不得按 full 是否正确或是否执行来选统计样本。

正负类各半，类内图像等权、再 action 等权、最后 full/off 平均。monitor 同一定义。这个处理不新增 Qwen 前向，也不是训练第二个对照模型。

阶段 C 固定设置：

| 项目 | 值 |
| --- | --- |
| 优化器 | AdamW |
| 学习率 | **0.001**；不照搬旧线性头的 0.02 |
| weight decay | 0.05；bias 不衰减 |
| optimizer steps | 最多 120，固定 cosine，无追加续训 |
| batch | 全量加权目标；可分块精确累积 |
| 梯度裁剪 | norm 1.0 |
| 检查点 | 每 step monitor BCE，取最佳；相同取较早 |
| 初始化监控 | 保存 step 0；若没有学习改善则如实标记 |

A/B/C 使用相同 train 分阶段拟合，不额外交叉训练。本轮收益读出属于训练内堆叠拟合，不称为独立校准；其泛化由隔离的 monitor、policy 和最终留出分别承担。

---

## 11. 无 GT 推理、策略锁定与消息关闭

### 11.1 推理唯一顺序

1. 读取固定 D₀、原生图与候选，执行非 GT 可执行性检查和 donor 去重。
2. 计算冻结 self、逐边关系与真实／NULL 局部读取，得到 M、δ、zᶠ。
3. 至少有一个原始可用 donor，必要输入输出全部有限。
4. 严格要求 `z_final[1] > max(z_final[0], z_final[2])`；相等拒绝。
5. 计算冻结风险 logit及 gain147，得到 gain logit。
6. 每图合格动作按 gain logit 取最大；并列按 ActionKey 字典序。
7. 最大值达到已锁定 gain logit cutoff 才执行，否则返回 D₀。
8. 仅替换原 slot 的 class ID，其他输出字段保持原值。

**不再增加：**旧风险硬 cap、JS 上限、极高语义概率门槛、原生 margin>0、预测 support 必须胜出、gate 必须非零等额外资格条件。关系与 NULL 已在消息内部使用；不能再暗加一层 gate。

零消息时，冻结 self 仍可能支持 c₁，因此可以形成改类动作。这是设计允许的自身判断，不能将这类动作归因于供体消息。

gain logit 是排序和 cutoff 的唯一标量；sigmoid 只供展示。这样避免 sigmoid 饱和导致数值同分，不改变数学上的单调排序关系。模型分数不能替换原检测分数，也不称为校准成功率。[R5]

~~~python
def apply_policy(D0, records, policy):
    if policy.gain_logit_cutoff is None:
        return D0

    eligible = [
        a for a in records
        if a.is_executable
        and a.has_raw_donor
        and a.all_finite
        and a.z_final[1] > max(a.z_final[0], a.z_final[2])
    ]
    if not eligible:
        return D0

    chosen = min(eligible, key=lambda a: (-a.gain_logit, a.key))
    if chosen.gain_logit < policy.gain_logit_cutoff:
        return D0

    result = copy_detection_records(D0)
    result[chosen.slot_id].class_id = chosen.proposed_class
    assert_only_one_class_changed(D0, result)
    return result
~~~

### 11.2 一次 policy 锁定

在既定 871 图上、A/B/C 全冻结后：

1. 先无 GT 按真实资格与排序生成每图唯一首选；
2. 全部首选按 gain logit 降序排列，同分整组并入；
3. 枚举这一条排序的前缀，包含空前缀；
4. 合法前缀：已知有害为 0；六档 lost 总数均为 0；未确证≤11；
5. 按字典序最大化：
   - new[0]+new[3]；
   - confirmed_correct 数；
   - 负的 unknown 数；
   - 负的执行数；
6. 保存最后一组的 logit 为 cutoff；若选择空前缀，保存 None；
7. 保存完整权重、normalizer、schema、数据、候选配置、代码版本和 cutoff 哈希。

new[0]+new[3] 是两个 IoU 档的计分和，可能重复计入同一物体，不称独立目标数。不搜索风险 cap、JS 系数、语义门槛或不同预算。

≤11 是 policy 集的经验预算，不保证测试未确证≤11或错误为零。空前缀合法，但表示没有形成可用覆盖，不算方法有效。

### 11.3 message-off 的精确定义

复用 full 的：

- D₀、图、raw candidates、non-GT donor 合法性；
- 冻结 self；
- native16、旧风险模型；
- gain 权重及同一147维 normalizer；
- full 锁定的 cutoff。

仅干预实际消息通路：

~~~python
M = zeros(128)
delta = zeros(3)
z_final = z_base
gain_input = rebuild_147(native16, risk_logit, z_final, M)
~~~

然后重新计算语义资格、gain logit、图内首选、cutoff 与最终动作。不能使用 full 的 gain 缓存，不能根据 full 的 gate 或 relation 先删候选，也不能为 off 重新锁阈值。

这是一种固定已训练系统中的消息干预，不是单独重新训练和优化的“无 donor”模型。候选提名与原生邻居／覆盖特征仍包含跨实例信息；也不能据此证明 LLM 相比其他模型不可替代。

---

## 12. 工程模块、缓存与数据隔离

### 12.1 推荐模块职责

建议独立版本目录，例如 `work/graphllm_receiver_evidence_v2/`，不覆盖 v1。

| 模块 | 职责 |
| --- | --- |
| schema.py | ActionKey、特征与预测记录、类别映射、版本契约 |
| native_bridge.py | 复用原 D₀、图、CLIP 与 native16 |
| local_cache.py | 预测框 local9、层与坐标审计、缓存 metadata |
| targets.py | receiver 语义、关系、收益标签与掩码 |
| training_views.py | train-only 有限配对、固定分层权重 |
| model.py | receiver 前缀、逐边真实/NULL读取、关系头、K/V与消息、残差 |
| losses.py | 阶段 A/B 目标及坏边删除贡献比较 |
| train_self.py | A 的冻结边界、早停与最佳快照 |
| train_message.py | B 的优化、固定 teacher 与缓存失效 |
| train_gain.py | full/off 成对147维收益拟合 |
| policy.py | 单一 gain 排序前缀锁定 |
| predict.py | 无 GT 推理及 full/off 共用执行器 |
| evaluate.py | 独立标注读取、动作对账、六档匹配及 COCO AP |
| run_once.py | 按本文阶段执行和产物校验，不自动追加实验 |

历史源码线索：

| 职责 | 原文提供的文件 |
| --- | --- |
| 原生前向与文本容量 | donor_transport_probe.py、objects365_cache.py |
| 图和候选 | local_runtime.py、hierarchical_graph/cache.py、semantic_action_pool.py |
| 适配器／Qwen | hierarchical_graph/model.py、layer_entropy_cache.py |
| 旧语义训练／读取 | omni_layer_train.py、omni_layer_cache.py |
| 旧分歧 | objects365_donor_view_entropy.py |
| 接收证据／核验／风险 | objects365_receiver_native_evidence.py、objects365_claim_cache.py、objects365_receiver_keep_gate.py |
| 收益／评估 | objects365_receiver_gain_gate.py、objects365_receiver_keep_holdout_run.py、objects365_receiver_gain_confirmation_run.py、objects365_fit.py |

旧 Windows 路径为 `C:/Users/XC/Documents/Codex/2026-11-10/general-ovd-graphllm`。这是定位线索，不是本文已经检查到的运行目录。

### 12.2 最小接口契约

~~~python
build_actions(native_cache, graph_cache, candidate_config) -> list[Action]
extract_local9(image_memory, masks, predicted_boxes) -> LocalCache
build_targets(actions, annotations, coverage_verifier) -> TargetTable
build_train_views(actions, targets, raw_edge_pool) -> TrainViews
make_loss_weights(views, targets) -> LossWeights

read_base(receiver_inputs) -> BaseRecord
read_relation(receiver_prefix, donor_inputs) -> hidden896
build_applied_messages(base_record, real_hidden, null_hidden) -> EdgeMessages
correct_semantics(base_record, message128) -> logits3
compute_message_losses(record, separate_targets, weights) -> LossRecord

build_gain147(native16, frozen_risk, logits3, message128) -> vector147
lock_policy(predictions, separate_policy_targets) -> LockedPolicy
predict(dataset_without_annotations, frozen_bundle, mode) -> PredictionFile
evaluate(fixed_prediction_files, annotations) -> EvaluationReport
~~~

`Action/PredictionInput` 不包含 targets/GT 路径。训练损失显式接收单独 TargetTable。推理模块禁止导入 label builder 作为运行依赖；评估在预测文件固定后单独读标注。

### 12.3 分层缓存失效

| 缓存 | 必须锁定的依赖 |
| --- | --- |
| D₀/原图 | 图像、DINO权重、词表映射、预处理、阈值、NMS、槽排序 |
| CLIP/native16 | CLIP权重、两种裁片及原注意力提取、类别文本、维度顺序 |
| local9 | 图像、DINO权重、encoder张量来源、stride层、预测框、padding/valid_ratios、采样规范 |
| 标签 | 标注版本、覆盖/crowd核验器、候选池、规范类别映射、GT owner规则、六档回放 |
| self cache | A最佳权重、全部前缀、模板、native5统计、local9版本 |
| 关系/消息 cache | A最佳快照、B最佳权重、真实/NULL模板、维度、gate与残差公式 |
| gain features | 原native16和风险版本、A/B最终快照、full/off定义、gain147字段顺序 |
| 最终预测 | A/B/C最终权重、全部normalizer、候选规则、cutoff、full/off模式 |

B 训练期间只可复用完全冻结的 receiver 部分缓存；真实／NULL后缀 hidden、query、K/V、消息不得跨 optimizer 更新复用。内存中的同一 action、同一步共享 NULL 前向可以保留计算图并正常汇总梯度。

新版本不使用旧 v1 Qwen hidden-state cache。缓存 key 不匹配就重建对应层，不以文件存在为充分依据。

D₀ 是四臂共同的唯一原生输出源。补建 local9 时不得悄悄改用另一个 batch/padding/预处理下产生的新 D₀。若需要重建 native bundle，先说明并固定统一 D₀，再对所有臂使用这一版本；不能让 v1 和 v2 各自用不同原生检测。

### 12.4 预测记录

每 action 保存：

~~~text
ActionKey, donor_query_ids, donor_predicted_classes
feature_schema, local_schema, self_hash, message_hash, gain_hash, policy_hash
z_base[3], z_final[3], delta[3]
relation_probs[D,3], nonnull_mass[D], gate[D]
per_edge_message_norm[D], applied_message128
risk_logit, gain_logit, display_gain_sigmoid
js_or_null, js_available
is_executable, semantic_eligible, image_top1, executed, reject_reason
mode: full / message_off
~~~

GT、semantic_target、relation_target、gain_target、new/lost、harm 写入独立标签／评估表。预测输出可保存完整 applied_message128 便于核对，不需要大量 attention 热图或逐 token 日志。

reject_reason 固定少量枚举：

~~~text
invalid_input
invalid_local_feature
duplicate_slot
no_raw_donor
semantic_not_c1
not_image_top1
below_gain_cutoff
policy_empty
executed
~~~

非有限必要输入在训练时报告错误；推理时拒绝对应 action 并记录。不能将 NaN 静默替成高分，也不能自动丢掉坏数值 donor 后重建一个新候选集。

---

## 13. 完整搭建后的必要检查

这些是实现正确性检查，不是额外效果实验。统一写成少量测试组，发现具体问题才追加对应测试。

### 13.1 输入与监督检查

- canonical 类别映射、ActionKey、原槽、query、多类别输出、同 query/c₁重复处理正确。
- local9 确实来自预测框及指定 encoder 层；合成坐标、padding、mask 归一化检查通过。
- 标签真值表覆盖支持、对比、来源错误、同 GT、unknown、crowd、IoU 灰区、确认第三类。
- `sem_mask & all_bad` 永远为 False；contrast 不进入 bad；unknown 不被改标。
- 主前向不接受 GT；更换仅存在于标签表的字段不改变预测。
- 每项全局权重和为 1；不同 micro batch 划分的完整加权损失一致。
- 汇报三态有效样本数、图像数、ignore 数、同 receiver 换来源组数、同 donor 换 receiver 组数。

### 13.2 数学与数据通路检查

- F(0,h₀)=0；全消息显式关闭时 zᶠ=z⁰。
- Wo 零初始化、Wm 非零初始化；消息可产生正负 residual，并无旧 B=2 截断残留。
- real 与 NULL 完全相同输入时 h/query/message 一致；不改变角色、mask 或位置。
- 九个视觉 value 相同而位置不同，m 仍为零；不能让位置在 V 中制造证据。
- 非 NULL 质量只乘一次；无差别 score 时 b_NULL=ln9 对应 s=0.5。
- 删除一条 bad 消息的比较保持原 D；混合好坏没有整动作 base 回退。
- 空损失掩码有限为零。
- gain 输入恰好147维，模型图没有 raw h/rel/g 的旁路。
- off 从**原始 M=0**重新拼接并标准化；不能把已经标准化的128维切片直接写零来代替。
- 记录公式检查通过，不把它当作数据有效性证据。

### 13.3 梯度与冻结检查

使用一个有有效监督的小批量，进行不计入正式运行的 smoke，随后恢复初始状态：

- stage A 的前缀和 H₀获得梯度，DINO/CLIP/Qwen 不更新；
- stage B 恢复 A 最佳快照并冻结，真实／NULL后缀均保留梯度；
- 检查 Wo 首步及之后上游梯度，不错误要求首步所有参数非零；
- B optimizer step 前后 self 参数哈希与固定输入输出不变；
- Qwen 因果前缀不受 donor 内容影响；FP32 默认 allclose atol/rtol=1e-5，BF16 可用1e-2，实际容差随运行精度记录；
- 禁用梯度的旧 hidden cache 不能混入训练；
- stage C 不更新 A/B 或旧风险模型；
- 按最终最佳检查点重载后的缓存与直接前向一致。

### 13.4 执行与评估检查

用小型合成检测集合验证：

- 0 个合格 action 时完整返回 D₀；
- 一图多 action 仍至多改一个槽；
- 仅 class ID 改变；框、原分、槽号、长度和其他项严格不变；
- 相同 query/c₁重复禁止；不重跑 NMS或排序 D₀；
- ≥0.35 的输出及相应六档第2、第5档匹配应完全不变；
- prefix 同分整组处理，空策略表示 None；
- full/off 使用同一策略但各自重算首选；
- 五类互斥执行状态之和等于执行总数；
- COCO评估参数、类别映射、预测数量和原分数一致。

完成这些组并具备全部可运行入口后，写出 `BUILD_COMPLETE.json`。它仅说明工程链路可运行，不说明训练有效或 AP 提升。

---

## 14. 一次小规模验证协议

### 14.1 最多四臂

| 臂 | 固定定义 |
| --- | --- |
| A | 同一个原生 D₀ |
| B | 已保存的 v1 JS-on：对应历史 71 次动作的版本，原权重、候选、normalizer及锁定策略不变 |
| C | v2 full |
| D | v2 message-off：同一训练模型、normalizer及 C 的 cutoff |

B 不是历史 50 次动作的更老门控。不得重训或按 v2 规则“修复”B，否则失去冻结基线身份。保留其原先未通过旧训练资格的说明。

冻结B的**预测策略**不等于沿用另一套比较口径。新的四臂表使用相同D₀、GT、v2语义/效用核验和同一AP评估器；历史原表另存，不覆盖。若新旧标签导致“确认改对/未确证”计数不同，附一行迁移对账。A/B/C的比较反映各自固定运行点，不能将所有差异单独归因于关系模块；C/D才是同一v2模型中的消息干预。

若没有完整 v1 冻结产物，明确标记 B 不可复现，不冒用另一个版本；可完成 A/C/D 的比较，不能声称完成全部四臂。

不再训练“无LLM”“无local”“改NULL系数”等分支。JS 只记录，不增加 JS on/off 臂。

### 14.2 执行顺序与规模

1. 训练前锁定一份现有未观察图像清单，数量不超过2,048；无新清单则使用已观察集并明确开发性质。
2. 完成 A/B/C/D 的无 GT 推理，固定预测文件和哈希。
3. 独立评估器一次读取该清单标注并计算全部结果。
4. 不看结果重新调 cutoff、训练、换 checkpoint、换图像或增加模块。
5. 仅真正实现错误可修复后重跑受影响步骤；如已看到测试结果，记录为“测试揭示后的修订”，不能重新包装成盲测。

一次运行包含 A、B、C 三个训练阶段，不意味着只允许一个 optimizer。四个评估臂也不意味着四次训练。

### 14.3 必须交付的三张表

**表一：总体结果**

每臂列出：执行数、互斥动作状态、确认改对、原类误改、六档 new/lost、AP/AP50/AP75，以及相对 D₀ 的百分点增量。

**表二：full 与 off 的配对动作**

至少列出：

- 最终动作完全相同的图数；
- full 执行、off 不执行；
- off 执行、full 不执行；
- 二者都执行但选择不同 action；
- 上述不同动作对应的无损新增、损害与未确证变化。

只有 C 相对 D 出现可确认的新收益，才能说这轮观察到消息的局部作用。分数变动而动作完全相同不能当作执行收益。

**表三：固定候选池的机会与漏斗**

同一个原始非 GT 候选池中统计：存在无损新增机会的图数 → v2 非 GT 可执行性后仍有机会 → 语义资格后 → 图内首选 → cutoff 后。

各臂报告机会回收率时使用同一原始分母，不用某臂过滤后的较小集合美化覆盖。原135图的分母只在原定义、原数据和原候选池复现后引用。

这是候选范围内的机会计数，不是整个检测任务的 AP 上界，也不额外做最优 AP oracle 搜索。

### 14.4 执行动作必须完全对账

为每个执行 action 按以下优先级赋予一个互斥状态：

1. **known_harm：**已核验原类被改错，或已核验存在任一档原匹配损失；
2. **unknown：**不属于已知有害，但效用无法完整核验；
3. **safe_new：**已核验确认改对、六档无损，且 new[0]>0 或 new[3]>0；
4. **correct_no_new：**已核验确认改对、六档无损，但上述两档均无新增；
5. **verified_other：**其余已核验动作；不把它自动描述为语义正确或有收益。

必须：

$$N_{\rm execute}=N_{\rm harm}+N_{\rm unknown}
+N_{\rm safe\_new}+N_{\rm correct\_no\_new}+N_{\rm verified\_other}.$$

原始 `confirmed_correct`、`original_class_harmed`、六档 new/lost 和语义标签另列，允许诊断字段交叠；不可用交叠列充当互斥账目。

先用已保存日志核对历史旧门控的2条差额和v1的1条差额，不必为对账重跑模型。日志不可得时注明未完成历史对账，同时保证新版完整闭合。

### 14.5 AP 与匹配解释

六档匹配继续使用既有确定性一对一回放，保存匹配 GT ID 集合差。COCO AP 使用原 pycocotools 评估流程，保存 evaluator 版本、IoU 阈值、maxDets、area ranges、类别映射及 ignored/crowd 处理。[R6]

两者用途不同，不能将自定义六档匹配数直接称为 COCO AP。AP75 改善可以来自类别变对；本轮没有直接改变定位。

本轮额外记录平均 action 数、Qwen 读取次数和端到端时间／显存概况即可，不跑独立性能基准矩阵。新 local9 和更长序列带来的成本应如实报告。

---

## 15. 验收、失败定位与停止

### 15.1 四层结论分开写

| 层次 | 可以确认的条件 | 不能推出的结论 |
| --- | --- | --- |
| 工程完整 | 入口接通、核心不变量和数据隔离通过 | 方法有效 |
| 训练有效完成 | 合法停止、最佳权重与缓存一致、梯度冻结正确 | 已证明数学收敛 |
| 消息有局部作用 | full 相对 off 出现可确认收益，损害也一并报告 | LLM必要性、真实因果对应、全局最优 |
| 整体检测有改善证据 | 对 D₀、v1、off 的AP与配对匹配收益共同支持 | 对所有数据稳定提升或最佳效果 |

极小正 AP、零动作、gate 非零、attention 看起来合理，都不能单独作为“成功”标准。本轮不预设任意 AP≥0.1 的通过线，也不开展大规模显著性和超参数实验。

### 15.2 一次结果的解释顺序

| 观察 | 首先对应的结论 |
| --- | --- |
| 训练标签／梯度／缓存／候选错误 | 实现未具备有效评估条件 |
| 机会池本来就很小 | 当前固定框、固定分、一次改类动作范围限制覆盖 |
| support/contrast 关系监督没有学习改善 | 当前特征、标签或训练预算未支持关系辨别 |
| 关系有改善而实际消息接近零 | 可能无需供体、局部值缺乏差异，或消息优化不足；不能仅凭这一现象确定原因 |
| full/off 分数不同但动作相同 | 后续排序／cutoff 未将消息转为实际收益 |
| full/off 动作不同而检测收益无提升 | 消息变化尚未转化成有效检测决策 |
| full 对 off 有收益，整体 AP 仍近似不变 | 局部纠错存在，覆盖／排序／动作范围仍限制总体指标 |
| AP 上升同时有已知损害 | 存在取舍，不能报告“无负迁移” |

这些是本轮读结果的框架，不是触发自动增加实验的清单。

### 15.3 停止规则

- 数据泄漏、标签真值表冲突、GT进入推理、冻结失败、维度／缓存错配、输出不变量破坏：停止正式效果评估，修具体实现错误。
- 必要权重或覆盖核验缺失：交付完整可构建部分和明确依赖，不擅自换目标或输入。
- 训练合法但无学习改善、零覆盖或无净收益：完成本轮报告，不自动续训、换种子或改预算。
- 一次正式比较完成即停止，无论正负。后续算法修改属于新的明确版本。
- Codex 必须区分“已完成”“检查失败”“缺依赖”“未验证”，不得把构建完成写成已证明有效。

---

## 16. 统一配置参考

下面是唯一默认配置的结构示例。实际路径由仓库盘点解析；路径解析属于实现适配，不能借此改变算法。

~~~yaml
version: graphllm_receiver_evidence_v2
seed: 42
build_before_training: true

backbones:
  grounding_dino_frozen: true
  clip_frozen: true
  qwen_frozen: true
  qwen_hidden_dim: 896
  qwen_eval_during_training: true
  dino_passes_per_image_at_inference: 1

candidate:
  native_score_min: 0.05
  native_class_nms_iou: 0.50
  native_max_detections: 100
  graph_class_agnostic_nms_iou: 0.82
  graph_max_nodes: 128
  concrete_edges_per_receiver_max: 2
  donor_score_min: 0.42
  donor_clip_min: 0.17
  donor_clip_gap_max: 0.045
  receiver_donor_iou_max_exclusive: 0.45
  receiver_donor_clip_cos_min_exclusive: 0.45
  receiver_c1_clip_min: 0.15
  receiver_c1_clip_gap_max: 0.075
  receiver_original_score_max_exclusive: 0.35
  donor_mutual_iou_max_exclusive: 0.50
  reject_duplicate_query_class: true

features:
  node_dim: 768
  native_receiver_dim: 5
  edge_dim: 12
  local_dim: 256
  local_grid: [3, 3]
  local_source: image_encoder_memory_before_decoder
  local_nominal_stride: 16
  local_boxes: predicted
  local_grid_align_corners: false
  local_padding_mode: zeros
  local_mask_renormalization: true
  min_valid_sample_weight: 0.000001
  native_standardize_std_floor: 0.01
  native_standardize_clip: 6.0

model:
  receiver_prefix_tokens: 14
  donor_suffix_tokens: 12
  total_pair_tokens: 26
  pairwise_qwen_reading: true
  matched_null_reference: true
  null_raw_donor_inputs: zero
  raw_relation_hidden_to_semantic_residual: false
  raw_relation_hidden_to_gain_head: false
  message_dim: 128
  value_has_position_or_role: false
  residual_output_dim: 3
  residual_wm_bias: false
  residual_wo_bias: false
  residual_wo_init: zero
  residual_wm_init: default_random
  null_bias_init: log_9
  invalid_probability_detached_in_message: true
  donor_aggregation: fixed_count_mean
  residual_hard_clip: null
  fixed_donor_quality_multiplier: false
  js_affects_decision: false
  js_diagnostic_temperature: 1.0
  dropout: 0.0

labels:
  semantic_classes: [c0, c1, other]
  relation_classes: [support, contrast, invalid_ref]
  ignore_index: -1
  positive_iou_min: 0.50
  competing_iou_max: 0.30
  related_crowd_coverage_max_exclusive: 0.50
  unique_owner_required_for_relation: true
  same_gt_relation: invalid_ref
  all_bad_fused_ce: false
  any_unknown_fused_ce: false

training:
  qualification_version: training_qualification_v2
  max_epochs_self: 12
  max_epochs_message: 12
  patience: 3
  min_delta: 0.0
  optimizer: AdamW
  adapter_query_lr: 0.0001
  task_head_lr: 0.0003
  weight_decay: 0.01
  effective_batch_size: 64
  micro_batch_size: 8
  grad_clip_norm: 1.0
  warmup_fraction: 0.05
  scheduler: cosine
  loss_semantic_weight: 1.0
  loss_relation_weight: 1.0
  loss_bad_weight: 1.0
  bad_message_norm_term_weight: 1.0
  bad_output_invariance_term_weight: 1.0
  alternative_donor_views_per_original_action_max: 1
  train_sampling: uniform_shuffle_with_precomputed_hierarchical_loss_weights

gain:
  raw_feature_dim: 147
  hidden_dim: 32
  frozen_legacy_risk_logit_as_input: true
  legacy_risk_hard_cap: null
  message_input: applied_message128
  train_full_off_paired_views: true
  full_off_weights: [0.5, 0.5]
  lr: 0.001
  weight_decay: 0.05
  full_batch_steps: 120
  scheduler: cosine
  score_for_ranking_and_cutoff: raw_logit
  standardize_std_floor: 0.01
  standardize_clip: 6.0

policy:
  lock_count: 1
  selection: tied_score_prefix_of_per_image_top1
  known_harm_budget: 0
  lost_match_budget_each_of_six: 0
  unknown_budget: 11
  allow_empty_prefix: true
  semantic_eligibility: c1_strictly_greater_than_c0_and_other
  max_actions_per_image: 1

evaluation:
  arms: [native_D0, frozen_v1_js_on, v2_full, v2_message_off]
  max_images_in_one_holdout: 2048
  old_2048_status: observed_historical_data
  predictions_before_annotations: true
  relock_for_message_off: false
  extra_ablation_runs: 0
  extra_seeds: 0
  change_boxes: false
  change_detection_scores: false
  rerun_nms_after_action: false
~~~

代码中关键尺寸与开关必须来自同一个配置对象。论文表述、配置、forward、loss和eval如果不一致，优先修一致性，不先训练。

---

## 17. Codex 的最终交付

至少交付以下产物：

1. **代码与配置：**实际修改文件清单、唯一 v2 配置、清晰的 A/B/C/policy/predict/evaluate 命令。
2. **盘点和构建报告：**实际源码定位、旧风险兼容性、local9 来源及坐标、GT隔离、核心检查结果。
3. **三个最佳快照：**self、message、gain 及各自训练历史、停止原因、冻结与依赖哈希。
4. **锁定策略：**cutoff 或 None、policy前缀统计、所有依赖身份。
5. **固定预测与评估：**实际可复现的最多四臂预测、三张主表、互斥动作对账、开发/独立验证性质。
6. **结果说明：**哪些问题实现上已修复、是否观察到学习、消息是否改变动作、是否带来检测收益、明确停止。

建议最后一份短结果报告按以下顺序写：

~~~text
完成状态
本次实际实现与本文的差异（没有差异也明确写）
构建与训练资格
数据独立性
总体四臂结果
full/off 配对动作
候选机会与主要阻塞
成本
结论及未验证边界
~~~

不得写“已确保有效”“不存在任何问题”“达到最好效果”。只有实际检查通过的工程性质和实际观察到的指标才能写成已完成事实。

---

## 18. 本文审查记录与已完成的公式检查

### 18.1 已处理的关键审查意见

| 审查问题 | 已落实位置 |
| --- | --- |
| 同类关系标签不能当真实收益 | 第2、7、10节 |
| 可靠异类参考被误标为错误来源 | 第7节 contrast |
| 同GT重复对象缺少明确处理 | 第7.4节 invalid_ref 定义及仅训练使用GT |
| self 的接收局部信息不对等、冻结不完整 | 第5.2、9节 |
| BASE/REL角色差异形成虚假供体增量 | 第5.4节同模板NULL |
| donor hidden绕过局部证据进入最终决策 | 第5、10、11节 |
| NULL质量重复衰减及9位置初始偏置 | 第5.7、9.3节 |
| 单独坏边与好边共享Qwen上下文 | 第5.3节逐边读取 |
| 全坏参考的CE与base回退冲突 | 第7.5–7.6节 |
| 坏消息经gain继续影响动作 | 第7.6节消息本体约束、第10节输入契约 |
| 只比较语义off导致gain仍读full | 第10.4、11.3节 |
| 旧风险工作点被直接当新风险保证 | 第10–11节 |
| 正常早停被旧收敛门槛判失败 | 第9.6节 |
| 已观察2048图再被称为盲测 | 第9、14节 |
| 动作总数不闭合、机会分母变化 | 第14节 |

### 18.2 本文阶段已经做了什么

本次编写时，对核心公式做了 **20 项 NumPy 合成数值检查**，覆盖：

- 零消息精确回退与 Wo 零初始化；
- 相同查询／相同视觉 value 的差分为零；
- NULL初始化、非NULL质量只乘一次；
- 有符号残差可以越过旧固定margin限制；
- 坏边删除保持原分母；
- 语义与全坏掩码不冲突、contrast不被当坏边；
- 好坏混合中的零坏消息不产生回退惩罚；
- 空掩码有限；
- gain147维数、off的原始输入重算；
- 公共logit平移不变性；
- valid_ratio与align_corners=False坐标公式的一致性。

检查全部通过。这些检查使用合成数组，**没有运行实际 Qwen、没有验证真实 DINO tensor位置、没有执行 PyTorch自动微分、没有完成用户仓库集成或训练**。因此第13节的实际实现检查仍是Codex必须完成的工作。

### 18.3 仍需由真实运行回答的问题

- 当前特征是否保存足够细的 receiver 视觉差异；
- 已核验关系配对是否足够、多样且与实际候选接近；
- 同模板NULL差分是否学习了有意义的参考作用；
- 保留自身预测后，消息是否能在不增加明显损害时产生有效动作；
- 固定一次改类的动作范围是否足以改善整体AP；
- 新结构成本是否与有限收益相称。

这些不确定性不能由更长的文档消除。本文的完整性体现在：它提供一个一致、可实现、可证伪的闭环，并防止把监督缺陷、计算旁路或评估口径错误误当作算法有效性。

---

## 19. 依据与参考资料

### 项目依据

- [P1] 用户提供的《GraphLLM当前流程与算法实现.md》，版本截点2026-11-12。历史流程、候选阈值、特征、数据划分和代码线索来自该文档。
- [P2] 《GraphLLM_算法修复与Codex实施方案_20261112.md》。本文将其称为v1设计；实际历史产物身份仍须以用户仓库权重和manifest核对。
- [P3] 用户在当前讨论中提供的2,048图四臂汇总及71次相同行动、7个资格变化、9次无损新增、135张机会图、epoch5检查点等描述。本文没有原始逐动作记录。

除第1节及明确注明的历史设置外，v2结构和默认参数均为本文设计，不能写成[P1]–[P3]已经验证的结论。

### 公开参考

- [R1] Belkin, Niyogi & Sindhwani. **Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples.** JMLR, 2006. [论文页面](https://www.jmlr.org/papers/v7/belkin06a.html)。用于局部几何先验的背景，不提供v2有效性证明。
- [R2] Doersch, Gupta & Zisserman. **CrossTransformers: spatially-aware few-shot transfer.** NeurIPS, 2020. [论文页面](https://proceedings.nips.cc/paper_files/paper/2020/hash/fa28c6cdf8dd6f41a657c3d7caa5c709-Abstract.html)。用于局部对应的设计背景；本文不是直接复现该方法。
- [R3] Geirhos et al. **Shortcut Learning in Deep Neural Networks.** [论文页面](https://arxiv.org/abs/2004.07780)。用于解释监督和输入结构为什么可能允许捷径；不据此断言本项目已经确认某种捷径。
- [R4] PyTorch. **Autograd mechanics.** [官方文档](https://docs.pytorch.org/docs/stable/notes/autograd.html)。用于区分冻结参数、eval与梯度记录。实际实现遵循仓库已安装版本，不自动升级依赖。
- [R5] Guo et al. **On Calibration of Modern Neural Networks.** ICML, 2017. [论文页面](https://proceedings.mlr.press/v70/guo17a.html)。用于分数与校准概率的区分；本轮不追加温度校准实验。
- [R6] COCO API. **cocoeval.py.** [官方实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。用于AP评估口径；项目须固定实际使用版本。
- [R7] Li et al. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.** ICML, 2023. [论文页面](https://proceedings.mlr.press/v202/li23q.html)。说明冻结大模型配合可训练接口是一类已有方法背景；不代表新适配器无需训练即可理解本项目的视觉图。

**最终实现原则：关系标签描述可核验条件，消息描述条件化视觉读取变化，收益标签描述实际检测动作结果。三者各司其职，并由同一条前向与执行路径连接。**





