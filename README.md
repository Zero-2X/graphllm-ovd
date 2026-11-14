# GraphLLM v2 开放词汇检测纠错

在单张图里，Grounding DINO 有时已经给出了覆盖目标的低分框，但类别仍有误。项目先用实例图找到同图高分参考，再用冻结的 Qwen2.5-0.5B 核验供体和接收框的关系。供体引导模型重读接收框内部的局部视觉证据。收益头据此选择是否改类，每图最多修改一个已有检测槽位。

| 文档 | 内容 |
| --- | --- |
| [项目流程与算法介绍](docs/WORKFLOW.md) | 输入输出、图的节点和边、三个训练阶段、推理与评估 |
| [算法详情](docs/METHOD.md) | 张量、公式、监督、排序与可执行条件 |
| [11 月 13 日实施方案](docs/design/2026-11-13-receiver-evidence-v2.md) | 审查后的原始设计与验收条件 |
| [实现盘点](docs/IMPLEMENTATION.md) | 实际源码、操作口径、工程检查和数据规模 |
| [参考文献](docs/REFERENCES.md) | 原论文、官方实现和与项目的关系 |
| [复现说明](docs/REPRODUCE.md) | 环境、外部模型、单图推理和在原实验资产上的完整运行 |
| [留出集结果](docs/RESULTS.md) | 四组指标、逐图配对、动作审计与局限 |

## 运行最新版

单图入口加载本仓库已验证的 self、message、gain 三份检查点和固定风险头。外部目录需提供 DINO、CLIP 与 Qwen 基础模型。

```bash
python scripts/infer.py photo.jpg --asset-root /path/to/assets
```

运行输出包含最终检测框、类别 ID、原检测分数，以及本图选中的改类动作。JSON 使用 Objects365 原始类别 ID，可查阅[类别映射](runtime/vocabulary.json)。

在只有 Python 的环境中，可以核对已上传的源码、结果和检查点身份。

```bash
python scripts/verify_release.py
```

## 最新验证

| 同一批 2,048 张 Objects365 留出图 | 执行改类 | 确认改对 | 无损补回 | 已知损害 | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生 Grounding DINO | 0 | 0 | 0 | 0 | 37.153004 |
| 冻结 v1，JS 开启 | 75 | 44 | 12 | 2 | 37.153620 |
| 完整 v2 | 37 | 20 | 2 | 0 | 37.153060 |
| v2 供体消息置零 | 103 | 64 | 5 | 1 | 37.153242 |

完整 v2 的 37 个动作中有 16 个未确证。原生框坐标和检测分数固定。供体消息影响了 72 张图的动作，其中挡住了置零时的一次匹配损失，也挡住了三次可以无损补回的动作。当前结果支持继续研究消息与收益分的接口，整体检测提升证据仍不足。 [详细结果与对账](docs/RESULTS.md)

算法实现位于 GraphLLM v2 目录，共用的候选生成、风险证据和评估源码也随项目保存。冻结 v1 的混合熵策略在四组验证中作为一组对照。 [算法源码](experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2) · [源码与依赖](docs/REPRODUCE.md#代码目录与资产)
