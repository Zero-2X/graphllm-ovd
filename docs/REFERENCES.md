# 参考文献

## 检测器、语义模型和数据

1. Liu et al. **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.** ECCV, 2024. 本项目用它提取原始检测、decoder query 和 encoder 空间特征。 [论文](https://arxiv.org/abs/2303.05499) · [官方代码](https://github.com/IDEA-Research/GroundingDINO)
2. Radford et al. **Learning Transferable Visual Models From Natural Language Supervision.** ICML, 2021. 裁片 CLIP 提供图节点外观、关系候选和类别相似度。 [论文](https://proceedings.mlr.press/v139/radford21a.html)
3. Qwen Team. **Qwen2.5 Technical Report.** 2024. 冻结 Qwen2.5-0.5B 处理接收端和供体的视觉、边与文本 token，返回内部隐藏状态。 [技术报告](https://arxiv.org/abs/2412.15115) · [0.5B 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B)
4. Shao et al. **Objects365: A Large-Scale, High-Quality Dataset for Object Detection.** ICCV, 2019. 最新实验从 Objects365 V2 的 train 部分创建项目训练、监控、策略和留出划分。 [论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.html) · [数据站点](https://www.objects365.org/download.html)
5. Schulter et al. **OmniLabel: A Challenging Benchmark for Language-Based Object Detection.** ICCV, 2023. 项目早期任务使用该基准。最新 v2 表格来自 Objects365，OmniLabel 名单用于历史数据排重。 [论文](https://openaccess.thecvf.com/content/ICCV2023/html/Schulter_OmniLabel_A_Challenging_Benchmark_for_Language-Based_Object_Detection_ICCV_2023_paper.html) · [基准官网](https://www.omnilabel.org/)

## 方法背景

6. Belkin, Niyogi and Sindhwani. **Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples.** JMLR, 2006. 局部几何是寻找参考的出发点。v2 测试的是附带来源、对象和类别条件的局部参考，流形有效性仍需实验支持。 [论文](https://www.jmlr.org/papers/v7/belkin06a.html)
7. Doersch, Gupta and Zisserman. **CrossTransformers: Spatially-aware few-shot transfer.** NeurIPS, 2020. 空间对应提供设计背景。v2 由供体调整查询，实际读取的视觉值来自接收框。 [论文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/fa28c6cdf8dd6f41a657c3d7caa5c709-Abstract.html)
8. Li et al. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.** ICML, 2023. 冻结基础模型与可训练跨模态接口的相关工作。 [论文](https://proceedings.mlr.press/v202/li23q.html)
9. Geirhos et al. **Shortcut Learning in Deep Neural Networks.** Nature Machine Intelligence, 2020. 用于分析供体身份、模板和关系监督可能产生的预测捷径。 [作者预印本](https://arxiv.org/abs/2004.07780)
10. Guo et al. **On Calibration of Modern Neural Networks.** ICML, 2017. 用于区分排序分数与已校准的概率。训练按类加权的收益 logit 表示相对排序证据。 [论文](https://proceedings.mlr.press/v70/guo17a.html)
11. Lin. **Divergence Measures Based on the Shannon Entropy.** IEEE Transactions on Information Theory, 1991. Jensen–Shannon 分歧的原始来源。v1 将其用于供体融合，v2 仅记录两条真实边的分歧。 [论文](https://doi.org/10.1109/18.61115)

## 实现依据

12. **COCO API。** bbox AP 和 AP50、AP75 用 `pycocotools 2.0.7` 的 COCOeval。项目另做六档固定删改回放以核验目标级匹配。 [官方实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
13. **PyTorch Autograd mechanics。** DINO、CLIP 与 Qwen 权重冻结。训练期间，Qwen 后缀保留从损失到适配器的梯度。 [官方文档](https://docs.pytorch.org/docs/2.4/notes/autograd.html)

本项目原始方案、结果与引用用途见[实施方案](design/2026-11-13-receiver-evidence-v2.md)和[最新留出评估](RESULTS.md)。
