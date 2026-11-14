# 运行与复现

## 环境

实测环境为 Linux、Python 3.8.10、PyTorch 2.4.1+cu121、TorchVision 0.19.1+cu121、Transformers 4.45.1、NVIDIA RTX 3090。数据缓存、训练和四组验证使用 CUDA。复刻时先创建虚拟环境并安装官方 CUDA 12.1 包。 [PyTorch 版本安装表](https://pytorch.org/get-started/previous-versions/)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

## 单图推理

将基础权重和各自配置放在一个资产根目录的 `models/` 下。`runtime/base_models.json` 记录实验匹配的模型来源、ref 和文件 SHA-256。模型目录使用 `dino`、`clip` 和 `qwen2.5-0.5b`。可使用仓库的下载脚本从公开模型页取得并核对文件，也可直接传入 sr008 已有的模型根目录。

```bash
python scripts/fetch_base_models.py --asset-root /path/to/assets
python scripts/infer.py /path/to/image.jpg --asset-root /path/to/assets
```

CLIP 的 `model.safetensors` 使用官方模型仓库转换版的固定提交，与本次实验文件匹配。 [CLIP safetensors 来源](https://huggingface.co/openai/clip-vit-base-patch32/discussions/59)

仓库中提供已锁定的 v2 权重及风险头。单图入口使用相同的 365 类词表、候选、局部读取和收益分界。每个检测对象有 `image_id`、`category_id`、`bbox`（原图像素 xywh）和原始 `score`。`action` 为 `[image, query, slot, original_category_id, proposed_category_id]` 或 null。 [类别映射](../runtime/vocabulary.json)

## 代码目录与资产

- `experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2/` 包含最新版完整配置、视觉缓存、标签、训练、策略、预测、评估与检查。
- 同级 `graphllm_local_message_unique_v2/` 是报告中的冻结 v1。其他纳入的 `experiments/` 源码包含公用候选、DINO/CLIP 接口、历史对照、原生风险特征和 AP 评估的传递依赖。目录结构保留训练时的模块路径，便于核对源码哈希。
- `runtime/` 放类别映射、风险权重及基础模型清单。`results/objects365-v2-20261113/` 存最佳检查点、训练曲线、锁定策略、结果与动作审计。预测清单记录了 64 个预测块的哈希。大块预测、数据集和基础模型由外部实验资产目录提供。

## 完整固定验证

在 sr008 的独立 checkout 中运行。将 `BASE` 指向原实验根，其中已有 Objects365 图像与清单、基础模型、冻结 v1 及风险头依赖。现有清单保留原字节和图像路径，固定旧对照的依赖验证需要原路径可读。首次在这一 checkout 上运行时，连接脚本为相同来源的固定输入建立符号链接。各阶段产生的缓存、权重、锁定策略、预测和评估记录写入 checkout 自己的输出目录。

```bash
export BASE=/path/to/original-experiment-root
export HF_HUB_OFFLINE=1
python scripts/link_experiment_assets.py --asset-root "$BASE"
cd experiments/factored_instance_v4/routed_association
export PYTHONPATH=.
bash graphllm_receiver_evidence_v2/run_once.sh
```

训练入口包含 `init`、`smoke`、`cache`、`seal`、`self`、`message`、`gain`、`policy-scores`、`lock`、`predict`、`seal-predictions` 与 `evaluate`。完整重跑需要原实验的固定输入，尤其是来源清单、旧对照和标注覆盖。 [实现盘点](IMPLEMENTATION.md) · [原始方案](design/2026-11-13-receiver-evidence-v2.md)

训练先用 6,038 图和 807 张监控图，策略集为 871 图。固定测试用训练前锁定的 2,048 张留出图。预测仅使用去掉标注的清单。评估前对各预测文件的哈希及覆盖进行检查。固定产物位于 `objects365/graphllm_receiver_evidence_v2/`，仓库的报告为已完成运行的记录。

本仓库评估源码包含已验证的 COCOeval 类别 ID JSON 序列化修复。原结果的构建收据仍保留修复前源码哈希，验证脚本会同时检查收据中的原始和修复后身份。 [修订收据](../results/objects365-v2-20261113/evaluation_repair.json)
