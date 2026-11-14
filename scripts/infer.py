"""Run the trained v2 method on one image with the packaged weights and locked policy."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"experiments/factored_instance_v4/routed_association"
RUN = ROOT/"results/objects365-v2-20261113"


def load_model(asset_root):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GraphLLM image inference needs CUDA.")
    asset_root = Path(asset_root).resolve()
    for model_name in ("dino", "clip", "qwen2.5-0.5b"):
        model_dir = asset_root/"models"/model_name
        for file_name in ("config.json", "model.safetensors"):
            if not (model_dir/file_name).is_file():
                raise FileNotFoundError(model_dir/file_name)
    os.environ["BASE"] = str(asset_root)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(SOURCE))
    torch.set_num_threads(3)

    from torch.nn.functional import normalize
    from transformers import AutoProcessor, CLIPModel
    # The bridge initializes the shared-layer module search paths.
    from graphllm_local_message_unique_v2.bridge import Vision
    from donor_transport_probe import Native
    from objects365_cache import text_capacity
    from objects365_claim_cache import ReceiverVisual
    from objects365_instance_gate import ReceiverAttention
    from objects365_receiver_keep_gate import KeepRisk
    from graphllm_receiver_evidence_v2.common import config, digest, load, read, vocabulary
    from graphllm_receiver_evidence_v2.model import Gain, GraphReader
    from graphllm_receiver_evidence_v2.native_bridge import LocalReader, NativeBridge

    cfg = config()
    doc = read(ROOT/"runtime/vocabulary.json")
    data_lock = read(RUN/"data_lock.json")
    policy = read(RUN/"locked_policy.json")
    chain = policy["chain"]
    assert digest(RUN/"data_lock.json") == policy["source"]
    assert digest(RUN/"BUILD_COMPLETE.json") == chain["build"]
    assert json.loads(json.dumps(vocabulary(doc))) == data_lock["vocabulary"]
    assert len(doc["parents"]) == cfg["classes"]
    for stage in ("self", "message", "gain"):
        assert digest(RUN/f"{stage}_best.pt") == chain[stage]
    risk_path = ROOT/"runtime/risk_best.pt"
    risk_hash, = (sha for name, sha in data_lock["prior"]["sources"].items()
                  if name.endswith("/receiver_keep_gate/best.pt"))
    assert digest(risk_path) == risk_hash
    risk = KeepRisk().eval().requires_grad_(False)
    risk.load_state_dict(load(risk_path)["model"], strict=True)

    # Match the qualified NativeBridge feature path without initializing
    # the unused historical Qwen/StarAdapter object inside GraphPack.
    class ImageVision(Vision):
        def __init__(self):
            native = Native(doc["names"])
            text_capacity(native, len(native.ids))
            native.model.requires_grad_(False).eval()
            visual = ReceiverVisual.__new__(ReceiverVisual)
            visual.native = native
            visual.attention = ReceiverAttention(visual)
            clip_path = asset_root/"models/clip"
            visual.clip = CLIPModel.from_pretrained(
                clip_path, local_files_only=True, torch_dtype=torch.float16, use_safetensors=True
            ).cuda().eval().requires_grad_(False)
            visual.processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)

            def text_features(names):
                text = visual.processor(text=[f"a photo of a {name}." for name in names],
                                          padding=True, return_tensors="pt").to("cuda")
                return normalize(visual.clip.get_text_features(**text).float(), dim=-1)

            with torch.inference_mode():
                graph_text = text_features(doc["names"])
                visual.text = text_features(doc["base"])
            self.graph = SimpleNamespace(native=native, reader=visual, text=graph_text,
                                         parents=torch.tensor(doc["parents"], device="cuda"))
            self.visual = visual
            self.risk = risk

    class ImageBridge(NativeBridge):
        def __init__(self):
            self.vision = ImageVision()
            self.local = LocalReader(self.vision.graph.native, cfg)
            self.categories = [doc["categories"][name] for name in doc["base"]]
            self.risk = risk

    bridge = ImageBridge()
    reader = GraphReader(doc, cfg, "cuda")
    reader.receiver.load_state_dict(load(RUN/"self_best.pt")["state"], strict=True)
    reader.add_evidence()
    reader.evidence.load_state_dict(load(RUN/"message_best.pt")["state"], strict=True)
    reader.requires_grad_(False).eval()
    gain = Gain(cfg).requires_grad_(False).eval()
    gain.load_state_dict(load(RUN/"gain_best.pt")["state"], strict=True)
    return bridge, reader, gain, cfg, policy


def predict(path, runtime, image_id=0):
    from PIL import Image
    from counterfactual import predictions
    from graphllm_receiver_evidence_v2.common import digest
    from graphllm_receiver_evidence_v2.predict import freeze_ids, key_index, one_slot, score_rows, select

    bridge, reader, gain, cfg, policy = runtime
    path = Path(path).resolve()
    with Image.open(path) as image:
        width, height = image.size
    frame = dict(id=image_id, source_id=image_id, file_name=path.name, path=str(path),
                 width=width, height=height, sha=digest(path))
    row, _ = bridge.scan(frame)
    classes = key_index(bridge.categories)
    identity = freeze_ids(policy["chain"], digest(RUN/"locked_policy.json"))
    items = [(a, row["nodes"], row["local"]) for a in row["actions"]]
    records, _ = score_rows(reader, gain, bridge.risk, items, classes, cfg, identity)
    key, _ = select(records["full"], policy)
    edited = one_slot(row["base"], key, classes)
    return dict(
        image=path.name,
        action=key,
        detections=predictions(frame, edited, bridge.categories),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--asset-root", type=Path, default=Path(os.environ.get("BASE", ROOT)),
                        help="directory that contains models/dino, models/clip and models/qwen2.5-0.5b")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    runtime = load_model(args.asset_root)
    print(json.dumps(predict(args.image, runtime), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
