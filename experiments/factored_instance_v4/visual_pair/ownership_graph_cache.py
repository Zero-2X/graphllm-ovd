"""Extract annotation-free evidence for the indexed native-query graphs."""
import argparse
import json

import numpy as np
from PIL import Image
import torch

from dense_donor import ROOT, SHARDS, save_json, vocabulary
from donor_transport_probe import xyxy
from identity_features import FEATURES, FootprintNative, pair_features
from operational_cache import sources
from ownership_witness import candidate_bank
from train_factor import by_image

NODE_FIELDS = ("class_logit", "log_area", "previous_iou", "early_iou", "support0", "support1", "support2")
EDGE_FIELDS = FEATURES[:-2]


def node_features(native, boxes, logits, q, c):
    b = boxes[q]
    span = (b[:, 2:] - b[:, :2]).clamp_min(1e-6)
    area = span.prod(1)
    values = [logits[q, c], area.log()]
    refs = native.last_output.intermediate_reference_points[0]
    for i in (-2, -4):
        p = xyxy(refs[i, q])
        inter = (torch.minimum(b[:, 2:], p[:, 2:]) - torch.maximum(b[:, :2], p[:, :2])).clamp_min(0).prod(1)
        values.append(inter / (area + (p[:, 2:] - p[:, :2]).clamp_min(1e-6).prod(1) - inter).clamp_min(1e-8))
    for points, weights in native.footprint.values():
        x = points[q]
        inside = ((x >= b[:, None, :2]) & (x <= b[:, None, 2:])).all(-1)
        values.append((weights[q] * inside).sum(1))
    return torch.stack(values, dim=1).cpu()


def edge_features(native, boxes, logits, triples):
    if not len(triples):
        return torch.empty(0, len(EDGE_FIELDS))
    keys = triples.copy()
    keys[:, :2] = np.sort(keys[:, :2], axis=1)
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    values = []
    for chunk in np.array_split(unique, (len(unique) + 511) // 512):
        values.append(pair_features(native, boxes, logits, *chunk.T)[:, :-2])
    return torch.from_numpy(np.concatenate(values)[inv])


def pack_image(native, boxes, logits, table, raw_rows, offsets, witnesses, start_group, start_node, witness_start=0):
    rows, node_ids, node_labels, group_ids = [], [], [], []
    endpoint_pairs, endpoint_labels, links, link_pairs, link_labels = [], [], [], [], []
    for local, entry in enumerate(table):
        j = int(entry[-1])
        row = raw_rows[j]
        q, r, c, g, h = map(int, row[[1, 4, 2, -3, -2]])
        ids, owners = witnesses[offsets[j]:offsets[j + 1]].T
        assert np.array_equal(ids, candidate_bank(boxes, logits, np.array([q, r]), c, witness_start))
        n = len(ids)
        bank = np.column_stack((ids, np.full(n, c)))
        node_ids.append(bank)
        node_labels.append(owners >= 0)
        group_ids.append(np.full(n, start_group + local))
        labels = np.where((owners[:, None] >= 0) & (np.array([g, h]) >= 0), owners[:, None] == [g, h], -1)
        endpoint_labels.append(labels)
        endpoint_pairs.append(np.column_stack((np.tile([q, r], n), np.repeat(ids, 2), np.full(2 * n, c))))
        a, b = np.triu_indices(n, k=1)
        links.append(np.column_stack((a, b)) + start_node)
        link_pairs.append(np.column_stack((ids[a], ids[b], np.full(len(a), c))))
        link_labels.append(np.where((owners[a] >= 0) & (owners[b] >= 0), owners[a] == owners[b], -1))
        start_node += n
        rows.append(row)
    node_ids = np.concatenate(node_ids).astype(np.int64)
    endpoint_pairs = np.concatenate(endpoint_pairs).astype(np.int64)
    link_pairs = np.concatenate(link_pairs).astype(np.int64)
    edge = edge_features(native, boxes, logits, np.concatenate((endpoint_pairs, link_pairs)))
    node = node_features(native, boxes, logits, *node_ids.T)
    assert torch.isfinite(node).all() and torch.isfinite(edge).all()
    return dict(rows=torch.from_numpy(np.asarray(rows)), node=node, node_y=torch.from_numpy(np.concatenate(node_labels)),
                group=torch.from_numpy(np.concatenate(group_ids).astype(np.int64)),
                endpoint=edge[:len(endpoint_pairs)].reshape(-1, 2, len(EDGE_FIELDS)),
                endpoint_y=torch.from_numpy(np.concatenate(endpoint_labels).astype(np.int8)),
                links=torch.from_numpy(np.concatenate(links).astype(np.int64)), edge=edge[len(endpoint_pairs):],
                edge_y=torch.from_numpy(np.concatenate(link_labels).astype(np.int8)))


@torch.inference_mode()
def cache(split, shard, smoke):
    receipt = json.loads((ROOT / "ownership_graph_index_receipt.json").read_text())
    assert receipt["nfine"] == 48 and receipt["disjoint_images"]
    assert receipt["splits"][split]["supported_labels_agree"]
    with np.load(ROOT / f"ownership_graph_index_{split}.npz") as f:
        rows, offsets, witness = (f[k] for k in ("rows", "offsets", "witness"))
    index = by_image(np.column_stack((rows[:, 0], np.arange(len(rows)))))
    source, tag, _ = sources(split)
    manifest = json.loads((source / "manifest.json").read_text())
    names, nfine, _ = vocabulary(manifest, tag)
    assert nfine == 48
    images = [row for row in manifest["splits"][tag] if row["id"] in index][shard::SHARDS]
    if smoke:
        selected = sorted(images, key=lambda r: -sum(rows[index[r["id"]][:, -1].astype(int), -1] == 2))
        images = selected[:4]
    native = FootprintNative(names)
    items = []
    groups = nodes = 0
    for j, image in enumerate(images, 1):
        iid = image["id"]
        im = Image.open(source / "images" / image["dataset"] / image["file_name"]).convert("RGB")
        boxes, logits, _, _ = native.forward(im)
        pack = torch.load(source / "cache" / tag / f"{iid}.pt", map_location="cpu", weights_only=True)
        assert torch.equal(boxes.cpu(), pack["boxes"]) and torch.equal(logits.cpu(), pack["logits"])
        item = pack_image(native, boxes, logits, index[iid], rows, offsets, witness, groups, nodes)
        assert (item["rows"][:, 0] == iid).all()
        items.append(item)
        groups += len(item["rows"])
        nodes += len(item["node"])
        if j % 50 == 0 or j == len(images):
            print("ownership graph cache", split, shard, j, len(images), groups, nodes, flush=True)
    data = {k: torch.cat([d[k] for d in items]) for k in items[0]}
    data.update(node_fields=NODE_FIELDS, edge_fields=EDGE_FIELDS, native_exact=True)
    summary = dict(images=len(images), groups=groups, witnesses=nodes, witness_edges=len(data["edge"]),
                   group_targets=torch.bincount(data["rows"][:, -1].long(), minlength=3).tolist())
    if smoke:
        assert all(summary["group_targets"])
        save_json(ROOT / "ownership_graph_smoke.json", summary)
    else:
        path = ROOT / f"ownership_graph_features_{split}_{shard}.pt"
        torch.save(data, path.with_suffix(".part"))
        path.with_suffix(".part").replace(path)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "mine", "cal"), default="train")
    parser.add_argument("--shard", type=int, choices=range(SHARDS), default=0)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    cache(a.split, a.shard, a.smoke)
