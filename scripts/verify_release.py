"""Check the published result, policy, and checkpoint identities on a CPU-only host."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"experiments/factored_instance_v4/routed_association/graphllm_receiver_evidence_v2"
RUN = ROOT/"results/objects365-v2-20261113"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    build = read(RUN/"BUILD_COMPLETE.json")
    data = read(RUN/"data_lock.json")
    repair = read(RUN/"evaluation_repair.json")
    policy = read(RUN/"locked_policy.json")
    sealed = read(RUN/"predictions_manifest.json")
    result = read(RUN/"result.json")
    spec = ROOT/"docs/design/2026-11-13-receiver-evidence-v2.md"
    assert digest(SOURCE/"config.json") == build["config"] == data["config_hash"]
    assert digest(spec) == read(SOURCE/"config.json")["design_sha256"]
    assert repair["completed"] and repair["original"] == build["code"]["evaluate.py"]
    assert {p.name for p in SOURCE.glob("*.py")} == set(build["code"])
    for name, expected in build["code"].items():
        assert digest(SOURCE/name) == (repair["fixed"] if name == "evaluate.py" else expected), name
    assert policy["source"] == build["data"] == digest(RUN/"data_lock.json")
    assert build["smoke"] == digest(RUN/"smoke.json")
    assert policy["chain"] == sealed["lock"]["chain"]
    assert policy["chain"]["build"] == digest(RUN/"BUILD_COMPLETE.json")
    assert policy["policy_top1"] == digest(RUN/"policy_top1_manifest.json")
    assert sealed["lock"]["ids"]["policy_hash"] == digest(RUN/"locked_policy.json")
    assert len(sealed["hashes"]) == 64 and sealed["images"] == result["images"] == 2048
    assert repair["predictions"] == result["prediction_sha"] == digest(RUN/"predictions_manifest.json")
    for key, name in (("result", "result.json"), ("initial_metrics", "result.csv"),
                      ("initial_actions", "action_audit.json"), ("initial_opportunities", "opportunity_images.json")):
        assert repair[key] == digest(RUN/name), name
    for stage in ("self", "message", "gain"):
        q = read(RUN/f"{stage}_qualification.json")
        assert q["completed"] and q["version"] == "training_qualification_v2"
        assert q["best_hash"] == policy["chain"][stage] == digest(RUN/f"{stage}_best.pt")
    risk_hash, = (sha for name, sha in data["prior"]["sources"].items()
                  if name.endswith("/receiver_keep_gate/best.pt"))
    assert digest(ROOT/"runtime/risk_best.pt") == risk_hash
    vocabulary = read(ROOT/"runtime/vocabulary.json")
    assert [vocabulary["categories"][name] for name in vocabulary["base"]] == data["vocabulary"]["text_to_canonical"]
    assert len(read(RUN/"opportunity_images.json")) == result["opportunity"]
    actions = read(RUN/"action_audit.json")
    for row in result["rows"]:
        counts = Counter(a["state"] for a in actions if a["arm"] == row["arm"])
        assert sum(counts.values()) == row["actions"]
        assert all(row[k] == counts[k] for k in ("safe_new", "correct_no_new", "known_harm", "unknown", "verified_other"))
    print("Verified v2 source, policy, weights, prediction manifest, and reported action counts.")


if __name__ == "__main__":
    main()
