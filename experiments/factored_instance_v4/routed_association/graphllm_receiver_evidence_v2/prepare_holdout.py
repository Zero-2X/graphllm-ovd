"""Lock one new holdout from the already present Objects365 archive."""
from pathlib import Path

import objects365_receiver_keep_holdout_data as sampler

sampler.OUT = sampler.FOLDER/"graphllm_receiver_evidence_v2"
sampler.NEW = sampler.OUT/"manifest.json"
sampler.PLAN = Path(__file__).with_name("config.json")
sampler.COUNT = 2048

if __name__ == "__main__":
    sampler.prepare()
