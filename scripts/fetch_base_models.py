"""Download and verify the published base-model files used by the release."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def fetch(root):
    from huggingface_hub import hf_hub_download

    for name, model in json.loads((ROOT/"runtime/base_models.json").read_bytes()).items():
        directory = root/"models"/name
        directory.mkdir(parents=True, exist_ok=True)
        for file_name, expected in model["files"].items():
            target = directory/file_name
            if target.is_file() and digest(target) == expected:
                continue
            if target.exists() or target.is_symlink():
                raise ValueError(f"Existing file does not match the experiment: {target}")
            file = Path(hf_hub_download(model["repository"], filename=file_name,
                                        revision=model["revision"], cache_dir=root/"cache/hf/hub",
                                        token=False))
            if digest(file) != expected:
                raise ValueError(f"Unexpected SHA-256 for {model['repository']}/{file_name}")
            temporary = target.with_suffix(target.suffix+".part")
            shutil.copyfile(file, temporary)
            temporary.replace(target)
        print(f"Verified {name} in {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    fetch(parser.parse_args().asset_root.resolve())
