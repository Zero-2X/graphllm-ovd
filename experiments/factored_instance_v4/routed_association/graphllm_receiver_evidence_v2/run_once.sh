#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=2
out=objects365/graphllm_receiver_evidence_v2
module=graphllm_receiver_evidence_v2.run_once
mkdir -p "$out"
pick_devices() {
    mapfile -t devices < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      sort -t, -nrk2 | awk -F, '$2 >= 14000 { print $1 }')
    (( ${#devices[@]} > 0 ))
    export CUDA_VISIBLE_DEVICES="${devices[0]}"
}
pick_devices
phase() {
    echo "$1" >"$out/stage"
}
py() {
    python -u -m "$module" "$@"
}
parallel() {
    local tag="$1"
    shift
    pick_devices
    local -a pids=()
    local n="${#devices[@]}" rc=0
    for ((worker=0; worker<n && worker<8; worker++)); do
        (
            export CUDA_VISIBLE_DEVICES="${devices[worker]}"
            for ((shard=worker; shard<8; shard+=n)); do
                py "$@" --shard "$shard" >"$out/${tag}_${shard}.log" 2>&1
            done
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    return "$rc"
}
if [[ ! -f "$out/data_lock.json" ]]; then
    phase init
    py init
fi
if [[ ! -f "$out/smoke.json" ]]; then
    phase smoke
    py smoke
fi
if [[ "${1:-}" == smoke ]]; then
    phase smoke_complete
    exit 0
fi
if [[ ! -f "$out/BUILD_COMPLETE.json" ]]; then
    for split in train monitor; do
        phase "cache_$split"
        parallel "cache_$split" cache --split "$split"
    done
    phase seal
    py seal
fi
for name in self message; do
    if [[ -f "$out/${name}_qualification.json" ]]; then continue; fi
    phase "$name"
    pick_devices
    world=1
    if (( ${#devices[@]} >= 4 )); then world=4; fi
    if [[ -f "$out/${name}_resume.pt" ]]; then
        world=$(python -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["world"])' "$out/${name}_resume.pt")
    fi
    if (( world == 1 )); then
        py "$name"
    else
        (( ${#devices[@]} >= world ))
        csv=$(IFS=,; echo "${devices[*]:0:world}")
        CUDA_VISIBLE_DEVICES="$csv" python -u -m torch.distributed.run --standalone --nproc-per-node="$world" \
            --module "$module" "$name"
    fi
done
if [[ ! -f "$out/gain_qualification.json" ]]; then
    phase gain
    pick_devices
    py gain
fi
if [[ ! -f "$out/locked_policy.json" ]]; then
    phase policy_scores
    parallel policy_scores policy-scores
    phase policy_lock
    py lock
fi
if [[ ! -f "$out/predictions_manifest.json" ]]; then
    phase predict
    parallel predict predict
    phase seal_predictions
    py seal-predictions
fi
if [[ ! -f "$out/result.json" ]]; then
    phase evaluate
    CUDA_VISIBLE_DEVICES= py evaluate
fi
phase complete
