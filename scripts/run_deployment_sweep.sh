#!/bin/sh
# Copyright (C) 2026, François-Guillaume Fernandez.
# Licensed under the Apache License 2.0; see LICENSE.

# targets.tsv: SSH alias (or local), Python executable, remote working directory.
set -eu
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 BUNDLE TARGETS.tsv RESULTS" >&2
    exit 2
fi
bundle=$(cd "$1" && pwd)
targets=$2
mkdir -p "$3"
results=$(cd "$3" && pwd)
checksum=$(shasum -a 256 "$bundle/bundle.json")
bundle_id=${checksum%% *}
while IFS="$(printf '\t')" read -r host python directory || [ -n "$host" ]; do
    case "$host" in ''|'#'*) continue ;; esac
    # Paths are intentionally whitespace-free; never interpolate arbitrary shell syntax into SSH.
    case "$host:$python:$directory" in *[!a-zA-Z0-9_@./:-]*|-*) echo "Invalid target row: $host" >&2; exit 2 ;; esac
    case "$host" in .|..|*/*|*:*) echo "Invalid SSH alias: $host" >&2; exit 2 ;; esac
    [ -n "$python" ] && [ -n "$directory" ]
    if [ "$host" != local ]; then
        case "$python:$directory" in /*:/*) ;; *) echo "Remote paths must be absolute: $host" >&2; exit 2 ;; esac
    fi
done < "$targets"
pids=
while IFS="$(printf '\t')" read -r host python directory || [ -n "$host" ]; do
    case "$host" in ''|'#'*) continue ;; esac
    mkdir -p "$results/$host"
    (
        if [ "$host" = local ]; then
            OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python" "$bundle/scripts/eval_deployment.py" \
                "$bundle" "$results/$host" --resume
        else
            destination="$directory/bundles/$bundle_id"
            remote_results="$directory/results/$bundle_id"
            ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
                "test -x '$python' && mkdir -p '$destination' '$remote_results'"
            rsync -r --checksum --delete --exclude '__pycache__' "$bundle/" "$host:$destination/"
            status=0
            ssh -o BatchMode=yes "$host" \
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 '$python' '$destination/scripts/eval_deployment.py' '$destination' '$remote_results' --resume" || status=$?
            rsync -r "$host:$remote_results/" "$results/$host/"
            exit "$status"
        fi
    ) >"$results/$host/launcher.log" 2>&1 &
    pids="$pids $!"
done < "$targets"
[ -n "$pids" ] || { echo "No targets provided" >&2; exit 2; }
status=0
for pid in $pids; do
    wait "$pid" || status=1
done
exit "$status"
