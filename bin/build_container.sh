#!/bin/bash
#
# Build an Apptainer (.sif) image from a definition file, as a workflow job.
#
# Pegasus stages the .def file and anything its %files section needs into this
# job's working directory, runs this script there, and then stages the resulting
# .sif out to whichever jobs declare it as an input.
#
# This job is pinned to the submit host (site "local") by workflow_generator.py:
# unprivileged `apptainer build` needs user-namespace/fakeroot support that the
# compute nodes do not currently provide, and building here avoids shipping a
# multi-GB image back from a worker before it can be redistributed.
#
# Usage:
#   build_container.sh --def <file.def> --output <image.sif> [--force]
#
set -euo pipefail

# HTCondor's local universe hands the job a minimal environment with no HOME,
# which both `set -u` and Apptainer itself object to. Give everything a home and
# a writable cache/tmp before doing anything else. APPTAINER_CACHEDIR is set by
# the workflow generator to a persistent path so base image layers survive
# between runs; APPTAINER_TMPDIR needs room for the unpacked image (~2x its size).
export HOME="${HOME:-$PWD}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

DEF=""
OUT=""
FORCE="--force"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --def)     DEF="$2"; shift 2 ;;
        --output)  OUT="$2"; shift 2 ;;
        --no-force) FORCE=""; shift ;;
        --force)   FORCE="--force"; shift ;;
        *) echo "build_container.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

if [[ -z "$DEF" || -z "$OUT" ]]; then
    echo "Usage: build_container.sh --def <file.def> --output <image.sif>" >&2
    exit 2
fi

if [[ ! -f "$DEF" ]]; then
    echo "build_container.sh: definition file '$DEF' not found in $PWD" >&2
    ls -la >&2
    exit 1
fi

# Prefer apptainer, fall back to the singularity alias on older hosts.
BUILDER=""
for candidate in apptainer singularity; do
    if command -v "$candidate" >/dev/null 2>&1; then
        BUILDER="$candidate"
        break
    fi
done
if [[ -z "$BUILDER" ]]; then
    echo "build_container.sh: neither apptainer nor singularity is on PATH" >&2
    exit 1
fi

echo "=== build_container.sh ==="
echo "host:       $(hostname)"
echo "builder:    $BUILDER ($($BUILDER --version 2>&1))"
echo "definition: $DEF"
echo "output:     $OUT"
echo "workdir:    $PWD"
echo "cache:      $APPTAINER_CACHEDIR"
echo "tmp:        $APPTAINER_TMPDIR"
echo "=========================="

start=$(date +%s)
# Unprivileged build. The base image layers are cached in APPTAINER_CACHEDIR, so
# repeat builds of an unchanged definition are much faster than the first.
status=0
"$BUILDER" build $FORCE "$OUT" "$DEF" || status=$?
elapsed=$(( $(date +%s) - start ))

if [[ $status -ne 0 || ! -f "$OUT" ]]; then
    echo "build_container.sh: build FAILED (exit $status) after ${elapsed}s" >&2
    exit 1
fi

echo "build_container.sh: built $OUT ($(du -h "$OUT" | cut -f1)) in ${elapsed}s"

# Fail loudly here rather than in every downstream job if the image is unusable.
if ! "$BUILDER" exec "$OUT" python3 -c "import sys; print('container python', sys.version.split()[0])"; then
    echo "build_container.sh: built image has no working python3" >&2
    exit 1
fi
