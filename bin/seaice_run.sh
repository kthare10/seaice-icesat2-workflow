#!/bin/bash
#
# Run one sea ice pipeline stage inside a staged Apptainer image.
#
# Pegasus stages three things into this job's working directory: this wrapper
# (as the job executable), the stage's Python script, and the .sif image built
# by the build_container job. Nothing is baked into the image, so editing a
# stage script never requires a container rebuild.
#
# Usage:
#   seaice_run.sh --container <image.sif> --script <stage.py> [--nv] [--np N] -- <stage args...>
#
# --np N launches the stage under `horovodrun -np N` for data-parallel training;
# it requires a Horovod-enabled image (Apptainer/seaice_gpu_horovod.def).
#
set -euo pipefail

# Some execution environments (HTCondor's local universe in particular) provide
# no HOME, which trips `set -u` and confuses Apptainer. The container is run with
# --no-home anyway, so any writable path will do.
export HOME="${HOME:-$PWD}"

SIF=""
SCRIPT=""
NV=""
NP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container) SIF="$2"; shift 2 ;;
        --script)    SCRIPT="$2"; shift 2 ;;
        --nv)        NV="--nv"; shift ;;
        --np)        NP="$2"; shift 2 ;;
        --)          shift; break ;;
        *) echo "seaice_run.sh: unknown argument '$1' before --" >&2; exit 2 ;;
    esac
done

if [[ -z "$SIF" || -z "$SCRIPT" ]]; then
    echo "Usage: seaice_run.sh --container <image.sif> --script <stage.py> [--nv] -- <args>" >&2
    exit 2
fi

for f in "$SIF" "$SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "seaice_run.sh: '$f' was not staged into $PWD" >&2
        ls -la >&2
        exit 1
    fi
done

RUNNER=""
for candidate in apptainer singularity; do
    if command -v "$candidate" >/dev/null 2>&1; then
        RUNNER="$candidate"
        break
    fi
done
if [[ -z "$RUNNER" ]]; then
    echo "seaice_run.sh: neither apptainer nor singularity is on PATH" >&2
    exit 1
fi

if [[ -n "$NV" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "seaice_run.sh: WARNING --nv requested but nvidia-smi is not present; " \
         "the stage will fall back to CPU" >&2
fi

# --np N runs the stage under horovodrun with one process per GPU. Horovod needs
# all ranks on machines it can reach over MPI; under HTCondor's vanilla universe a
# job gets a single machine, so N is bounded by that worker's GPU count.
LAUNCH=(python3 "/srv/$SCRIPT")
if [[ -n "$NP" ]]; then
    if [[ "$NP" -lt 1 ]]; then
        echo "seaice_run.sh: --np must be >= 1" >&2
        exit 2
    fi
    LAUNCH=(horovodrun -np "$NP" -H "localhost:$NP" python3 "/srv/$SCRIPT")
fi

echo "seaice_run.sh: $RUNNER exec $NV $SIF ${LAUNCH[*]} $*"

# --no-home keeps the worker's home directory out of the container. The job
# directory is bound to /srv and used as the working directory, so relative
# input/output filenames resolve exactly as Pegasus staged them.
#
# HOME is redirected to /srv as well. With --no-home the host home is not mounted,
# so anything that writes under $HOME hits a read-only path: horovodrun creates
# $HOME/.horovod before launching, and matplotlib and friends want a config dir.
# Pointing HOME at the job directory keeps all of that inside the sandbox, where
# it is writable and disappears with the job.
#
# The redirect is applied with `env` INSIDE the container rather than apptainer's
# --env, because apptainer refuses that one:
#   "Overriding HOME environment variable with APPTAINERENV_HOME is not permitted"
# and silently leaves HOME pointing at the unmounted host home.
exec "$RUNNER" exec $NV \
    --no-home \
    --bind "$PWD:/srv" \
    --pwd /srv \
    "$SIF" \
    env HOME=/srv "${LAUNCH[@]}" "$@"
