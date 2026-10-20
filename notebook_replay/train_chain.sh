#!/usr/bin/env bash
cd ~/nb_replay
for arm in 6feat 8feat; do for i in 1 2 3 4 5; do
  apptainer exec ~/seaice-icesat2-workflow/containers/seaice_gpu.sif python3 replay_train_$arm.py > train_${arm}_run$i.log 2>&1
  mv replay_metrics.json replay_metrics_${arm}_run$i.json 2>/dev/null
  echo "$(date -u +%FT%TZ) $arm run $i: $(cat replay_metrics_${arm}_run$i.json 2>/dev/null | tr -d "\n ")" >> chain.log
done; done
echo "$(date -u +%FT%TZ) CHAIN COMPLETE" >> chain.log
