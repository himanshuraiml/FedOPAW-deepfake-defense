#!/bin/sh
# Full run: 6000 identities (24000 images), 3 seeds x 3 held-out generators x 9 methods, 16 rounds, plus watermark study per model.
cd "$(dirname "$0")"
PY=python
$PY fedopaw_pilot.py --data data/dff_n6000_s128.npz --out results_full --folds 1 2 3 --seeds 0 1 2 --rounds 16 --save_models > run_full.log 2>&1
for s in 0 1 2; do for h in 1 2 3; do
  $PY wm_experiment.py --data data/dff_n6000_s128.npz --model results_full/s${s}_held${h}_full.pt --held $h --seed $s --out results_full > wm_full_s${s}_h${h}.log 2>&1
done; done
echo FULL_DONE >> run_full.log
