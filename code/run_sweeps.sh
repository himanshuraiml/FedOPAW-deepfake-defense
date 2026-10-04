#!/bin/sh
# Sensitivity runs (seed 0, 3 folds): EWC strength and client heterogeneity.
cd "$(dirname "$0")"
PY=python
D=data/dff_n6000_s128.npz
for lam in 100 10000 100000; do
  $PY fedopaw_pilot.py --data $D --out results_sweep --folds 1 2 3 --seeds 0 --rounds 16 --methods kd_ewc_gem --lam $lam --tag lam$lam >> run_sweeps.log 2>&1
done
for al in 0.1 5.0; do
  $PY fedopaw_pilot.py --data $D --out results_sweep --folds 1 2 3 --seeds 0 --rounds 16 --methods fedavg full --alpha $al --tag alpha$al >> run_sweeps.log 2>&1
done
echo SWEEPS_DONE >> run_sweeps.log
