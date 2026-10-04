#!/bin/sh
# Run after run_full.sh prints FULL_DONE in run_full.log.
cd "$(dirname "$0")"
python aggregate.py --results results_full --out generated_full --n_ident 6000
cd .. && sed -i '' 's#\\def\\gen{pilot/generated}#\\def\\gen{pilot/generated_full}#' FedOPAW_IEEE.tex
export PATH=$PATH:$HOME/Library/TinyTeX/bin/universal-darwin
pdflatex -interaction=nonstopmode FedOPAW_IEEE.tex >/dev/null; bibtex FedOPAW_IEEE >/dev/null; pdflatex -interaction=nonstopmode FedOPAW_IEEE.tex >/dev/null; pdflatex -interaction=nonstopmode FedOPAW_IEEE.tex | grep -E "ndefined|Output written"
echo "Now re-read Results/Discussion: hand-typed per-fold BWT numbers and qualitative claims must be checked against generated_full/tab_*.tex"
