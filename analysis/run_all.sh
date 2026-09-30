#!/usr/bin/env bash
# Rebuild every output, number, table and figure from the corpus, in order.
# Text-bearing intermediates go to $ELS23_WORK (outside the paper folder).
# Model predictions (sentiment, embeddings) are cached there and reused when the
# sentences are unchanged; delete $ELS23_WORK/sentiment and
# $ELS23_WORK/topic_embeddings.npz to recompute them from scratch.
#
#   ./run_all.sh            full run
#   JOBS=4 ./run_all.sh     parallel folds for the classifiers (default 5)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step units;       python3 units.py
step features;    python3 features.py > /dev/null
step sentiment;   python3 sentiment.py | tail -1
step geography;   python3 geography.py
step topics;      python3 topics.py | tail -8
step assemble;    python3 assemble.py
step stats;       python3 stats.py | tail -8
step classify;    python3 classify.py
step robustness;  python3 robustness.py
step environment; python3 environment.py > /dev/null
step numbers;     python3 make_numbers.py
step appendix;    python3 make_appendix.py
step figures;     python3 figures.py
step done
