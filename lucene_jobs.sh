#!/bin/bash

FEATURES="rmac_rot"
OUTDIR="scoring/lucene-thr-sq"

python3 sq_lucene.py holidays+mirflickr1m $FEATURES $OUTDIR -c -t 20 24 28 32 36
python3 sq_lucene.py oxford+flickr100k $FEATURES $OUTDIR -c

# scalability (auto-defined n. of cells)
# for L in $(seq 50000 50000 950000); do
for L in 50000 100000 250000 500000 750000 950000; do
    python3 sq_lucene.py holidays+mirflickr1m $FEATURES $OUTDIR -c -l $L -t 16 18 20 22 24 26 32
done
