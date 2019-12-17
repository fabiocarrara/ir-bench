#!/bin/bash

#FEATURES="resnext101_32x8d"
FEATURES="rmac"
OUTDIR="scoring/ivfpq"
#OUTDIR="$1"
P="-p ivf2500pq1024_H+M100k.faiss"

python3 ivfpq.py holidays+mirflickr1m $FEATURES $OUTDIR -c 1024 -n 16010
python3 ivfpq.py oxford+flickr100k $FEATURES $OUTDIR -c 512 -n 1510

for OUTDIR in "scoring/ivfpq_H+M100k" "scoring/ivfpq_bt4sa"; do
# scalability (auto-defined n. of cells)
for L in $(seq 50000 50000 950000); do
    python3 ivfpq.py holidays+mirflickr1m $FEATURES $OUTDIR -c 1024 -l $L $P
done
done
