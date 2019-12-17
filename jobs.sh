#!/bin/bash

FEATURES="resnext101_32x8d"
OUTDIR="$HOME/SLOW/ir-bench/scoring/thr"

for DATASET in 'oxford' 'paris' 'holidays' 'oxford+flickr100k' 'holidays+mirflickr1m'; do
for CRELU in '' '--crelu'; do

python3 scalar_quantization.py $DATASET $FEATURES $OUTDIR $CRELU

done
done
