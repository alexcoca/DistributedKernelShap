#!/bin/bash
START=$1
END=$2
BATCH_MODE=$3
BATCH_SIZE=(1 2 5 10 15 20)
echo "Workers range tested: {$START..$END}"
echo "Batch mode: $BATCH_MODE"
cd ./cluster || exit
for i in $(seq "$START" "$END"); do
  for j in "${BATCH_SIZE[@]}"; do
    echo "Distributing explanations over $i workers"
    echo "Current batch size: $j instances"
    make -f Makefile.serve deploy
    make -f Makefile.serve upload-script
    make -f Makefile.serve run-experiment WORKERS="$i" BATCH="$j" BATCH_MODE="$BATCH_MODE"
    make -f Makefile.serve pull-results
    make -f Makefile.serve destroy
  done
done
