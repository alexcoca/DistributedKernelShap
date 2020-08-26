#!/bin/bash
START=$1
END=$2
echo "Workers range tested: {$START..$END}"
cd ./cluster || exit
for i in $(seq "$START" "$END"); do
  echo "Distributing over a pool of size $i actors"
  make -f Makefile.pool deploy
  make -f Makefile.pool upload-script
  make -f Makefile.pool run-experiment WORKERS="$i"
  make -f Makefile.pool pull-results
  make -f Makefile.pool destroy
done
