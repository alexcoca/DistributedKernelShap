#!/bin/bash
START=$1
END=$2
echo "Workers range tested: {$START..$END}"
cd cluster || exit
for i in $(seq "$START" "$END"); do
  make deploy
  make upload-script
  make run-experiment WORKERS="$i"
  make pull-results
  make destroy
done
