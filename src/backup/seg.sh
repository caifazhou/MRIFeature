#!/bin/bash
for ((numSeg = 20; numSeg <21; numSeg = numSeg + 1)); do
    for ((trainIndex = 1; trainIndex < 2; trainIndex = trainIndex + 1)); do
        python3 segmentationEuler_V2.py $((numSeg)) $((trainIndex)) 0
    done;
    for ((testIndex = 1; testIndex < 2; testIndex = trainIndex + 1)); do
        python3 segmentationEuler_V2.py $((numSeg)) $((testIndex)) 1
    done;
done;