#!/bin/bash

# for ((numSeg = 64; numSeg < 65; numSeg = numSeg + 1));do
#     # for ((trainIndex = 1; trainIndex < 279; trainIndex = trainIndex + 1));do
#     #     bsub -n 1 "python segmentationEuler_V2.py $((numSeg)) $((trainIndex)) 0"
#     # done;
#     # for ((testIndex = 1; testIndex < 139; testIndex = testIndex + 1));do
#     #     bsub "python segmentationEuler_V2.py $((numSeg)) $((testIndex)) 1"
#     # done;
# done;

# for ((numSeg = 200; numSeg < 201; numSeg = numSeg + 1));do
#     # for ((trainIndex = 1; trainIndex < 279; trainIndex = trainIndex + 1));do
#     #     bsub -n 1 "python segmentationEuler_V2.py $((numSeg)) $((trainIndex)) 0"
#     # done;
#     # for ((testIndex = 1; testIndex < 139; testIndex = testIndex + 1));do
#     #     bsub "python segmentationEuler_V2.py $((numSeg)) $((testIndex)) 1"
#     # done;
# done;
for ((numSeg = 400; numSeg < 401; numSeg = numSeg + 1));do
    # for ((trainIndex = 1; trainIndex < 279; trainIndex = testIndex + 1));do
    #     bsub -n 1 "python segmentationEuler_V2.py $((numSeg)) $((trainIndex)) 0"
    # done;
    for ((testIndex = 1; testIndex < 139; testIndex = testIndex + 1));do
        bsub -n 1 "python segmentationEuler_V2.py $((numSeg)) $((testIndex)) 1"
    done;
done;