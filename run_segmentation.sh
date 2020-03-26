#!/bin/bash
# Args:
# input folder, output folder
mv "$1"/* CDCL-human-part-segmentation/input
cd CDCL-human-part-segmentation/ || exit
python3 inference_7parts.py --scale=1
rm input/*
cd .. || exit
mv CDCL-human-part-segmentation/output/* "$2"
