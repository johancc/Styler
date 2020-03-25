#!/bin/bash
echo "Downloading segmentation data..."
git submodule init
git submodule update
cd CDCL-human-part-segmentation || exit
bash fetch_data.sh
rm input/*
rm output/*
echo "Creating conda environment..."
cd .. || exit
conda env create -f environment.yaml
conda activate styler

