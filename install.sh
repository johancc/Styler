#!/bin/bash
echo "Downloading segmentation data..."
git submodule init
git submodule update
bash CDCL-human-part-segmentation/fetch_data.sh
rm CDCL-human-part-segmentation/input/*
rm CDCL-human-part-segmentation/output/*
echo "Installing dependencies..."
pip3 install -r requirements.txt --user

