#!/bin/bash
echo "Downloading segmentation data..."
git submodule init
git submodule update
bash CDCL-human-segmentation/fetch_data.sh
rm CDCL-human-segmentation/input/*
rm CDCL-human-segmentation/output/*
echo "Installing dependencies..."
pip3 install -r requirements.txt --user

