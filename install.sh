#!/bin/bash
echo "Creating a virtual environment..."
virtualenv venv
. venv/bin/activate
echo "Downloading segmentation data..."
git submodule init
git submodule update
cd CDCL-human-part-segmentation || exit
bash fetch_data.sh
rm input/*
rm output/*
echo "Installing dependencies..."
cd .. || exit
pip3 install -r requirements.txt

