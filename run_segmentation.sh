#!/bin/bash
# Args:
# input folder, output folder, video_name, gpus
mkdir CDCL-human-part-segmentation/"$3"_input
cp "$1"/* CDCL-human-part-segmentation/"$3"_input
cd CDCL-human-part-segmentation/ || exit
mkdir "$3"_output
python3 inference_7parts.py --scale=1 --gpus="$4" --input_folder="$3"_input --output_folder="$3"_output
mv "$3"_output/* ../"$2"
rm -r "$3"_output
rm -r "$3"_input
cd .. || exit
