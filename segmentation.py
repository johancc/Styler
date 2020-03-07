"""
Runs segmentation on a batch of frames. Wrapper for the CDCL-human-part-segmentation.
"""
import os
import subprocess


def segment(frames_folder: str, output_folder):
    for f in [frames_folder, output_folder]:
        if not os.path.exists(f):
            os.makedirs(f)

    command = "./run_segmentation.sh {} {}".format(frames_folder, output_folder)
    p = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    return out, err


def apply_style_over_segmentation(original_folder: str, style_folder: str,
                                  segmentation_folder: str, output_folder: str,
                                  mode: int):
    pass
