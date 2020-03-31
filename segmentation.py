"""
Runs segmentation on a batch of frames. Wrapper for the CDCL-human-part-segmentation.
"""
import os
import subprocess
import tqdm
import cv2
from PIL import Image

def segment(base_name: str, frames_folder: str, output_folder, gpus: int):
    for f in [frames_folder, output_folder]:
        if not os.path.exists(f):
            os.makedirs(f)

    command = "./run_segmentation.sh {} {} {} {}".format(frames_folder, output_folder, base_name, gpus)
    p = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    assert os.path.exists(output_folder)
    return output_folder


def apply_style_over_segmentation(original_folder: str, style_folder: str,
                                  segmentation_folder: str, output_folder: str,
                                  mode: int):
    """
    mode - 0 if the human should be styled, 1 if the background should be styled.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    original_frames = sorted(os.listdir(original_folder))
    styled_frames = sorted(os.listdir(style_folder))
    segmented_frames = sorted(os.listdir(segmentation_folder))
    final_frames = []
    for i in tqdm.tqdm(list(range(len(original_frames)))):
        original_frame_path = os.path.join(original_folder, original_frames[i])
        change_frame_path = os.path.join(style_folder, styled_frames[i])
        segmented_frame_path = os.path.join(segmentation_folder, segmented_frames[i])
        out_name = os.path.join(output_folder, original_frames[i])
        merged_frame = apply_style_over_original_with_mask(original_frame_path, change_frame_path, segmented_frame_path, out_name, mode)
        final_frames.append(merged_frame)
    return output_folder



def make_mask_bw(mask):
    grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (_, blackAndWhiteImage) = cv2.threshold(grayImage, 1, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def apply_style_over_original_with_mask(original_file, style_file, mask_file, output_file_name, mode: int = 0):
    original = cv2.imread(original_file, 1)
    style = cv2.imread(style_file, 1)
    mask = cv2.imread(mask_file, 1)

    # Make the mask bw to perform bitwise operations, note that we still use RGB to keep the same shape.
    mask_bw = cv2.cvtColor(make_mask_bw(mask),cv2.COLOR_GRAY2RGB)
    inv_mask_bw = cv2.bitwise_not(mask_bw)

    if mode == 1:
        mask_bw, inv_mask_bw = inv_mask_bw, mask_bw
    # mask_bw are the pixels we want to keep from the styled
    # inv_mask_bw are the pixels we want to keep from the original
    original_with_mask = cv2.bitwise_and(original, inv_mask_bw)
    styled_with_mask = cv2.bitwise_and(style, mask_bw)
    merged_final = cv2.bitwise_or(original_with_mask, styled_with_mask)
    cv2.imwrite(output_file_name, merged_final)
    return merged_final

if __name__ == "__main__":
    apply_style_over_segmentation("original", "styled", "segmented", "out", 0)