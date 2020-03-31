"""
Runs segmentation on a batch of frames. Wrapper for the CDCL-human-part-segmentation.
"""
import os
import subprocess
import tqdm
from PIL import Image

# The human mask can be any one of the 4 mask colors. We have to have some range of pixel values which
# we can accept since the mask color + underlying human has some variability.
MASK_DIFF_THRESHOLD = 65
MASK_COLORS = [(240, 90, 100), # red
               (160, 80, 10), # orange
               (190, 179, 18), # yellow
               (155, 143, 139), # white-gray
               ]

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
        merged_frame = merge_difference(original_frame_path, segmented_frame_path, change_frame_path, out_name, mode)
        final_frames.append(merged_frame)
    return output_folder


def merge_difference(original_path: str, segmented_img_path: str, style_img_path: str, out_name: str, mode: int = 0):
    if style_img_path is None:
        style_img_path = segmented_img_path
    original = Image.open(original_path)
    o_pix = original.load()
    segmented = Image.open(segmented_img_path)
    custom = Image.open(style_img_path)
    c_pix = custom.load()
    assert original.size == segmented.size
    width, height = original.size

    for x in range(width):
        for y in range(height):
            # Change to >= if you want to get the foreground
            pixel_is_foreground = is_foreground(segmented.getpixel((x,y)))
            if mode == 0 and pixel_is_foreground:
                o_pix[x, y] = c_pix[x, y]
            elif mode == 1 and not pixel_is_foreground:
                o_pix[x, y] = c_pix[x, y]

    original.save(out_name)
    assert os.path.isfile(out_name)
    return out_name

def is_foreground(pixel):
    for mask_color in MASK_COLORS:
        if distance_fx(pixel, mask_color) <= MASK_DIFF_THRESHOLD:
            return True
    return False

def distance_fx(a, b):
    assert len(a) == len(b)
    dist = 0
    for i in range(len(a)):
        dist += abs(a[i] - b[i]) ** 2
    return dist ** 0.5
