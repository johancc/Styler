"""
Runs segmentation on a batch of frames. Wrapper for the CDCL-human-part-segmentation.
"""
import os
import subprocess

from PIL import Image

# The segmented frame has a dark mask over the whole image, and the
# human sections have are colored. Thus, we cannot use a simple bitwise_and
# to create the final segmented image. We thus need to look at how different
# the pixels are. The threshold is the max rgb difference from the original
# and segmented image.
THRESHOLD = 155


def segment(frames_folder: str, output_folder):
    for f in [frames_folder, output_folder]:
        if not os.path.exists(f):
            os.makedirs(f)

    command = "./run_segmentation.sh {} {}".format(frames_folder, output_folder)
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
    original_frames = sorted(os.listdir(original_folder))
    styled_frames = sorted(os.listdir(style_folder))
    segmented_frames = sorted(os.listdir(segmentation_folder))

    final_frames = []
    for i in range(len(original_folder)):
        original_frame_path = os.path.join(original_folder, original_frames[i])
        change_frame_path = os.path.join(style_folder, styled_frames[i])
        segmented_frame_path = os.path.join(segmentation_folder, segmented_frames[i])
        out_name = os.path.join(output_folder, original_frames[i])
        merged_frame = merge_difference(original_frame_path, segmented_frame_path, change_frame_path, out_name, mode)
        final_frames.append(merged_frame)
    return output_folder


def merge_difference(original_path: str, change_path: str, custom_replacement: str, out_name: str, mode: int = 0):
    if custom_replacement is None:
        custom_replacement = change_path
    original = Image.open(original_path)
    o_pix = original.load()
    change = Image.open(change_path)
    custom = Image.open(custom_replacement)
    c_pix = custom.load()
    assert original.size == change.size
    width, height = original.size

    for x in range(width):
        for y in range(height):
            # Change to >= if you want to get the foreground
            dist = distance_fx(original.getpixel((x, y)), (0, 0, 0))
            is_foreground = dist < THRESHOLD
            is_background = dist >= THRESHOLD
            if mode == 0 and is_foreground:
                o_pix[x, y] = c_pix[x, y]
            elif mode == 1 and is_background:
                o_pix[x, y] = c_pix[x, y]

    original.save(out_name)
    assert os.path.isfile(out_name)
    return out_name


def distance_fx(a, b):
    assert len(a) == len(b)
    dist = 0
    for i in range(len(a)):
        dist += abs(a[i] - b[i]) ** 2
    return dist ** 0.5
