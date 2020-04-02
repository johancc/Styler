"""
This is a command line tool that styles a video using style transfer. The user can specify if the background
or humans in the video should be styled.
To create a styled video, we use a pre-trained model based on the Fast Neural Style Transfer model
(https://github.com/eriklindernoren/Fast-Neural-Style-Transfer).
The style can be applied to only segments of the video. Currently, we can differentiate the background
from the people in videos using Human Part Segmentation
(https://github.com/kevinlin311tw/CDCL-human-part-segmentation).
- Johan Cervantes, March 2020.
"""
import argparse
from typing import Tuple

from torchvision.utils import save_image

from segmentation import segment, apply_style_over_segmentation
from styler_lib import *

parser = argparse.ArgumentParser()
parser.add_argument("--source_frames", metavar="--v", type=str, help="Path to the video frames.")
parser.add_argument("--video",  type=str, help="Path to the video")
parser.add_argument("--output_name", type=str, help="Name of output file")
parser.add_argument("--styled_frames",type=str, help="Path to style frames")
parser.add_argument("--segmented_frames", type=str, help="Path to segmented frames")

parser.add_argument("--background", default=False, type=bool, help="Apply the style only to the background (no-humans)")
parser.add_argument("--foreground", default=False, type=bool, help="Apply the style only to the foreground (humans).")
parser.add_argument("--frame_rate", default=30, type=int, help="Frame rate of the video (normally 24 or 30)."
                                                              " Defaults to 30.")
parser.add_argument("--gpus", default=1, type=int, help="Number of gpus to use during segmentation.")
# TODO: Add the option to add the source directory for frames_path (no need to split the frames_path if already split)
# TODO: Add automatic frame rate detection.


def validate_arguments(args) -> Tuple[str, bool]:
    message = ""
    err = False
    if args.background and args.foreground:
        message += "Background and foreground cannot be both true. " \
                   "If the whole video should be styled, specify neither.\n"
        err = True
    if not os.path.exists(args.video):
        message += "Invalid video path.\n"
        err = True
    if not os.path.exists(args.style_model):
        message += "Invalid model path.\n"
        err = True
    return message, err


def main(video_path, model_path, background, foreground, output_path, frame_rate, keep_temp=False, gpus=1):
    video_name = get_base_name(video_path)
    style_name = get_base_name(model_path)
    audio_file = create_audio_file(video_path)
    # Get all the frames_path
    print("Extracting frames...")
    frame_dir = extract_video_frames(video_path)
    print("Styling frames...")
    style_dir = style_frames(model_path, frame_dir, style_dir="{}_styled/".format(video_name))
    if background or foreground:
        print("Applying segmentation...(will take a while (~10 min per 500 frames)")
        segmentation_dir = segment(video_name, frame_dir, "{}_segmented/".format(video_name), gpus)
        print("Merging styled and segmented frames...")
        segmented_styled_frames = apply_style_over_segmentation(original_folder=frame_dir,
                                                                style_folder=style_dir,
                                                                segmentation_folder=segmentation_dir,
                                                                output_folder="{}_final".format(video_name),
                                                                mode=0 if foreground else 1)
        styled_video = frames_to_video(frames_path=segmented_styled_frames,
                                       video_name=video_name,
                                       output_path=output_path,
                                       frame_rate=frame_rate)
    else:
        styled_video = frames_to_video(frames_path=style_dir,
                                       video_name=video_name,
                                       output_path=output_path,
                                       frame_rate=frame_rate)
    final_video_name = os.path.join(output_path, video_name + "_{}_style.mp4".format(style_name))
    add_audio_to_video(styled_video, audio_file, final_video_name)
    assert (os.path.isfile(final_video_name))
    if not keep_temp:
        cleanup_files = {
            "directories": [frame_dir, style_dir],
            "files": [styled_video, audio_file]
        }
        cleanup_temp_files(cleanup_files)

    return final_video_name


def style_frames(checkpoint_path: str, frame_dir: str, style_dir: str):
    if not os.path.exists(style_dir):
        os.makedirs(style_dir)
    # Load the video
    preprocessor = get_preprocessor()
    transformer = get_transformer(checkpoint_path)
    source_frames = sorted(os.listdir(frame_dir))
    styled_frames = []
    with torch.no_grad():
        for frame in tqdm.tqdm(source_frames):
            frame_path = os.path.join(frame_dir, frame)
            styled_image = denormalize(transformer(preprocessor(frame_path))).cpu()
            styled_frames.append(styled_image)
    print("Saving images...")
    for i, frame in tqdm.tqdm(enumerate(styled_frames)):
        i = str(i).zfill(4)
        save_image(frame, os.path.join(style_dir, "{}.jpg".format(i)))

    return style_dir

def merge(video_name, source, styled, segmented, foreground = True):
    segmented_styled_frames = apply_style_over_segmentation(original_folder=source,
                                                                style_folder=styled,
                                                                segmentation_folder=segmented,
                                                                output_folder="{}_final".format(video_name),
                                                                mode=0 if foreground else 1)
    styled_video = frames_to_video(frames_path=segmented_styled_frames,
                                    video_name=video_name,
                                    output_path=args.output_path,
                                    frame_rate=args.frame_rate)
    audio_file = create_audio_file(args.video)
    final_video_name = os.path.join(args.output_path, video_name + "_merged.mp4")
    add_audio_to_video(styled_video, audio_file, final_video_name)
    return final_video_name

if __name__ == '__main__':
    args = parser.parse_args()
    message, err = validate_arguments(args)
    if err:
        print(message)
        exit(1)
    
    vid = merge(get_base_name(args.video), args.source_frames, args.styled_frames, args.segmented_frames, args.foreground)
    print("Merged frames into: ", vid)
