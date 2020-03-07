"""
This library provides a set of functions needed to perform style transfer.
"""
import ntpath
import os
import shutil
import subprocess
from math import ceil, log10
from typing import Callable

import numpy as np
import skvideo.io
import torch
import tqdm
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from models import TransformerNet

# Constants
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# Image processing utilities

# PyTorch constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocessor(image_size=None) -> Callable:
    resize = [transforms.Resize(image_size)] if image_size else []
    processing_pipeline = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def preprocessor(image_path: str):
        image_tensor = Variable(processing_pipeline(Image.open(image_path))).to(device)
        return image_tensor.unsqueeze(0)

    return preprocessor


def get_transformer(checkpoint_path: str) -> torch.nn.Module:
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_path, map_location=device))
    transformer.eval()
    return transformer


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(STD[c]).add_(STD[c])
    return tensors


def zero_padding(i: int, max_count: int):
    padding = ceil(log10(max_count))

    return str(i).zfill(padding)


def deprocess(image_tensor):
    """ Denormalizes and rescales image tensor """
    image_tensor = denormalize(image_tensor)[0]
    image_tensor *= 255
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np


def extract_video_frames(video_path: str) -> str:
    # Create output dir
    video_name = get_base_name(video_path)
    source_frame_path = os.path.join(os.curdir, "{}_frames/".format(video_name))
    if not os.path.exists(source_frame_path):
        os.makedirs(source_frame_path)

    # FFmpeg extracts all the frames_path
    command = "ffmpeg -i {} {}%04d.jpg".format(video_path, source_frame_path)
    p = subprocess.Popen(command.split(" "),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()
    return source_frame_path


def get_base_name(video_path: str) -> str:
    return str(ntpath.basename(video_path).split(".")[0])


def create_audio_file(video_path: str) -> str:
    audio_name = video_path.split(".")[0] + ".aac"
    command = "ffmpeg -i {} -vn {}".format(video_path, audio_name)
    p = subprocess.Popen(command.split(" "),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()
    assert os.path.isfile(audio_name)
    return audio_name


def frames_to_video(frames_path: str, frame_rate: int, video_name: str, output_path: str):
    output_video_path = os.path.join(output_path, video_name + ".mp4")
    writer = skvideo.io.FFmpegWriter(output_video_path,
                                     inputdict={"-r": str(frame_rate)},
                                     outputdict={"-pix_fmt": "yuv420p",
                                                 "-r": str(frame_rate)})

    frames = sorted(os.listdir(frames_path))
    for frame in tqdm.tqdm(frames, desc="Writing video"):
        writer.writeFrame(np.array(Image.open(os.path.join(frames_path, frame))))
    writer.close()
    return output_video_path


def add_audio_to_video(video_path: str, audio_path: str, output_name: str) -> str:
    command = "ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental {}" \
        .format(video_path, audio_path, output_name)

    p = subprocess.Popen(command.split(" "),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()
    assert os.path.isfile(output_name)
    return output_name


def cleanup_temp_files(files_to_cleanup: dict):
    directories = files_to_cleanup.get("directories")
    files = files_to_cleanup.get("files")
    for d in directories:
        shutil.rmtree(d)
    for file in files:
        os.remove(file)
