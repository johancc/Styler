## Styler
Applies style transfer on a video. It supports segmented style transfer,
meaning only the foreground or background of the video are styled.

Example Usage:

`python3 styler.py --video videos/video.mp4 --style_model models/mosaic.pth`

### Requirements
Command line tools:
- ffmpeg

Python packages: requirements.txt

### TODO:
- Add segmentation functionality
- Implement the style & segmentation merge functionality
