## Styler
Applies style transfer on a video. It supports segmented style transfer,
meaning only the foreground or background of the video are styled.

Example Usage:

`python3 styler.py --video videos/video.mp4 --style_model models/mosaic.pth`

### Requirements
- ffmpeg
- Anaconda

## Installation
To install, simply run install.sh. If the script fails to create a conda environment
create one as follows:

1) ``` conda env create -f environment.yaml ```
2) ``` conda activate styler  ```


## Segmentation
Segmentation is done by CDCL Human Part Segmentation. Styler merely 
feeds the video frames into the segmentation library and uses its
output to style the foreground or background. 

### TODO:
- Flow-based styling might yield more interesting videos.