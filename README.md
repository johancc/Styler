## Styler
Applies style transfer on a video. It supports segmented style transfer,
meaning only the foreground or background of the video are styled.

Example Usage:

`python3 styler.py --video videos/video.mp4 --style_model models/mosaic.pth`

### Requirements
Command line tools:
- ffmpeg

Python packages: requirements.txt

## Segmentation
Segmentation is done by CDCL Human Part Segmentation. Styler merely 
feeds the video frames into the segmentation library and uses its
output to style the foreground or background. 


### TODO:
- Implement training 
- Implement the style & segmentation merge functionality
- Flow-based styling might yield more interesting videos.