<h3>About</h3>
This is a forensic tool which attempts to extract MP4/AVCC iframes from a disk image. It does two(or three) passes on the input image

First it goes through the disk looking for the SPS/PPS headers. These are stored in the MP4 avcc atom and define
to the decoder how to decode a frame. 

Secondly it looks for iframe headers and attempts to decode them
and save them as jpegs to an output directory called FrameFinder_output

For a quick proof of concept you can pass it a complete video file in MP4/AVCC format and it will extract iframes


The motivation behind this tool is that extraction and reassembly of many fragments from a large video file is a computationally costly process, but a single i-frame may hold the information you need from the video

<h3>Requirements</h3>
Use Python 3.9 or above. Python 3.8 was found to be about half the performance of Python 9+

Requires FFmpeg to be installed, only tested with version 4.4.2-1+b1

run pip install -r /path/to/requirements.txt


<h3>Usage</h3>

```
python3 frameFinder.py (path_to_file)
python3 frameFinder.py -h  for help
```

<h3>Usage in other scripts</h3>
Check testharness.py

```
from frameFinder import frameFinder
ff = frameFinder(clustersize=4096, output_dir_name=f"{basedir}/{file_name_only}", nocleanup=True)
ff.process_image(file)
```

