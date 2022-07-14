<h3>About</h3>
This Script attempts to extract MP4/AVC/h.264 iframes from a disk image. It does two(or three) passes on the input image
first it goes through the disk looking for the SPS/PPS headers. These are stored in the MP4 avcc atom and define
to the decoder how to decode a frame. It does a second pass looking for iframe headers and attempts to decode them
and save them as jpegs to an output directory called FrameFinder_output

For a quick proof of concept you can pass it a complete video file and it will extract iframes to jpegs

<h3>Requirements</h3>
Currently only tested on python 3.9 and Linux
Requires FFmpeg to be installed, only tested with version 4.4.2-1+b1

<h3>Usage</h3>
python3 frameFinder.py <path_to_file> 
python3 frameFinder.py -h  for help
