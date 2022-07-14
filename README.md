# mp4-frame-finder
This Script attempts to extract MP4/AVC/h.264 iframes from a disk image. It does two(or three) passes on the input image
first it goes through the disk looking for the SPS/PPS headers. These are stored in the MP4 avcc atom and define
to the decoder how to decode a frame. It does a second pass looking for iframe headers and attempts to decode them
and save them as jpegs to an output directory called FrameFinder_output

It is a work in progress, only tested on python 3.9 and Kali linux 

For a quick proof of concept you can pass it a complete video file and it will extract iframes to jpegs
