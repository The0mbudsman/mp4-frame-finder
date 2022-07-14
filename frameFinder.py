# pylint: disable=C0301
"""This Script attempts to extract MP4/AVC iframes from a disk image. It does two(or three) passes on the input image
    first it goes through the disk looking for the SPS/PPS headers. These are stored in the MP4 avcc atom and define
    to the decoder how to decode a frame. It does a second pass looking for iframe headers and attempts to decode them
    and save them as jpegs to an output directory called FrameFinder_output"""

import argparse
import hashlib
import math
import os
import re
from glob import glob
# from multiprocessing import Pool, connection
 
import ffmpeg
from tqdm import tqdm

NALU_IDR_HEADER = b"\x65"  # Coded slice of an IDR picture
NALU_SPS_HEADER = b"\x67"  # Sequence Parameter Set
NALU_PPS_HEADER = b"\x68"  # Picture Parameter Set
AVCC_BOX_HEADER = b"\x61\x76\x63\x43"  # avcC atom header
OUTPUT_DIR_NAME = "FrameFinder_output"

parser = argparse.ArgumentParser(
    "frameCarver", description="Attempts to extract i-frames from h264 encoded videos in a disk image, for use when the filesystem information or file header is missing or corrupted")
parser.add_argument("imgfile", help="path to raw image file", type=str)
parser.add_argument("--cluster", dest="cluster_size_bits", type=int,
                    default=4096, help="cluster size (default: 4096)")
parser.add_argument("--unique", action="store_const", dest="unique",
                    const=True, default=False, 
                    help="Removes duplicate h264 headers before decoding found frames. Will speed up operation. (default: Do not remove defaults). When this option is not enabled, i-frames \
                    from the same video may exist in multiple output directories if two videos have the same headers")

parser.add_argument("--nocleanup", action="store_const", dest="nocleanup",
                    const=True, default=False, 
                    help="Attempts to remove false positive matches in the output directory. (default: Does not attempt a cleanup). There is a possibility that a small number \
                        of legitimate frames are removed in this process, but most if not all false positives will be removed ")
# Parse Arguments and set up some operational params
args = parser.parse_args()
clustersize = args.cluster_size_bits
unique = args.unique
nocleanup = args.nocleanup

# These hold partial and complete h264 headers.
partial_h264_headers = []
complete_h264_headers = []

# These hold partial and complete iframes headers.
potential_iframes = []
complete_iframes = []

# Output dir to hold frames.
# Make unique dir for each run if needed
original_name = OUTPUT_DIR_NAME
dir_num = 1
if not os.path.exists(f"./{OUTPUT_DIR_NAME}"):
    os.mkdir(f"./{OUTPUT_DIR_NAME}")
else:
    # Dir exists so make a new one
    while not os.path.exists(f"./{original_name}_{dir_num}"):   
        os.mkdir(f"./{original_name}_{dir_num}")
        dir_num = dir_num + 1 
    else:
        print("2")

exit(1)

#########################################################################################
# Set of methods which assist with Finding and extracting SPS and PPS headers on disk
#########################################################################################
def determine_SPS_size(match:int, bytes:bytes):
    """This function takes a group of bytes and an index of the '67', and works backwards to
    determine the size based on the preceeding bytes. The valid sizes are defined below, in bytes."""
    SPS_HEADER_MIN_SIZE = 10
    SPS_HEADER_MAX_SIZE = 30
    potential_byte_sizes = [1, 2, 4]  # byte size can be 1,2, or 4 bytes
    prev_size = 0
    SPS_header_size = 0
    for i in potential_byte_sizes:
        potential_byte_size = bytes[max(match-i,0):match]
        SPS_header_size = int.from_bytes(potential_byte_size, "big")
        if SPS_header_size == prev_size:  # skip if 2 bytes gives the same index as we've already found it most likely
            continue
        if (SPS_header_size < SPS_HEADER_MIN_SIZE or SPS_header_size > SPS_HEADER_MAX_SIZE):  # skip if length is ridiculous number
            continue
        prev_size = SPS_header_size
        return True, SPS_header_size
    return False, SPS_header_size


def determine_PPS_size(match:int, bytes:bytes):
    """This function takes a group of bytes and an index of the '68', and works backwards to
    determine the size based on the preceeding bytes. The valid sizes are defined below, in bytes."""
    PPS_HEADER_MIN_SIZE = 2
    PPS_HEADER_MAX_SIZE = 30
    potential_byte_sizes = [1, 2, 4]  # byte size can be 1,2, or 4 bytes
    prev_size = 0
    for i in potential_byte_sizes:
        potential_byte_size = bytes[max(match-i,0):match]
        PPS_header_size = int.from_bytes(potential_byte_size, "big")
        if PPS_header_size == prev_size:  # skip if 2 bytes gives the same index as we've already found it most likely
            continue
        if (PPS_header_size < PPS_HEADER_MIN_SIZE or PPS_header_size > PPS_HEADER_MAX_SIZE):  # skip if length is ridiculous number
            continue
        prev_size = PPS_header_size
        return True, PPS_header_size
    return False, PPS_header_size


def parse_full_SPS_PPS_header(bytes:bytes):
    """This function takes some bytes that are likely to contain a complete SPS/PPS
        header, and it extracts it and adds it to the global complete_h264_headers
        Call this when the input data is >50 bytes for safety."""
    # 61766343...........|00 01 67 .........|00 01 68..........|
    # |------------------|------------------|------------------|
    # |----AVCC header---|--SPS_header------|--PPS_header------|
    # |------------------|------------------|------------------|
    # |------------------|------------------|------------------|
    potential_SPS_indices = [_.start() for _ in re.finditer(NALU_SPS_HEADER, bytes)]
    if potential_SPS_indices == []:
        return
    else:
        for match in potential_SPS_indices:
            found, size = determine_SPS_size(match, bytes)
            if not found:
                continue
            else:
                SPS = bytes[match:match+size]
                bytes_after_SPS = bytes[match+size:]
                potential_PPS_indices = [_.start() for _ in re.finditer(NALU_PPS_HEADER, bytes_after_SPS)]
                if potential_SPS_indices == []:
                    return
                else:
                    for match in potential_PPS_indices:
                        found, size = determine_PPS_size(match, bytes_after_SPS)
                        if not found:
                            continue
                        else:
                            PPS = bytes_after_SPS[match:match+size+1]
                            complete_header = b"\x00\x00\x00\x01" + SPS + b"\x00\x00\x01" + PPS
                            complete_h264_headers.append({"data": complete_header, "saved_count": 0})


def register_partial_SPS_PPS_header(i:int, bytes:bytes):
    """This function is called when a SPS/PPS header is found but the cluster is ending, so the header will be cut off
    It is saved in a partial header list. If the SPS length is found in the cluster, the program knows roughly where the PPS should appear, so looks
    in next clusters at that index. If the SPS length isn't found, you're out of luck and it will generate a few false positives."""
    potential_SPS_index = bytes.find(NALU_SPS_HEADER)
    potential_PPS_index = bytes.find(NALU_PPS_HEADER)

    SPS_found = False if potential_SPS_index == -1 else True
    PPS_found = False if potential_PPS_index == -1 else True
    PPS_range = 0
    if SPS_found:
        found, size = determine_SPS_size(potential_SPS_index, bytes)
        PPS_range = list(range(1 + size - (len(bytes) - potential_SPS_index),
                         6 + size - (len(bytes) - potential_SPS_index)))
    partial_header = {
        "new_this_cluster": True,
        "found_in_cluster": i,
        "SPS_found": SPS_found,
        "PPS_found": PPS_found,
        "PPS_found_in_range": PPS_range,
        "partial_payload": bytes,
        "remaining_length": 50-len(bytes),
        "complete": False
    }
    partial_h264_headers.append(partial_header)


def parse_remainder_SPS_PPS_header(partial_header:dict, bytes:bytes):
    """This tries to parse the remainder of the header by wrapping the parse_full_SPS_PPS_header function"""
    len_before = len(complete_h264_headers)
    parse_full_SPS_PPS_header(bytes)
    if len(complete_h264_headers) != len_before:
        partial_header["complete"] = True

#########################################################################################
# Methods which assist with finding and decoding iframes
#########################################################################################


def get_iframes(cluster:bytes, match:int):
    """This parses the iframe start. It checks whether the I-frame size falls into limits defined below,
    similar to the SPS/PPS header size """
    MIN_IFRAME_SIZE = 8000 
    MAX_IFRAME_SIZE = 1510000
    potential_byte_size = cluster[match-4:match]
    iframe_size = int.from_bytes(potential_byte_size, "big")
    if (iframe_size > MIN_IFRAME_SIZE and iframe_size < MAX_IFRAME_SIZE):  # limits determined from tests
        length_of_data_to_append = min(iframe_size, (len(cluster)-match))
        # Here we define our frame object
        # total length = as determined above
        # remaining_length = The remaining number of bytes to append to it (this gets decremented)
        # new_this_cluster = Tells the next bit of code not to keep appending data this cluster.
        # data = this is the actual frame bytes
        frame = {
            "total_length": iframe_size,  # total length of the iframe
            "remaining_length": iframe_size-length_of_data_to_append,
            "new_this_cluster": True,
            "data": cluster[match:match+length_of_data_to_append]
        }
        potential_iframes.append(frame)


def decode_iframes(frames:dict):
    """This function calls ffmpeg to attempt to decode the frames in the global object. It
    brute forces each header with each newly completed frame, and saves them into a directory"""
    for i, header in enumerate(complete_h264_headers):
        saved_frame_count = header["saved_count"]
        for frame in frames:
            blob = header["data"] + b"\x00\x00\x01" + frame["data"]
            process = (ffmpeg
                       .input("pipe:")
                       .output(f"./{OUTPUT_DIR_NAME}/header_{i}/frame_{saved_frame_count}.jpg", loglevel="quiet")
                       .run_async(pipe_stdin=True)
                       )
            process.communicate(input=blob)
            if os.path.exists(f"./{OUTPUT_DIR_NAME}/header_{i}/frame_{saved_frame_count}.jpg"):
                header["saved_count"] = saved_frame_count + 1

    # Flush decoded frames
    global complete_iframes
    complete_iframes = []


#########################################################################################
# Step 1 happens here, scanning the input image for SPS/PPS headers
#########################################################################################
print(f"[Step 1/{2 if nocleanup else 3}] Finding SPS/PPS Headers...")
with open(args.imgfile, "rb") as f:
    for i, cluster in tqdm(enumerate(iter(lambda: f.read(clustersize), b'')), desc="Reading clusters", total=math.ceil(os.path.getsize(args.imgfile)/clustersize)):
        potential_AVCC_boxes = [_.start()
                                for _ in re.finditer(AVCC_BOX_HEADER, cluster)]
        for box_index in potential_AVCC_boxes:
            # These cases we dont have space for a full header, the match comes too late in the cluster
            if box_index > clustersize-50:
                register_partial_SPS_PPS_header(i, cluster[box_index:clustersize])
            else:
                parse_full_SPS_PPS_header(cluster[box_index:box_index+50])
        for partial_header in partial_h264_headers:
            # its already initialised with all the data it needs from this cluster
            if partial_header["new_this_cluster"]:
                # look at next cluster
                partial_header["new_this_cluster"] = False
            else:
                bytes_of_interest = cluster[0:partial_header["remaining_length"]]
                any_SPS_present = bytes_of_interest.find(NALU_SPS_HEADER)
                any_PPS_present = bytes_of_interest.find(NALU_PPS_HEADER)
                if (partial_header["SPS_found"] or any_SPS_present != -1) and (partial_header["PPS_found"] or any_PPS_present != -1):
                    # If the fragmentation point is between the SPS and PPS, we know roughly where the PPS header should be in a future cluster
                    if partial_header["SPS_found"]:
                        all_PPS_starts = [
                            _.start() for _ in re.finditer(NALU_PPS_HEADER, bytes_of_interest)]
                        for start in all_PPS_starts:
                            if start in partial_header["PPS_found_in_range"]:
                                parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)
                            else:
                                continue
                    else:
                        # no choice but to brute force
                        parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)

        # Transfer the headers which have found a valid end moved into the complete list and removed from the partial list
            partial_h264_headers[:] = [partial_header for partial_header in partial_h264_headers if partial_header["complete"] == False]
            complete_h264_headers[:] +=  [{"data":partial_header["data"], "saved_count": 0} for partial_header in partial_h264_headers if partial_header["complete"] == True]

#########################################################################################
# Step 2 sometimes happens here, If any cut-short SPS/PPS headers are found, then rescan the drive to see if their other halves are in some earlier cluster
#########################################################################################
if len(partial_h264_headers) > 0:
    print("===========================================")
    cluster_to_recheck_to = 0 
    for partial_header in partial_h264_headers:
        cluster_to_recheck_to = max(cluster_to_recheck_to, partial_header["found_in_cluster"])
    print(f"[Step 1.5/3] Unfinished SPS/PPS Headers found, re-checking image from start to cluster {cluster_to_recheck_to} for out of order fragmentation...")
    with open(args.imgfile, "rb") as f:
        for i, cluster in tqdm(enumerate(iter(lambda: f.read(clustersize), b'')), desc="Reading clusters ", total=cluster_to_recheck_to):
            if i > cluster_to_recheck_to + 1:
                break
            for partial_header in partial_h264_headers:
                bytes_of_interest = cluster[0:partial_header["remaining_length"]]
                any_SPS_present = bytes_of_interest.find(NALU_SPS_HEADER)
                any_PPS_present = bytes_of_interest.find(NALU_PPS_HEADER)
                if (partial_header["SPS_found"] or any_SPS_present != -1) and (partial_header["PPS_found"] or any_PPS_present != -1):
                    # If the fragmentation point is between the SPS and PPS, we know roughly where the PPS header should be in a future cluster
                    if partial_header["SPS_found"]:
                        all_PPS_starts = [_.start() for _ in re.finditer(NALU_PPS_HEADER, bytes_of_interest)]
                        for start in all_PPS_starts:
                            if start in partial_header["PPS_found_in_range"]:
                                parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)
                            else:
                                continue
                    else:
                        # no choice but to brute force
                        parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)

            # Transfer the headers which have found a valid end moved into the complete list and removed from the partial list
            partial_h264_headers[:] = [partial_header for partial_header in partial_h264_headers if not partial_header["complete"]]
            complete_h264_headers[:] += [{"data": partial_header["data"], "saved_count": 0} for partial_header in partial_h264_headers if partial_header["complete"]]

#########################################################################################
# Step 2.5 is to make output directories for each of the headers and optionally exit if no headers are found.
#########################################################################################
if len(complete_h264_headers) == 0:
    print("No headers found in image, exiting.")
    exit(1)

if unique:
    # https://stackoverflow.com/questions/11092511/list-of-unique-dictionaries
    lenbefore = len(complete_h264_headers)
    complete_h264_headers = [dict(s) for s in set(frozenset(header.items()) for header in complete_h264_headers)]
    print(f"Successfully found {lenbefore} SPS/PPS headers, removed {lenbefore - len(complete_h264_headers) } duplicate(s)")
else:
    print(f"Successfully found {len(complete_h264_headers)} SPS/PPS headers")


for i in range(len(complete_h264_headers)):
    # just in case that folder got deleted during runtime
    if not os.path.exists(f"./{OUTPUT_DIR_NAME}"):
        os.mkdir(f"./{OUTPUT_DIR_NAME}")
    if not os.path.exists(f"./{OUTPUT_DIR_NAME}/header_{i}"):
        os.mkdir(f"./{OUTPUT_DIR_NAME}/header_{i}")


#########################################################################################
# Step 3 is finally to look for iframes on the drive, and attempt to decode them on the fly with any of the SPS/PPS headers found
#########################################################################################
with open(args.imgfile, "rb") as f:
    print("===========================================")
    print(f"[Step 2/{2 if nocleanup else 3}] Finding IDR Frames that could fit SPS/PPS headers...")
    for i, cluster in tqdm(enumerate(iter(lambda: f.read(clustersize), b'')), desc="Reading clusters", total=math.ceil(os.path.getsize(args.imgfile)/clustersize)):
        # check the cluster for the potential start of an IDR frame
        potential_iframe_starts = [_.start() for _ in re.finditer(NALU_IDR_HEADER, cluster)]
        for match in potential_iframe_starts:
            # finds iframes and appends them to global list(if any, after checking size)
            get_iframes(cluster, match)
        for partial_frame in potential_iframes:
            if (partial_frame["new_this_cluster"]): #its already initialised with all the data it needs from this cluster
                partial_frame["new_this_cluster"] = False
            else:
                match_end_index = min(partial_frame["remaining_length"], (clustersize))
                partial_frame["data"] = partial_frame["data"] + cluster[0:match_end_index]
                partial_frame["remaining_length"] = partial_frame["remaining_length"] - match_end_index

        # Transfer the frames which have 0 remaining length to be appended to the "complete_iframes" list
        # Also remove them from the "potential_iframes" list
        complete_iframes[:] += [partial_frame for partial_frame in potential_iframes if partial_frame["remaining_length"] == 0 ]
        potential_iframes[:] =  [partial_frame for partial_frame in potential_iframes if partial_frame["remaining_length"] != 0 ]
        if len(complete_iframes) > 0:
            decode_iframes(complete_iframes)

#########################################################################################
# Step 4 is to cleanup the output
#########################################################################################

if nocleanup:
    print("Skipping cleanup as requested")
    print("Finished!")
    exit(1)
else:
    print("===========================================")
    print("[Step 3/3] Cleaning up output folders")
    cleanup = 0
    for j in tqdm(range(len(complete_h264_headers)), desc="Cleaning up output folders", total=len(complete_h264_headers)):
        size = 0
        for i,file in enumerate(glob(f"./{OUTPUT_DIR_NAME}/header_{j}/*.jpg")):
            size += os.path.getsize(file)
        av_size = size/i
        for file in glob(f"./{OUTPUT_DIR_NAME}/header_{j}/*.jpg"):
            if os.path.getsize(file) < av_size:
                os.remove(file)
                cleanup = cleanup + 1
    print(f"Removed {cleanup} images in the cleanup process")
    print("Finished!")
    exit(1)
                

