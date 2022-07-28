# pylint: disable=C0301
"""This Script attempts to extract MP4/AVC iframes from a disk image. It does two(or three) passes on the input image
    first it goes through the disk looking for the SPS/PPS headers. These are stored in the MP4 avcc atom and define
    to the decoder how to decode a frame. It does a second pass looking for iframe headers and attempts to decode them
    and save them as jpegs to an output directory called FrameFinder_output"""

import argparse
import math
import os
import re
from hashlib import md5
from itertools import repeat
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, cpu_count

import ffmpeg
from bitstring import BitArray


class frameFinder:
    def __init__(self, clustersize:int=4096, nocleanup:bool=False, unique:bool=False, output_dir_name:str="FrameFinder_output", decoding_buffer:int=49):
        """This is the main function that does all the work.
        Arguments: 
        @clustersize: Size of a cluster to iterate through (optional, default: 4096)
        @nocleanup: Switch to stop clean up of the output folders. There is a possibility that a small number \
            of legitimate frames are removed in this process, but most if not all false positives will be removed (optional, default: Cleanup is performed)
        @unique: Removes duplicate h264 headers before decoding found frames. Will speed up operation. When this option is not enabled, i-frames \
            from the same video may exist in multiple output directories if two videos have the same headers" (optional, default: Do not remove defaults).
        @output_dir_name: Where to dump the output, base name, after multiple runs, numbers will be appended to this directory name. (optional, default: 'FrameFinder_output')
        @decoding_buffer: May provide a small speed up increasing this above 49, but it is probably at the cost of performance.
        """
        # Initialising some useful variables

        self.NALU_IDR_HEADER = b"\x65"  # Coded slice of an IDR picture
        self.NALU_SPS_HEADER = b"\x67"  # Sequence Parameter Set
        self.NALU_PPS_HEADER = b"\x68"  # Picture Parameter Set
        self.AVCC_BOX_HEADER = b"\x61\x76\x63\x43"  # avcC atom header
        self.OUTPUT_DIR_NAME = output_dir_name

        #Operational params 
        self.clustersize = clustersize
        self.nocleanup = nocleanup
        self.unique = unique
        self.decoding_buffer = decoding_buffer

        #This'll be used to check if the cluster is all zeros (so skipped)
        self.null_block = b'\x00' * clustersize

        # These hold partial and complete h264 headers.
        self.partial_h264_headers = []
        self.complete_h264_headers = []

        # These hold partial and complete iframes headers.
        self.potential_iframes = []
        self.complete_iframes = []

        # Multithread setup
        self.NUM_PROCS = cpu_count()-1 or 1

        # Output dir to hold frames.
        # Make unique dir for each run if needed
        self.original_name = self.OUTPUT_DIR_NAME
        dir_num = 1

        if not os.path.exists(f"./{self.OUTPUT_DIR_NAME}"):
            os.makedirs(f"./{self.OUTPUT_DIR_NAME}", exist_ok=True)
        else:
            # Dir exists so make a new one
            while os.path.exists(f"./{self.original_name}_{dir_num}"):   
                dir_num = dir_num + 1 
            else:
                os.makedirs(f"./{self.original_name}_{dir_num}", exist_ok=True)
                self.OUTPUT_DIR_NAME = f"{self.original_name}_{dir_num}"

    #########################################################################################
    # Set of functions which assist with Finding and extracting SPS and PPS headers on disk, finding i_frames, and decoding the iframe
    #########################################################################################
    def determine_avcC_size(self, match:int, bytes:bytes):
        """This function takes a group of bytes and an index of the 'avcC', and works backwards to
        determine the size based on the preceeding bytes. The valid sizes are defined below, in bytes."""
        AVCC_HEADER_MIN_SIZE = 10
        AVCC_HEADER_MAX_SIZE = 100
        potential_byte_sizes = [1, 2, 4]  # byte size can be 1,2, or 4 bytes
        prev_size = 0
        AVCC_header_size = 0
        for i in potential_byte_sizes:
            potential_byte_size = bytes[max(match-i,0):match]
            AVCC_header_size = int.from_bytes(potential_byte_size, "big")
            if AVCC_header_size == prev_size:  # skip if 2 bytes gives the same index as we've already found it most likely
                continue
            if (AVCC_header_size < AVCC_HEADER_MIN_SIZE or AVCC_header_size > AVCC_HEADER_MAX_SIZE):  # skip if length is ridiculous number
                continue
            prev_size = AVCC_header_size
            return True, AVCC_header_size
        return False, AVCC_header_size
    
    def determine_SPS_size(self, match:int, bytes:bytes):
        """This function takes a group of bytes and an index of the '67', and works backwards to
        determine the size based on the preceeding bytes. The valid sizes are defined below, in bytes."""
        SPS_HEADER_MIN_SIZE = 2
        SPS_HEADER_MAX_SIZE = 60
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

    def determine_PPS_size(self, match:int, bytes:bytes):
        """This function takes a group of bytes and an index of the '68', and works backwards to
        determine the size based on the preceeding bytes. The valid sizes are defined below, in bytes."""
        PPS_HEADER_MIN_SIZE = 2
        PPS_HEADER_MAX_SIZE = 50
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

    def parse_full_SPS_PPS_header(self, bytes:bytes):
        """This function takes some bytes that are likely to contain a complete SPS/PPS
            header, and it extracts it and adds it to the global complete_h264_headers
            Call this when the input data is >50 bytes for safety."""
        # 61766343...........|00 01 67 .........|00 01 68..........|
        # |------------------|------------------|------------------|
        # |----AVCC header---|--SPS_header------|--PPS_header------|
        # |------------------|------------------|------------------|
        # |------------------|------------------|------------------|
        potential_SPS_indices = [_.start() for _ in re.finditer(self.NALU_SPS_HEADER, bytes)]
        if potential_SPS_indices == []:
            return
        else:
            for match in potential_SPS_indices:
                found, size = self.determine_SPS_size(match, bytes)
                if not found:
                    continue
                else:
                    SPS = bytes[match:match+size+1]
                    bytes_after_SPS = bytes[match+size:]
                    potential_PPS_indices = [_.start() for _ in re.finditer(self.NALU_PPS_HEADER, bytes_after_SPS)]
                    if potential_SPS_indices == []:
                        return
                    else:
                        for match in potential_PPS_indices:
                            found, size = self.determine_PPS_size(match, bytes_after_SPS)
                            if not found:
                                continue
                            else:
                                PPS = bytes_after_SPS[match:match+size+1]
                                complete_header = b"\x00\x00\x01" + SPS + b"\x00\x00\x01" + PPS
                                self.complete_h264_headers.append({"data": complete_header, "saved_count": 0, "SPS": SPS, "PPS": PPS})

    def register_partial_SPS_PPS_header(self, i:int, bytes:bytes):
        """This function is called when a SPS/PPS header is found but the cluster is ending, so the header will be cut off.
        This function attempts to assess how much of the header has been found in order to search for it later
        It is saved in a partial header list. If the SPS length is found in the cluster, the program knows roughly where the PPS should appear, so looks
        in next clusters at that index. If the SPS length isn't found, you're out of luck and it will generate a few false positives."""
        potential_SPS_index = bytes.find(self.NALU_SPS_HEADER)
        potential_PPS_index = bytes.find(self.NALU_PPS_HEADER)
        potential_avcc_index = bytes.find(self.AVCC_BOX_HEADER)

        SPS_start_found = False if potential_SPS_index == -1 else True
        PPS_start_found = False if potential_PPS_index == -1 else True

        found,total_atom_size = self.determine_avcC_size(potential_avcc_index, bytes)
        complete_SPS_found = False
        complete_PPS_found = False
        remaining_length = 0

        if (SPS_start_found):
            found, SPS_size = self.determine_SPS_size(potential_SPS_index, bytes)
            if found:
                if potential_SPS_index+SPS_size < len(bytes):
                    complete_SPS_found = True

        if (PPS_start_found):
            found, PPS_size = self.determine_PPS_size(potential_PPS_index, bytes)
            if found:
                if potential_PPS_index+PPS_size < len(bytes):
                    complete_PPS_found = True

        if (complete_SPS_found and complete_PPS_found):
            # Best case, we actually have the lot
            self.parse_full_SPS_PPS_header(bytes)
            return
        remaining_length = total_atom_size+1 - len(bytes)
        partial_header = {
            "new_this_cluster": True,
            "found_in_cluster": i,
            "SPS_start_found": SPS_start_found,
            "complete_SPS_found": complete_SPS_found,
            "PPS_start_found": PPS_start_found,
            "complete_PPS_found": complete_PPS_found,
            "partial_payload": bytes,
            "remaining_length": remaining_length,
            "complete": False
        }
        self.partial_h264_headers.append(partial_header)

    def parse_remainder_SPS_PPS_header(self, partial_header:dict, bytes:bytes):
        """This tries to parse the remainder of the header by wrapping the parse_full_SPS_PPS_header function"""
        len_before = len(self.complete_h264_headers)
        self.parse_full_SPS_PPS_header(bytes)
        if len(self.complete_h264_headers) != len_before:
            partial_header["complete"] = True

    def get_iframes(self, i, cluster:bytes, match:int):
        """This parses the iframe start. It checks whether the I-frame size falls into limits defined below,
        similar to the SPS/PPS header size """
        MIN_IFRAME_SIZE = 100
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
            
            self.potential_iframes.append(frame)

    def decode_iframes(self, NUM_PROCS):
        """This function calls ffmpeg to attempt to decode the frames in the global object. It
        brute forces each header with each newly completed frame, and saves them into a directory"""       
        process_size = min(NUM_PROCS, len(self.complete_h264_headers))
        pool = Pool(processes=process_size)
        results = pool.map(self.decode_frames_with_header_worker, zip(enumerate(self.complete_h264_headers), repeat(self.complete_iframes))) ##############
        for i,result in enumerate(results):
            self.complete_h264_headers[i]["saved_count"] = result

    def decode_frames_with_header_worker(self, data_zip:list):
        '''This helps with multipool'''
        headers_tup, complete_iframes = data_zip
        i, header = headers_tup
        blob = b"\x00" # starting delimiter has an additional 0 byte.
        ## assemble an annex B compliant blob of SPS, PPS, IDR FRAME, SPS, PPS, IDR FRAME, ETC....
        for frame in complete_iframes:
            mb_in_slice, slice_type = self.parse_exp_golomb(frame["data"][1:30],2)
            if (mb_in_slice == 0):
                blob += header["data"] + b"\x00\x00\x01" + frame["data"]
            else:
                blob += b"\x00\x00\x01" + frame["data"]
        with open (f"./{self.OUTPUT_DIR_NAME}/header_{i}/blob.h264", "wb") as f:
            f.write(blob)
        while not os.access(f"./{self.OUTPUT_DIR_NAME}/header_{i}/blob.h264", os.R_OK):
            pass
        process = (ffmpeg
                .input(f"./{self.OUTPUT_DIR_NAME}/header_{i}/blob.h264", f="h264", r="3")
                .filter("setpts", "N")
                .output(f"./{self.OUTPUT_DIR_NAME}/header_{i}/frame%5d.jpg", start_number=(header["saved_count"]+1), vsync="0", vcodec="mjpeg", loglevel="quiet")
                .run_async()
                )
        files_saved = sorted(glob(f"./{self.OUTPUT_DIR_NAME}/header_{i}/*.jpg"))
        if (len(files_saved) == 0 ):
            return 0
        tokens = files_saved[len(files_saved)-1].split("/")
        last_file = int(tokens[len(tokens)-1].split(".")[0].split("frame")[1])
        return last_file

    def parse_exp_golomb(self,payload:bytes, numberToParse:int):
        c = BitArray(hex=payload.hex()).bin
        codes = []
        for i in range(numberToParse):
            leadingZeroBits = -1
            for digit in c :
                if digit == "0":
                    leadingZeroBits+=1
                else: # up to and including 1
                    leadingZeroBits+=1
                    break
            if (leadingZeroBits != 0 ):
                payload = int(c[leadingZeroBits+1:leadingZeroBits+1+leadingZeroBits],2)
            else:
                payload = 0
            codes.append((2**leadingZeroBits)-1 + payload)
            c = c[leadingZeroBits+1+leadingZeroBits:]
        if len(codes) == 1:
            return codes[0]
        else:
            return(codes)

    # def entropy_of_cluster(self,data):
    #     p_data = data.value_counts()           # counts occurrence of each value
    #     entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    #     return entropy

    def file_as_bytes(self,file):
        with file:
            return file.read()

    def process_image(self,filepath):
        """This is the main function that does all the work.
        Arguments: 
        @filepath: Path to a file to analyse (required)
        """
        clustersize = self.clustersize
        nocleanup = self.nocleanup
        unique = self.unique
        #########################################################################################
        # Step 1 happens here, scanning the input image for SPS/PPS headers
        #########################################################################################
        print(f"[Step 1/{2 if nocleanup else 3}] Finding SPS/PPS Headers...")
        with open(filepath, "rb") as f:
            for i, cluster in tqdm(enumerate(iter(lambda: f.read(clustersize), b'')), desc="Reading clusters", total=math.ceil(os.path.getsize(filepath)/clustersize)):
                potential_AVCC_boxes = [_.start()
                                        for _ in re.finditer(self.AVCC_BOX_HEADER, cluster)]
                for box_index in potential_AVCC_boxes:

                    # These cases we dont have space for a full header, the match comes too late in the cluster
                    if (box_index > clustersize-100):
                        start = (max(box_index-4, 0))
                        self.register_partial_SPS_PPS_header(i, cluster[start:clustersize])
                    else:
                        start = (max(box_index-4, 0))
                        self.parse_full_SPS_PPS_header(cluster[start:box_index+100])
                for partial_header in self.partial_h264_headers:
                    # its already initialised with all the data it needs from this cluster
                    if partial_header["new_this_cluster"]:
                        # look at next cluster
                        partial_header["new_this_cluster"] = False
                    else:
                        bytes_of_interest = cluster[0:partial_header["remaining_length"]]
                        if (partial_header["SPS_start_found"] and partial_header["complete_SPS_found"] and partial_header["PPS_start_found"]):
                            self.parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)
                        else:
                            if (bytes_of_interest.find(self.NALU_PPS_HEADER) != -1):
                                self.parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)

                # Transfer the headers which have found a valid end moved into the complete list and removed from the partial list
                self.partial_h264_headers[:] = [partial_header for partial_header in self.partial_h264_headers if partial_header["complete"] == False]
        #########################################################################################
        # Step 2 sometimes happens here, If any cut-short SPS/PPS headers are found, then rescan the drive to see if their other halves are in some earlier cluster
        #########################################################################################
        if len(self.partial_h264_headers) > 0:
            print("===========================================")
            cluster_to_recheck_to = 0 
            for partial_header in self.partial_h264_headers:
                cluster_to_recheck_to = max(cluster_to_recheck_to, partial_header["found_in_cluster"])
            print(f"[Step 1.5/3] Unfinished SPS/PPS Headers found, re-checking image from start to cluster {cluster_to_recheck_to} for out of order fragmentation...")
            with open(filepath, "rb") as f:
                for i, cluster in tqdm(enumerate(iter(lambda: f.read(clustersize), b'')), desc="Reading clusters ", total=cluster_to_recheck_to):
                    if i > cluster_to_recheck_to + 1:
                        break
                    for partial_header in self.partial_h264_headers:
                        # its already initialised with all the data it needs from this cluster
                        if partial_header["new_this_cluster"]:
                            # look at next cluster
                            partial_header["new_this_cluster"] = False
                        else:
                            bytes_of_interest = cluster[0:partial_header["remaining_length"]]
                            if (partial_header["SPS_start_found"] and partial_header["complete_SPS_found"] and partial_header["PPS_start_found"]):
                                self.parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)
                            else:
                                if (bytes_of_interest.find(self.NALU_PPS_HEADER) != -1):
                                    self.parse_remainder_SPS_PPS_header(partial_header, partial_header["partial_payload"]+bytes_of_interest)
                                    
                    # Transfer the headers which have found a valid end moved into the complete list and removed from the partial list
                    self.partial_h264_headers[:] = [partial_header for partial_header in self.partial_h264_headers if partial_header["complete"] == False]
        #########################################################################################
        # Step 2.5 is to make output directories for each of the headers and optionally exit if no headers are found.
        #########################################################################################
        if len(self.complete_h264_headers) == 0:
            print("No headers found in image, exiting.")
            if __name__ == "__main__":
                exit(1)
            else:
                return

        if unique:
            # https://stackoverflow.com/questions/11092511/list-of-unique-dictionaries
            lenbefore = len(self.complete_h264_headers)
            self.complete_h264_headers = [dict(s) for s in set(frozenset(header.items()) for header in self.complete_h264_headers)]
            print(f"Successfully found {lenbefore} SPS/PPS headers, removed {lenbefore - len(self.complete_h264_headers) } duplicate(s)")
        else:
            print(f"Successfully found {len(self.complete_h264_headers)} SPS/PPS headers")
        for i in range(len(self.complete_h264_headers)):
            # just in case that folder got deleted during runtime
            if not os.path.exists(f"./{self.OUTPUT_DIR_NAME}"):
                os.makedirs(f"./{self.OUTPUT_DIR_NAME}", exist_ok=True)
                print(f"Made Directory ./{self.OUTPUT_DIR_NAME}")
            if not os.path.exists(f"./{self.OUTPUT_DIR_NAME}/header_{i}"):
                os.makedirs(f"./{self.OUTPUT_DIR_NAME}/header_{i}", exist_ok=True)
                print(f"Made Directory ./{self.OUTPUT_DIR_NAME}/header_{i}")

        #########################################################################################
        # Step 3 is finally to look for iframes on the drive, and attempt to decode them on the fly with any of the SPS/PPS headers found
        #########################################################################################
        with open(filepath, "rb") as f:
            print("===========================================")
            print(f"[Step 2/{2 if nocleanup else 3}] Finding IDR Frames that could fit SPS/PPS headers...")
            for i, cluster in tqdm(enumerate(iter(lambda: f.read(clustersize), b'')), desc="Reading clusters", total=math.ceil(os.path.getsize(filepath)/clustersize)):
                if (cluster == self.null_block):
                    continue
                # check the cluster for the potential start of an IDR frame
                potential_iframe_starts = [_.start() for _ in re.finditer(self.NALU_IDR_HEADER, cluster)]
                for match in potential_iframe_starts:
                    # finds iframes and appends them to global list(if any, after checking size)
                    self.get_iframes(i,cluster, match)
                for partial_frame in self.potential_iframes:
                    if (partial_frame["new_this_cluster"]): #its already initialised with all the data it needs from this cluster
                        partial_frame["new_this_cluster"] = False
                    else:
                        match_end_index = min(partial_frame["remaining_length"], (clustersize))
                        partial_frame["data"] = partial_frame["data"] + cluster[0:match_end_index]
                        partial_frame["remaining_length"] = partial_frame["remaining_length"] - match_end_index

                # Transfer the frames which have 0 remaining length to be appended to the "complete_iframes" list
                # Also remove them from the "potential_iframes" list
                # If we find a single instance of 00 00 00 00/01/02/03, then it can't be a valid frame, as these should have had efmulation prevention bytes placed in them.
                
                self.complete_iframes += [partial_frame for partial_frame in self.potential_iframes if partial_frame["remaining_length"] <= 0 and b"\x00\x00\x00\x00" not in partial_frame["data"]
                and b"\x00\x00\x00\x01" not in partial_frame["data"]
                and b"\x00\x00\x00\x02" not in partial_frame["data"]
                and b"\x00\x00\x00\x03" not in partial_frame["data"]]


                self.potential_iframes =  [partial_frame for partial_frame in self.potential_iframes if partial_frame["remaining_length"] != 0 ]
                
                if (len(self.complete_iframes) > self.decoding_buffer) and (self.parse_exp_golomb(self.complete_iframes[-1]["data"][1:30],1) != 0 ):
                    self.decode_iframes(self.NUM_PROCS)
                    self.complete_iframes = []

        #Catch any remaining files and decode them
        if len(self.complete_iframes) > 0:
            self.decode_iframes(self.NUM_PROCS)
            self.complete_iframes = []
                        

        #########################################################################################
        # Step 4 is to cleanup the output
        #########################################################################################
        total_saved = 0
        for i, header in enumerate(self.complete_h264_headers):
            total_saved += header["saved_count"]
            os.remove(f"./{self.OUTPUT_DIR_NAME}/header_{i}/blob.h264")

        if nocleanup:
            print("Skipping cleanup as requested")
            print(f"Recovered a total of {total_saved} iframes from {len(self.complete_h264_headers)} header sets")
            print(f"Output is in {self.OUTPUT_DIR_NAME}")
            print("Finished!")
        else:
            print("===========================================")
            print("[Step 3/3] Cleaning up output folders")
            cleanup = 0
            for j in tqdm(range(len(self.complete_h264_headers)), desc="Cleaning up output folders", total=len(self.complete_h264_headers)):
                hashes = []
                for file in glob(f"./{self.OUTPUT_DIR_NAME}/header_{j}/*.jpg"):
                    filehash = md5(self.file_as_bytes(open(file, 'rb'))).hexdigest()
                    if filehash in hashes:
                        os.remove(file)
                        cleanup += 1
                    hashes.append(filehash)
            print(f"Recovered a total of {total_saved} iframes from {len(self.complete_h264_headers)} header sets but removed {cleanup} images in the cleanup process, totalling {total_saved-cleanup}")
            print(f"Output is in {self.OUTPUT_DIR_NAME}")
            print("Finished!")

if __name__ == "__main__":

    #Parse command line arguments if running directly
    OUTPUT_DIR_NAME = "FrameFinder_output"

    parser = argparse.ArgumentParser(
        "frameFinder", description="Attempts to extract i-frames from h264 encoded videos in a disk image, for use when the filesystem information or file header is missing or corrupted")
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
    #########################################################################################
    # Step 0 Initialise some stuff
    #########################################################################################
    # Parse Arguments and set up some operational params
    args = parser.parse_args()
    clustersize = args.cluster_size_bits
    unique = args.unique
    nocleanup = args.nocleanup
    filepath = args.imgfile

    ff = frameFinder(clustersize, nocleanup, unique, output_dir_name=OUTPUT_DIR_NAME)
    ff.process_image(filepath)
