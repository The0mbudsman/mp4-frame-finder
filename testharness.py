from frameFinder import frameFinder
import os
from glob import glob

import ffmpeg

basedir = "testrun_no_cleanup"
filepath = "../media/MP4-FROM-WEB-total/*.mp4"

correct = []
toomany = []
toofew = []
number_of_files = len(sorted(glob(filepath)))
for file in sorted(glob(filepath)):
    print(f"======={file}======")
    tokens = file.split("/")
    file_name_only = tokens[len(tokens)-1].split(".")[0]
    
    # Get frameFinder to do the file
    ff = frameFinder(clustersize=4096, output_dir_name=f"{basedir}/{file_name_only}", nocleanup=True)
    ff.process_image(file)
    
    # Get ffmpeg to do the file
    try:
        process = (ffmpeg
            .input(file)
            .output(f"{basedir}/{file_name_only}/header_0/yi%03d.jpg", vf="select='eq(pict_type,PICT_TYPE_I)'", vsync="passthrough", loglevel="quiet")
            .run()
            )
        ffmpeg_files= glob(f"{basedir}/{file_name_only}/header_0/yi*.jpg")
    except:
        ffmpeg_files= [0]

    framefinder_files= glob(f"{basedir}/{file_name_only}/header_0/frame*jpg")

     # Compare nunmbers
    if (len(ffmpeg_files) == len(framefinder_files)):
        correct.append(f"{basedir}/{file_name_only}")
    elif (len(ffmpeg_files) > len(framefinder_files)):
        toofew.append(f"{basedir}/{file_name_only}")
    elif (len(ffmpeg_files) < len(framefinder_files)):
        toomany.append(f"{basedir}/{file_name_only}")

with open ("correct.txt", "w") as f:
    for a in correct:
        f.write(f"{str(a)}\n")
with open ("toofew.txt", "w") as f:
    for a in toofew:
        f.write(f"{str(a)}\n")
with open ("toomany.txt", "w") as f:
    for a in toomany:
        f.write(f"{str(a)}\n")

print("============")
print(f"Analysing {number_of_files} files")
print(f"Number correct = {len(correct)}")
print(f"Too Many found = {len(toomany)}")
print(f"Too Few found = {len(toofew)}")
