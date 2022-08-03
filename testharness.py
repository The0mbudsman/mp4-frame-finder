from glob import glob
from frameFinder import frameFinder
import ffmpeg
from colorthief import ColorThief
import os

basedir = "5k_new_cleanup_N_SR_TB"
filepath = "../media/MP4-FROM-WEB-h264/*.mp4"
correct = []
toofew = []
toomany = []
number_of_files = len(sorted(glob(filepath)))
for file in sorted(glob(filepath)):
    tokens = file.split("/")
    file_name_only = tokens[len(tokens)-1].split(".")[0]
    ff = frameFinder(clustersize=4096, output_dir_name=f"{basedir}/{file_name_only}")
    ff.process_image(file)
    try:
        process = (ffmpeg
            .input(file, skip_frame="nokey")
            .output(f"{basedir}/{file_name_only}/header_0/yi%03d.jpg", vsync="0", frame_pts="true", loglevel="quiet")
            .run()
            )
        files= glob(f"{basedir}/{file_name_only}/header_0/*.jpg")
        for img in files:
            color_thief = ColorThief(img)
            try:
                palette = color_thief.get_palette(color_count=10)[1:]
                delete = all(colour == palette[0] for colour in palette)
                if delete:
                    os.remove(img)
            except:
                continue

        ffmpeg_files= glob(f"{basedir}/{file_name_only}/header_0/yi*.jpg")
    except:
        ffmpeg_files= [0]

    process = ffmpeg.input(file).output(f"{basedir}/{file_name_only}/header_0/{file_name_only}.h264", vcodec="copy", frame_pts="true", an=None, **{'bsf:v': 'h264_mp4toannexb'}).run()



    framefinder_files= glob(f"{basedir}/{file_name_only}/header_0/frame*jpg")

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
