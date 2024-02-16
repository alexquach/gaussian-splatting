import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont 

def images_to_video(image_folder, video_output, DUAL):
    # Ensure ffmpeg is installed
    assert subprocess.call('type ffmpeg', shell=True, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0, "ffmpeg was not found on your system; please install ffmpeg"

    for count, filename in enumerate(sorted(os.listdir(os.path.join(image_folder)))):
        dst = str(count).zfill(5) + ".png"
        src = os.path.join(image_folder, filename)
        dst = os.path.join(image_folder, dst)
        os.rename(src, dst)
        # flip the r and b channels
        # img = Image.open(dst)
        # r, g, b = img.split()
        # img = Image.merge("RGB", (b, g, r))
        # img.save(dst)
    command = "ffmpeg -framerate 30 -i '{}/%05d.png' -c:v libx264 -pix_fmt yuv420p {}".format(image_folder, video_output)
    subprocess.call(command, shell=True)

def combine_videos(eval_dir, video_filename="video.mp4", combined_video_filename="combined_video.mp4"):
    import subprocess

    video_paths = [os.path.join(eval_dir, f"{absolute_path}/{video_filename}") for absolute_path in sorted(os.listdir(eval_dir)) if os.path.isdir(os.path.join(eval_dir, absolute_path)) and f"{video_filename}" in os.listdir(os.path.join(eval_dir, absolute_path))]
    # concatenate all videos in video_paths
    with open("input.txt", "w") as f:
        for video_path in video_paths:
            f.write(f"file {video_path}\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", f"{eval_dir}/{combined_video_filename}"])

if __name__ == "__main__":
    # Get the image folder from command line arguments
    DUAL = False
    number_of_runs = 10
    directory_folder = sys.argv[1]
    video_name = "single_video.mp4" if not DUAL else "video.mp4"
    combined_video_filename="single_combined_video.mp4" if not DUAL else "combined_video.mp4"

    # # ! Single Folder
    # image_folder = os.path.join(directory_folder, directory_folder)
    # video_output = os.path.join(directory_folder, directory_folder, video_name)
    # print(image_folder)
    # images_to_video(image_folder, video_output, DUAL)

    # ! Folder of folders
    for image_folder_name in sorted(os.listdir(directory_folder))[:number_of_runs]:
        if os.path.isfile(os.path.join(directory_folder, image_folder_name)):
            continue

        image_folder = os.path.join(directory_folder, image_folder_name)
        video_output = os.path.join(directory_folder, image_folder_name, video_name)
        print(image_folder)
    
        if not os.path.isdir(os.path.join(directory_folder, image_folder_name)) or video_name in os.listdir(os.path.join(directory_folder, image_folder_name)):
            os.system(f"rm -rf {video_output}")
            # continue
        images_to_video(image_folder, video_output, DUAL)

    # Combine all videos in the directory
    combine_videos(directory_folder, video_name, combined_video_filename)
    

