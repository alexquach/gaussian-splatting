import os
import sys
import subprocess

def images_to_video(image_folder, video_output):
    # Ensure ffmpeg is installed
    assert subprocess.call('type ffmpeg', shell=True, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0, "ffmpeg was not found on your system; please install ffmpeg"

    # Define the command for converting images to video using ffmpeg
    command = "ffmpeg -framerate 30 -i '{}/%05d.png' -c:v libx264 -pix_fmt yuv420p {}".format(image_folder, video_output)

    # Execute the command
    subprocess.call(command, shell=True)

def combine_videos(eval_dir, video_filename="video.mp4"):
    import subprocess

    video_paths = [os.path.join(eval_dir, f"{absolute_path}/video.mp4") for absolute_path in sorted(os.listdir(eval_dir)) if os.path.isdir(os.path.join(eval_dir, absolute_path)) ]
    combined_video_filename = "combined_video.mp4"
    # concatenate all videos in video_paths
    with open("input.txt", "w") as f:
        for video_path in video_paths:
            f.write(f"file {video_path}\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", f"{eval_dir}/{combined_video_filename}"])

if __name__ == "__main__":
    # Get the image folder from command line arguments
    directory_folder = sys.argv[1]

    for image_folder_name in os.listdir(directory_folder):
        image_folder = os.path.join(directory_folder, image_folder_name, "pics0")
        video_output = os.path.join(directory_folder, image_folder_name, "video.mp4")
        print(image_folder)
        if f"video.mp4" in os.listdir(os.path.join(directory_folder, image_folder_name)):
            continue
        images_to_video(image_folder, video_output)

    # Combine all videos in the directory
    combine_videos(directory_folder)
    

