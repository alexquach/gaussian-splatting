import os
import sys
import subprocess

def images_to_video(image_folder):
    # Ensure ffmpeg is installed
    assert subprocess.call('type ffmpeg', shell=True, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0, "ffmpeg was not found on your system; please install ffmpeg"

    # Define the command for converting images to video using ffmpeg
    command = "ffmpeg -framerate 30 -i '{}/%05d.png' -c:v libx264 -pix_fmt yuv420p {}.mp4".format(image_folder, image_folder)

    # Execute the command
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    # Get the image folder from command line arguments
    image_folder = sys.argv[1]

    # Convert images to video
    images_to_video(image_folder)
