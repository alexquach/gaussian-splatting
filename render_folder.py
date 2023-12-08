import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont 

def images_to_video(image_folder, video_output, DUAL):
    # Ensure ffmpeg is installed
    assert subprocess.call('type ffmpeg', shell=True, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0, "ffmpeg was not found on your system; please install ffmpeg"

    if DUAL:
        rows = 1
        cols = 2

        os.makedirs(os.path.join(image_folder, "labeled_pics"), exist_ok=True)
        for i, (img_path, img_path2) in enumerate(zip(sorted(os.listdir(os.path.join(image_folder, "pics0"))), sorted(os.listdir(os.path.join(image_folder, "pybullet_pics0"))))):
            img = Image.open(os.path.join(image_folder, "pics0", img_path))
            img2 = Image.open(os.path.join(image_folder, "pybullet_pics0", img_path2))
        # for i, (img_path, img_path2) in enumerate(zip(sorted(os.listdir(os.path.join(image_folder, "gt"))), sorted(os.listdir(os.path.join(image_folder, "renders", "pics0"))))):
        #     img = Image.open(os.path.join(image_folder, "gt", img_path))
        #     img2 = Image.open(os.path.join(image_folder, "renders", "pics0", img_path2))
        #     img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            width, height = img.size
            output_image = Image.new('RGB', (width * cols, height * rows), color=(255, 255, 255))
        
            draw = ImageDraw.Draw(output_image)
            font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", size=20)

            positions = [(x, y) for x in range(0, width * cols, width) for y in range(0, height * rows, height)]
            for image, pos in zip([img, img2], positions):
                output_image.paste(image, pos)

            if i < 30:
                draw.text((cols * width - 60, 10), "begin", fill="red", font=font)

            output_image.save(os.path.join(image_folder, "labeled_pics", img_path))

        os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{image_folder}/labeled_pics/0*.png' -c:v libx264 -pix_fmt yuv420p {video_output} > /dev/null 2>&1")
        os.system(f"rm -rf {image_folder}/labeled_pics")
    else:
        for count, filename in enumerate(sorted(os.listdir(os.path.join(image_folder, "pybullet_pics0")))):
            dst = str(count).zfill(5) + ".png"
            src = os.path.join(image_folder, "pybullet_pics0", filename)
            dst = os.path.join(image_folder, "pybullet_pics0", dst)
            os.rename(src, dst)
        command = "ffmpeg -framerate 30 -i '{}/pybullet_pics0/%05d.png' -c:v libx264 -pix_fmt yuv420p {}".format(image_folder, video_output)
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
    DUAL = True
    directory_folder = sys.argv[1]
    video_name = "single_video.mp4" if not DUAL else "video.mp4"
    combined_video_filename="single_combined_video.mp4" if not DUAL else "combined_video.mp4"

    # # ! Single Folder
    # image_folder = os.path.join(directory_folder, directory_folder)
    # video_output = os.path.join(directory_folder, directory_folder, video_name)
    # print(image_folder)
    # images_to_video(image_folder, video_output, DUAL)

    # ! Folder of folders
    for image_folder_name in sorted(os.listdir(directory_folder))[:20]:
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
    

