import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont 

single_choice = "pybullet_pics0"

dual_first = "pics0"
dual_second = "pybullet_pics0"
label_new = False

def images_to_video(image_folder, video_output, DUAL):
    """
        Creates a new video at `video_output` from ordered images in `image_folder`
    """
    # Ensure ffmpeg is installed
    assert subprocess.call('type ffmpeg', shell=True, 
           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0, "ffmpeg was not found on your system; please install ffmpeg"

    if DUAL or label_new:
        if DUAL:
            rows = 1
            cols = 2
        else:
            rows = 1
            cols = 1

        os.makedirs(os.path.join(image_folder, "labeled_pics"), exist_ok=True)
        for i, (img_path, img_path2) in enumerate(zip(sorted(os.listdir(os.path.join(image_folder, dual_first))), sorted(os.listdir(os.path.join(image_folder, dual_second))))):
            img = Image.open(os.path.join(image_folder, dual_first, img_path))
            img2 = Image.open(os.path.join(image_folder, dual_second, img_path2))
            width, height = img.size
            output_image = Image.new('RGB', (width * cols, height * rows), color=(255, 255, 255))
        
            draw = ImageDraw.Draw(output_image)
            font = ImageFont.load_default()

            positions = [(x, y) for x in range(0, width * cols, width) for y in range(0, height * rows, height)]
            if DUAL:
                imgs = [img, img2]
            elif label_new:
                if single_choice == dual_first:
                    imgs = [img]
                else:
                    imgs = [img2]
            for image, pos in zip(imgs, positions):
                try:
                    output_image.paste(image, pos)
                except OSError as e:
                    print("Caught OSERROR: ", e)


            if i < 30:
                draw.text((cols * width - 60, 10), "begin", fill="red", font=font)

            output_image.save(os.path.join(image_folder, "labeled_pics", img_path))

        os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{image_folder}/labeled_pics/0*.png' -c:v libx264 -pix_fmt yuv420p {video_output}")
        # os.system(f"rm -rf {image_folder}/labeled_pics")
    else:
        for count, filename in enumerate(sorted(os.listdir(os.path.join(image_folder, dual_second)))):
            dst = str(count).zfill(5) + ".png"
            src = os.path.join(image_folder, single_choice, filename)
            dst = os.path.join(image_folder, single_choice, dst)
            os.rename(src, dst)
        command = "ffmpeg -framerate 30 -i '{}/{}/%05d.png' -c:v libx264 -pix_fmt yuv420p {}".format(image_folder, single_choice, video_output)
        subprocess.call(command, shell=True)

def combine_videos(eval_dir, video_filename="video.mp4", combined_video_filename="combined_video.mp4"):
    """
        Combines all videos in `eval_dir` into a single video `combined_video_filename`

        eval_dir
            - folder1
                - video.mp4
            - folder2
                - video.mp4 
            - (newly generated) combined_video.mp4
    """

    video_paths = [os.path.join(eval_dir, f"{absolute_path}/{video_filename}") for absolute_path in sorted(os.listdir(eval_dir)) if os.path.isdir(os.path.join(eval_dir, absolute_path)) and f"{video_filename}" in os.listdir(os.path.join(eval_dir, absolute_path))]
    # concatenate all videos in video_paths
    with open("input.txt", "w") as f:
        for video_path in video_paths:
            f.write(f"file {video_path}\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", f"{eval_dir}/{combined_video_filename}"])

def save_multi_layer_videos(main_output_folders, dual_video=True):
    """
        Saves videos from all the folders in `main_output_folders` in the following structure
    
        main_output_folder[0]
            val
                - run1
                    - (generated if not already) video.mp4
                - run2
                    - (generated if not already) video.mp4 
                - (newly generated) combined_video.mp4
            checkpoint50
                ...
        main_output_folder[1]
            ...
            
    """

    for main_output_folder in main_output_folders:
        for different_checkpoint_model in os.listdir(main_output_folder):
            if not os.path.isdir(os.path.join(main_output_folder, different_checkpoint_model)):
                continue
            for image_folder_name in sorted(os.listdir(os.path.join(main_output_folder, different_checkpoint_model))):
                if not os.path.isdir(os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)):
                    continue
                image_folder = os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)
                video_output = os.path.join(main_output_folder, different_checkpoint_model, image_folder_name, "video.mp4")
                print(image_folder)
                
                if f"video.mp4" in os.listdir(os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)):
                    # os.system(f"rm -rf {video_output}")
                    continue
                images_to_video(image_folder, video_output, DUAL=dual_video)

            # Combine all videos in the directory
            combine_videos(os.path.join(main_output_folder, different_checkpoint_model))

if __name__ == "__main__":
    # Get the image folder from command line arguments
    DUAL = True
    multi_folder = True
    number_of_runs = 20
    directory_folder = sys.argv[1]
    video_name = "pyb_single_video.mp4" if not DUAL else "video.mp4"
    combined_video_filename="pyb_single_combined_video.mp4" if not DUAL else "combined_video.mp4"

    # # ! Single Folder
    if not multi_folder:
        image_folder = os.path.join(directory_folder, directory_folder)
        video_output = os.path.join(directory_folder, directory_folder, video_name)
        print(image_folder)
        images_to_video(image_folder, video_output, DUAL)
    # ! Folder of folders
    else:
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
    

