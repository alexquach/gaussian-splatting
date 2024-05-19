# this is a script to make a video of the images after resizing them to 224x224
import os
import cv2
import numpy as np
from PIL import Image

# path to the folders
path = "/home/makramchahine/repos/gaussian-splatting/generated_paths2"

# list the folders and keep track of their names
folders = os.listdir(path)
folders.sort()

folders = ["cl_blip2_5obj_50pp_nogpu_3hz_red ball"]

def save_folders(folders):
    for fold in folders:
        folda = os.path.join(path, fold, "val")
        # list the runs under the folder
        runs = os.listdir(folda)
        runs.sort()

        # create the video writer in mp4 format and 10 fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # if no out_video folder exists, create it
        if not os.path.exists("out_video"):
            os.makedirs("out_video")
        # save video in the out_video folder
        out = cv2.VideoWriter(f"out_video/{fold}.mp4", fourcc, 20, (224, 224))

        for run in runs:
            last_view = os.path.join(folda, run, "last_view.txt")
            run = os.path.join(folda, run, "pybullet_pics0")
            # list the images under the run
            images = os.listdir(run)
            images.sort()

            with open(last_view, "r") as f:
                first_line = f.readline().strip()
                if first_line == "fail":
                    string = "F"
                elif first_line == "success":
                    string = "S"

            for i, image in enumerate(images):
                if i < 8000:
                    image = os.path.join(run, image)
                    # load the image
                    img = Image.open(image)
                    # make the image have 3 channels
                    img = img.convert('RGB')
                    # resize the image to 224x224
                    img = img.resize((224, 224))
                    # convert the image to a numpy array
                    img = np.array(img)
                    # convert the numpy array to a BGR image
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # add text to the image with the name of the folder, have it in the top left corner in black color
                    # readable size for 224x224 images
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # smaller text for 224x224 images
                    # cv2.putText(img, f"{string}{fold}", (10, 18), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, f"{fold}", (10, 18), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # write the image to the video
                    out.write(img)

        # release the video writer with video name as the initial folder name
        out.release()
        print(f"Video {fold}.mp4 has been saved!")

if __name__ == "__main__":
    save_folders(folders)

# # write a script that takes a mp4 video as input and returns the same video after inverting the red and blue channels
# # path to the video
# path = "/media/makramchahine/BackendData/Devens_2021-08-04/data_raw/1628107077.97.mp4"
# # create a video capture object
# cap = cv2.VideoCapture(path)
# # get the width and height of the video
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width, height)
# # get the fps of the video
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# # create a video writer object with the same width and height as the input video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("/media/makramchahine/BackendData/Devens_2021-08-04/data_raw/1628107077.97_inverted.mp4", fourcc, fps, (width, height))
# # read the video frame by frame
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # invert the red and blue channels
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # write the frame to the output video
#     out.write(frame)
# # release the video writer and the video capture
# out.release()
# cap.release()
# print("Video has been saved!")