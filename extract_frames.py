import cv2
import os

# Input/output directories
input_dir = r"C:\Users\Andrew Jeon\OneDrive\Desktop\AISAA\videos"
output_dir = r"C:\Users\Andrew Jeon\OneDrive\Desktop\AISAA\data"

def vid2frames():
    for video in os.listdir(input_dir):
        if video.endswith(".mp4"):
            video_path = os.path.join(input_dir, video)
            vid = cv2.VideoCapture(video_path)

            fps = vid.get(cv2.CAP_PROP_FPS)
            duration = vid.get(cv2.CAP_PROP_FRAME_COUNT) / fps #total seconds of vid

            video_name = os.path.splitext(video)[0] #video name
            output_path = os.path.join(output_dir)
            os.makedirs(output_path, exist_ok=True) #directory already exist, we want to keep adding frames

            min = 0
            extracted_count = 0

            while min * 60 < duration:
                vid.set(cv2.CAP_PROP_POS_MSEC, min * 60 * 1000) #set exact milisecond
                ret, frame = vid.read()
                if not ret:
                    break
                ###########################################################################
                ### TODO: Crop minimap w and h pixels bottom left per frame ###



                ###########################################################################
                frame_name = f"{video_name}_{min:03d}.jpg"
                frame_path = os.path.join(output_path, frame_name)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
                min += 1

            vid.release()
            print(f"Extracted {extracted_count} frames from {video}")

if __name__ == "__main__":
    vid2frames()