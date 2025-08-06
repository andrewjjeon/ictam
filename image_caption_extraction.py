import argparse
import cv2
import os
import json
import subprocess
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
# Download youtube videos and save them to a directory of your choice using the below command line commands.

# yt-dlp -f "bv[height=1080][ext=mp4]" --remux-video mp4 -o "%(title)s.%(ext)s" -a links.txt -P "/video_dir"
# yt-dlp -f "bv[height=1080][ext=mp4]" --remux-video mp4 -o "%(title)s.%(ext)s" "https://www.youtube.com/watch?v=WNr9fuccoxg&t=169s" -P "/video_dir"

def extract_images(video_dir, img_dir, ffmpeg_path):
    """ Takes a video directory and crops, snapshots frames from the videos using ffmpeg executable process.
    Inputs
        video_dir (path): path of downloaded videos
        img_dir (path): path to save images to
    """
    for filename in os.listdir(video_dir):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            name, ext = os.path.splitext(filename)
            img_name = os.path.join(img_dir, f"{name}_%03d.jpg")

            cmd = [
                ffmpeg_path,
                "-i", video_path,
                "-vf", "fps=1/60,crop=295:295:10:775",  #775-1070 height, 10-305 width
                "-q:v", "2",  # high quality
                img_name
            ]
            subprocess.run(cmd, check=True)

def generate_captions(img_dir, caption_path):
    """ Takes a video directory and crops, snapshots frames from the videos using ffmpeg executable process.
    Inputs
        caption_path (path): path of captions .json file to save to
        img_dir (path): path of images
    """
    entries = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.lower().endswith(".jpg"):
            entries.append({
                "image": filename,
                "text_input": "Who is winning? Describe the minimap.",
                "text_output": "",
            })
    with open(caption_path, "w") as f:
        json.dump(entries, f, indent=4)
    logger.info(f"Saved {len(entries)} prompt-response entries to {caption_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract StarCraft minimap images from youtube videos and generate corresponding captions file for image captioning task.")
    parser.add_argument("--video_dir", required=True, help="Directory with downloaded videos from yt-dlp")
    parser.add_argument("--img_dir", required=True, help="Directory to save extracted images to.")
    parser.add_argument("--caption_path", required=True, help="File path of captions.json to generate.")
    parser.add_argument("--ffmpeg_path", required=True, help="ffmpeg executable to download in order to crop and extract images.")
    args = parser.parse_args()

    extract_images(args.video_dir, args.img_dir, args.ffmpeg_path)
    generate_captions(args.img_dir, args.caption_path)