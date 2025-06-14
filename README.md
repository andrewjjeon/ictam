Training/Fine-tuning Image Captioning Foundation Models on tactical image-caption pairs. Working towards real-time video-captioning tactical advisor model (VCTAM). First iteration ICTAM complete.

Environment Setup
```bash
conda create -n ictam python=3.13 -y
conda activate ictam
pip install -r requirements.txt
```
If you do by chance want to try training your own ICTAM, you will have to generate your own data similar to how I did and you can even try different HuggingFace image captioning transformers. I went with git and blip. You can download starcraft youtube videos with yt-dlp and then sample, crop minimap images from those downloaded videos using ffmpeg. Consult ICTAM.pdf and the data_processing script for help and guidance, it shouldn't be difficult.

A better README.md will be rolled out but for now please see ICTAM.pdf for further details.
