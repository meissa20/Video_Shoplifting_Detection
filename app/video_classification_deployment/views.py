from django.shortcuts import render
from django.http import HttpResponse
import cv2, tempfile, os
import sys
sys.path.append("C:/Users/Lenovo/Documents/Summer 2025/Cellula/Task 3/pytorch-i3d")
from pytorch_i3d import InceptionI3d
import numpy as np
import torch

def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0  
        frames.append(frame)
    cap.release()
    return np.array(frames)  # Shape: (num_frames, height, width, channels)

def read_data(path):
    video_numpy = load_video_frames(path) # Shape: (num_frames, height, width, channels)
    video_tensor = torch.tensor(video_numpy, dtype=torch.float32)
    # video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0) # (num_frames, channels, height, width) From scratch model
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    return video_tensor


def index(request):
    return render(request, 'app.html')

def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        file = request.FILES['video']
        file_bytes = file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp: # Creates a temporary file on disk with the .mp4 extension.
                                                                              # delete=False means the file will stay until we delete it manually.
                                                                              # Writes the uploaded video bytes into that file.
                                                                              # Saves the path of that temp file into temp_path.
            tmp.write(file_bytes)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)    # Can not open raw bytes directly, It only understands
                                             # a path to a file  OR  camera index
        if not cap.isOpened():
            os.remove(temp_path)
            return HttpResponse("Could not open video")

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        FPS = 25
        out_video = cv2.VideoWriter(
            "../model_preprocessed.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS,
            (width, height),
            isColor=True
        )

        step = int(frames / 70)
        frame_idx, frame_written = 0, 0
        while cap.isOpened and frame_written < 70:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                break
            out_video.write(frame)
            frame_written+=1
            frame_idx+=step
        cap.release()
        out_video.release()
        os.remove(temp_path)
        
        i3d = InceptionI3d(1, in_channels=3)
        i3d.load_state_dict(torch.load("../best_i3d.pth", map_location='cpu'))
        
        video_tensor = read_data("../model_preprocessed.mp4")
        
        outputs = i3d(video_tensor)
        outputs = torch.mean(outputs, dim=2)
        preds = (torch.sigmoid(outputs) > 0.5 ).float()
        pred = int(preds.item())      # convert tensor([[1.]]) → 1

        labels = {0: "not shoplifter", 1: "shoplifter"}
        result = labels[pred]
        
        return render(request, 'app.html', {"prediction": labels[pred]})

    return HttpResponse("No video uploaded")

