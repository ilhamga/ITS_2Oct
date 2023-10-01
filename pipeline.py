import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
import time

# YoloFace imports
import sys
sys.path.append('modules/yolo_face_detection/')
from face_detector import YoloDetector

# AttractiveNet imports
from tensorflow.keras.models import load_model
sys.path.append('modules/attractive_scale/')
import helper

# Load face detector model
face_model = YoloDetector(target_size=720, device="cpu", min_face=50)

# Load attractivenet model
model_name = 'attractiveNet_mnv2'
model_path = 'modules/attractive_scale/models/' + model_name + '.h5'
model = load_model(model_path)

# Create an argument parser
parser = argparse.ArgumentParser(description='Extract frames from a video and save them as images.')
parser.add_argument('video_path', type=str, help='Path to the input MP4 video file')
args = parser.parse_args()
video_base_name = video_base_name = os.path.splitext(os.path.basename(args.video_path))[0]

# Open the video file
cap = cv2.VideoCapture(args.video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the FPS value of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Get the frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a directory to save the frames (if it doesn't exist)
output_dir = video_base_name + '_frames'
os.makedirs(output_dir, exist_ok=True)

# Initialize frame counter
frame_number = 0

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_det_duration = 0

with tqdm(total=total_frames, unit="frames") as progress_bar:
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Save the frame as an image with the naming convention
        frame_filename = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(args.video_path))[0]}_{frame_number}.jpg')

        start_time = time.time()
        face_bboxes, face_points = face_model.predict(frame)
        total_det_duration += time.time() - start_time
        

        for face_bbox in face_bboxes[0]:
            cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 0, 0), 2)
            face_frame = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]

            score = model.predict(np.expand_dims(helper.preprocess_image(face_frame,(350,350)), axis=0))
            text1 = f'Score: {str(round(score[0][0],1))}'
            cv2.putText(frame, text1, (face_bbox[0] + 5 , face_bbox[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

        cv2.imwrite(frame_filename, frame)

        # Increment the frame counter
        frame_number += 1

        # Update the progress bar
        progress_bar.update(1)

# Release the video file
cap.release()

# Print the total number of frames extracted
print(f"Total frames extracted: {frame_number}")

# Create a video from the extracted frames
frame_files = [os.path.join(output_dir, f'{os.path.splitext(os.path.basename(args.video_path))[0]}_{i}.jpg') for i in range(frame_number)]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_name = f'{video_base_name}_inferenced.mp4'
output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

for frame_file in tqdm(frame_files, unit="frames"):
    frame = cv2.imread(frame_file)
    output_video.write(frame)

# Release the output video file
output_video.release()

print(f"Total detection time: {total_det_duration} seconds")
print(f"Avg detection time: {total_det_duration/total_frames} seconds")