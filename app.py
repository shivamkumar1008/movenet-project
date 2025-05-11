import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import math
import tempfile
import os

# Load MoveNet model
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model.signatures['serving_default']

movenet = load_model()

# Global variables for eye movement tracking
prev_left_eye = None
prev_right_eye = None
eye_movement_total = 0
last_print_time = time.time()

def draw_keypoints(frame, keypoints, threshold=0.5):
    height, width, _ = frame.shape
    for kp in keypoints[0, 0, :, :]:
        y, x, confidence = kp
        if confidence > threshold:
            cx, cy = int(x * width), int(y * height)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    return frame

def face_view(keypoints, threshold=0.4):
    kp = keypoints[0, 0, :, :]
    nose, left_eye, right_eye, left_ear, right_ear = kp[0], kp[1], kp[2], kp[3], kp[4]
    alerts = []
    if nose[2] < threshold:
        alerts.append("Face not visible")
    if left_eye[2] < threshold and right_eye[2] < threshold:
        alerts.append("Eyes not visible")
    if left_ear[2] < threshold and right_ear[2] < threshold:
        alerts.append("Possibly looking away")
    return alerts

def track_eye_movement(left_eye, right_eye):
    global prev_left_eye, prev_right_eye, eye_movement_total, last_print_time
    def euclidean_dist(pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    current_time = time.time()
    if prev_left_eye is not None and prev_right_eye is not None:
        eye_movement_total += euclidean_dist(left_eye, prev_left_eye)
        eye_movement_total += euclidean_dist(right_eye, prev_right_eye)
    if current_time - last_print_time >= 3:
        st.session_state['eye_movement'] = f"Total eye movement in last 3 seconds: {eye_movement_total:.2f}"
        eye_movement_total = 0
        last_print_time = current_time
    prev_left_eye = left_eye
    prev_right_eye = right_eye

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, []
    
    output_frames = []
    alerts_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame for MoveNet
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        
        # Run MoveNet inference
        outputs = movenet(img)
        keypoints = outputs['output_0'].numpy()
        
        # Draw keypoints and process alerts
        frame = draw_keypoints(frame, keypoints)
        alerts = face_view(keypoints)
        alerts_list.append(alerts)
        
        # Track eye movement
        height, width, _ = frame.shape
        left_eye = (keypoints[0, 0, 1, 1] * width, keypoints[0, 0, 1, 0] * height)
        right_eye = (keypoints[0, 0, 2, 1] * width, keypoints[0, 0, 2, 0] * height)
        track_eye_movement(left_eye, right_eye)
        
        # Display alerts on frame
        for idx, alert in enumerate(alerts):
            cv2.putText(frame, alert, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        
        output_frames.append(frame)
    
    cap.release()
    os.unlink(tfile.name)
    
    return output_frames, alerts_list

# Streamlit app layout
st.title("Pose Estimation with MoveNet")
st.write("Upload a video file to perform pose estimation and track eye movement.")

# Initialize session state for eye movement
if 'eye_movement' not in st.session_state:
    st.session_state['eye_movement'] = "Waiting for eye movement data..."

# Video file uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.write("Processing video...")
    frames, alerts_list = process_video(uploaded_file)
    
    if frames:
        # Save processed video to temporary file
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Display video
        st.video(out_file.name)
        
        # Display alerts (showing alerts from the last frame for simplicity)
        st.write("### Alerts")
        for alert in alerts_list[-1] if alerts_list else []:
            st.write(f"- {alert}")
        
        # Display eye movement
        st.write("### Eye Movement")
        st.write(st.session_state['eye_movement'])
        
        os.unlink(out_file.name)