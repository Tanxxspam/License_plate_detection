import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path

st.title("YOLO Object Detection Application")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

model = YOLO(r"C:\Users\91939\Documents\projects\plate_detection\plate_dection_project\best.pt")


def process_media(input_path, output_path):
    stem = Path(input_path).stem
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        predict_and_save_image(input_path, output_path)
    elif input_path.lower().endswith(('.mp4', '.avi')):
        predict_and_save_video(input_path, output_path)
    else:
        st.error("Unsupported file format. Please upload an image or video.")
        return None
    return output_path
    


def predict_and_save_image(path_test_car, output_image_path):
    results = model.predict(path_test_car)
    image = cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{confidence *100:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, image)
    return output_image_path



def predict_and_save_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file.{video_path}")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgp_frame, device="cpu")
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{confidence *100:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    return output_video_path





if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_path = f"temp/{uploaded_file.name}"
    output_path = f"output/{uploaded_file.name}"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing media...")
    result_path = process_media(input_path, output_path)
    st.write("Processing complete.")

    if result_path and os.path.exists(result_path):
        if result_path.endswith(('.jpg', '.jpeg', '.png')):
            st.image(result_path)
        elif result_path.endswith(('.mp4', '.avi')):
            st.video(result_path)
    else:
        st.error("Processed file not found.")