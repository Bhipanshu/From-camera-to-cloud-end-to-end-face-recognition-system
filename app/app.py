import streamlit as st
import cv2
import boto3
from PIL import Image
from io import BytesIO
import json
from db_query import query_db
from utils import draw_info_card

sm_client = boto3.client("sagemaker-runtime", region_name="ap-south-1")


st.set_page_config(layout="wide")
st.title("Real-Time Face Recognition")

run = st.checkbox("Start Camera")
frame_area = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG')

        try:
            response = sm_client.invoke_endpoint(
                EndpointName='yolov11-face-detector',
                ContentType='application/octet-stream',
                Body=buffer.getvalue()
            )
            detections = json.loads(response['Body'].read().decode())
        except Exception as e:
            st.error(f"YOLO Endpoint Error: {e}")
            detections = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
            face_buffer = BytesIO()
            face_pil.save(face_buffer, format='JPEG')

            try:
                classify_response = sm_client.invoke_endpoint(
                    EndpointName='densenet-face-classifier',
                    ContentType='application/octet-stream',
                    Body=face_buffer.getvalue()
                )
                classified = json.loads(classify_response['Body'].read().decode())
                name = classified["label"]
            except Exception as e:
                st.error(f"DenseNet Endpoint Error: {e}")
                name = "Unknown"

            dob, addr, phone, img_path = query_db(name)
            draw_info_card(frame, name, dob, phone, (x1, y1, x2, y2), img_path)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame_area.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
