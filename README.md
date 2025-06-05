# From-Camera-to-Cloud: End-to-End Face Recognition System

This project is a **fully integrated, beginner-friendly pipeline** for building a real-time **Face Recognition System** using:

- **YOLOv11** for face detection  
- **DenseNet121** for face classification  
- **dlib** for automated annotation  
- **Amazon SageMaker** for model training and deployment  
- **PostgreSQL** for identity metadata  
- **Streamlit** for a live webcam-based UI  

---

## 🔧 How It Works

### 1. Prepare Dataset
- Organize images in folders per identity
- Use `dlib` for bounding box generation
- Create:
  - `yolo_format/` → For YOLOv11 detection
  - `cropped_faces/` → For DenseNet classification
- Perform model training on `Amazon SageMaker`
- Deploy the models on `SageMaker endpoints`
- Run live inference on `streamlit` interface.

---

## 📖 Full Guide

Refer to the blog for detailed steps and code walkthrough:

👉 [Read the Blog](https://medium.com/@bhipanshudhupar/from-camera-to-cloud-building-a-smart-face-recognition-system-end-to-end-cd6a30bffbdc)

---

---

Designed for developers and researchers exploring practical applications of AI and cloud integration.

