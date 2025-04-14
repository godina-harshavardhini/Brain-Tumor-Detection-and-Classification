import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLO model
@st.cache_resource  # Cache to prevent reloading the model multiple times
def load_model():
    model = YOLO(r"/Users/harshavardhini/Downloads/yolov8_model.pt")

    return model

model = load_model()

# Streamlit UI
st.title("YOLO Object Detection")
st.write("Upload an image to detect objects using YOLOv8")

# Upload image for testing
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_image)
    image = np.array(image)  # Convert PIL to numpy array (for OpenCV)
    
    # Run YOLO inference
    results = model(image)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert OpenCV image back to PIL format for Streamlit display
    st.image(image, caption="Detected Objects", use_column_width=True)
