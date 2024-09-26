import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import requests
import os

# Set page layout for mobile responsiveness
st.set_page_config(page_title="NBC Ring Detection", layout="centered")

# Custom CSS for improved UI
st.markdown("""
    <style>
    /* Background color */
    .stApp {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Title styling */
    h1 {
        text-align: center;
        color: #333333;
    }
    
    /* Custom font and button styling */
    button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    
    /* Center the uploader */
    .stFileUploader {
        margin-top: 30px;
        display: flex;
        justify-content: center;
    }

    /* Image with a border */
    img {
        border: 2px solid #ddd;
        border-radius: 8px;
    }

    /* Footer styling */
    hr {
        border: 1px solid #ccc;
        margin-top: 50px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #666;
    }
    
    /* Button alignment */
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

    </style>
""", unsafe_allow_html=True)

# Function to download the YOLO model checkpoint
def download_model(link):
    response = requests.get(link)
    model_path = "best_epoch_535.pt"
    
    with open(model_path, "wb") as file:
        file.write(response.content)
    
    return model_path

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = download_model("https://drive.google.com/file/d/18_2u328wBQAp21PMHbyiSqjT6NCoERe9/view?usp=sharing")  # Replace with your model link
    model = YOLO(model_path)  # Use the YOLOv8 model
    return model

# Process the image and apply object detection
def process_image(image, model):
    # Save the uploaded image temporarily
    temp_file_path = "temp_image.jpg"
    image.save(temp_file_path)

    # Run YOLO model on the image
    res = model(temp_file_path, max_det=2000)
    
    result = res[0].boxes.xywh.cpu().numpy()  # Get bounding box coordinates
    
    # Prepare image for plotting
    fig, ax = plt.subplots(1, figsize=(12, 12))  # Increase the figure size for visibility
    ax.imshow(image)

    # Draw bounding boxes on the image
    for i, box in enumerate(result):
        x, y, w, h = box
        # Create a Rectangle patch in green
        rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')  # Turn off axis numbers and ticks
    
    # Save the image with bounding boxes to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    return buf, len(result)


# Load YOLOv8 model
detection_model = load_model()

# Streamlit UI
st.title("🔍 NBC Ring Detection")

# Centered uploader
st.markdown("<h3 style='text-align: center;'>Upload an Image for Detection</h3>", unsafe_allow_html=True)

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Center the Detect Objects button
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    
    if st.button("🚀 Detect Objects"):
        with st.spinner("Processing..."):
            processed_image_buf, num_boxes = process_image(img, detection_model)

            # Show the result
            st.markdown(f"<p style='color: red;'>Number of objects detected: {num_boxes}</p>", unsafe_allow_html=True)
            st.success(f"Number of objects detected: {num_boxes}")

            # Provide download button for the processed image
            st.download_button(
                label="📥 Download Processed Image",
                data=processed_image_buf,
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer section
st.markdown("<br><br><hr><p class='footer'>Developed by Akaike Technologies</p>", unsafe_allow_html=True)
