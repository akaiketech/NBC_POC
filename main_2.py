import streamlit as st
import cv2
from sahi.predict import get_prediction
from sahi import AutoDetectionModel
from PIL import Image
import numpy as np
from io import BytesIO
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering

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

# Initialize the YOLOv8 model
@st.cache_resource
def load_model():
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path='best_epoch_235.pt',
        confidence_threshold=0.3,
        device="cpu",
    )
    return detection_model
    

def process_image(image, detection_model):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width, _ = image_cv.shape
    full_shape = [height, width]

    temp_file_path = "temp_image.jpg"
    image.save(temp_file_path)

    result = get_prediction(temp_file_path, detection_model, full_shape=full_shape)

    bounding_boxes = []
    areas = []
    
    for prediction in result.object_prediction_list[:2000]:
        bbox = prediction.bbox.to_xyxy()
        x_min, y_min, x_max, y_max = map(int, bbox)
        # Calculate the area of the bounding box
        area = (x_max - x_min) * (y_max - y_min)
        areas.append(area)
        bounding_boxes.append(bbox)

    # Cluster the areas using AgglomerativeClustering
    agglom_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
    clusters = agglom_cluster.fit_predict(np.array(areas).reshape(-1, 1))

    # Calculate the mean areas for each cluster
    mean_area_cluster_0 = np.mean([areas[i] for i in range(len(areas)) if clusters[i] == 0])
    mean_area_cluster_1 = np.mean([areas[i] for i in range(len(areas)) if clusters[i] == 1])

    # Print the mean areas of the two clusters
    print(f"Mean area of cluster 0: {mean_area_cluster_0}")
    print(f"Mean area of cluster 1: {mean_area_cluster_1}")

    # Check the modulus difference of the cluster means
    if abs(mean_area_cluster_0 - mean_area_cluster_1) > 10000:
        # Only include bounding boxes with area greater than 4000
        filtered_bounding_boxes = [bbox for bbox, area in zip(bounding_boxes, areas) if area > 4000]
    else:
        filtered_bounding_boxes = bounding_boxes

    # Draw the bounding boxes on the image
    for bbox in filtered_bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    processed_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    return processed_image, filtered_bounding_boxes




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

    # Display the uploaded image with styling
    st.image(img, caption="Uploaded Image", use_column_width=True, output_format="auto")

    # Center the Detect Objects button
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    if st.button("🚀 Detect Objects"):
        with st.spinner("Processing..."):
            processed_image, bounding_boxes = process_image(img, detection_model)

            # Check if processed_image is valid
            if processed_image is not None and isinstance(processed_image, np.ndarray):
                # Display the processed image with bounding boxes
                st.markdown(f"<p style='color: red;'>Number of objects detected: {len(bounding_boxes)}</p>", unsafe_allow_html=True)

                # Show bounding box count
                st.success(f"Number of objects detected: {len(bounding_boxes)}")

                # Provide download button for the processed image
                buffered = BytesIO()
                try:
                    result_image = Image.fromarray(processed_image)
                    result_image.save(buffered, format="JPEG")
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=buffered.getvalue(),
                        file_name="processed_image.jpg",
                        mime="image/jpeg"
                    )
                except Exception as e:
                    st.error(f"Error saving image: {e}")
            else:
                st.error("Processed image is not valid. Please check the process_image function.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer section
st.markdown("<br><br><hr><p class='footer'>Developed by Akaike Technologies</p>", unsafe_allow_html=True)