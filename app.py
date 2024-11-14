import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI for the app
st.title('Object Detection and EDA with YOLO')

# --- FUNCTION 1: IMAGE UPLOAD AND PREPROCESSING ---
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    height, width, channels = image.shape
    return image, width, height

# --- FUNCTION 2: YOLO OBJECT DETECTION ---
def detect_objects(image, width, height, net, output_layers):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    threshold = 0.5

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > threshold:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    return class_ids, confidences, boxes

# --- FUNCTION 3: NON-MAXIMA SUPPRESSION AND DATA CREATION ---
def apply_nms(class_ids, confidences, boxes, classes):
    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    if len(indices) == 0:
        return [], []
    else:
        indices = indices.flatten() if indices is not None else []
    
    # Collect detection data for the detected objects
    detected_objects = {
        'class_ids': [class_ids[i] for i in indices],
        'confidences': [confidences[i] for i in indices],
        'boxes': [boxes[i] for i in indices],
        'classes': [classes[class_ids[i]] for i in indices],
    }
    return detected_objects

# --- FUNCTION 4: DRAW BOUNDING BOXES ON IMAGE ---
def draw_bounding_boxes(image, detected_objects):
    for i in range(len(detected_objects['boxes'])):
        x, y, w, h = detected_objects['boxes'][i]
        label = detected_objects['classes'][i]
        confidence = detected_objects['confidences'][i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# --- FUNCTION 5: DISPLAY OBJECT DETECTION DATA ---
def display_detection_data(detected_objects):
    df = pd.DataFrame({
        'Class': detected_objects['classes'],
        'Confidence': detected_objects['confidences'],
        'Box': detected_objects['boxes'],
    })

    st.subheader('Basic Information of Detected Objects')
    st.write(df.describe())

    # Class Distribution
    st.subheader('Class Distribution of Detected Objects')
    class_counts = df['Class'].value_counts()

    st.subheader('Custom Class Distribution Bar Chart')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(class_counts))
    ax.bar(class_counts.index, class_counts.values, color=colors)
    ax.set_title('Class Distribution of Detected Objects')
    ax.set_xlabel('Object Class')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Pie Chart
    st.subheader('Pie Chart for Object Type Distribution')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(class_counts)))
    ax.axis('equal')
    ax.set_title('Distribution of Detected Object Types')
    st.pyplot(fig)

    # Confidence Distribution
    st.subheader('Confidence Score Distribution of Detected Objects')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Confidence'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Confidence Scores for Detected Objects')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Bounding Box Dimensions (Width and Height)
    st.subheader('Bounding Box Dimensions (Width and Height) for Detected Objects')
    df['Width'] = df['Box'].apply(lambda x: x[2])
    df['Height'] = df['Box'].apply(lambda x: x[3])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['Width'], bins=20, kde=True, color='orange', ax=ax)
    ax.set_title('Distribution of Bounding Box Widths for Detected Objects')
    ax.set_xlabel('Width')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['Height'], bins=20, kde=True, color='green', ax=ax)
    ax.set_title('Distribution of Bounding Box Heights for Detected Objects')
    ax.set_xlabel('Height')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# --- FUNCTION 6: DOWNLOAD IMAGE WITH BOUNDING BOXES ---
def download_image(image_rgb):
    st.download_button(
        label="Download Image with Bounding Boxes",
        data=cv2.imencode('.jpg', image_rgb)[1].tobytes(),
        file_name='image_with_bounding_boxes.jpg',
        mime='image/jpeg'
    )

# Main code execution
def main():
    # Show file uploader for the image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image
        image, width, height = load_image(uploaded_file)

        # Load YOLO model and class names
        config_path = 'yolov3.cfg'
        weights_path = 'yolov3.weights'
        coco_names_path = 'coco.names'

        net = cv2.dnn.readNet(weights_path, config_path)
        with open(coco_names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Detect objects using YOLO
        class_ids, confidences, boxes = detect_objects(image, width, height, net, output_layers)

        # Apply NMS and extract detection data
        detected_objects = apply_nms(class_ids, confidences, boxes, classes)

        if len(detected_objects['classes']) == 0:
            st.write("No objects detected in the image.")
        else:
            # Draw bounding boxes on the image
            image_rgb = draw_bounding_boxes(image, detected_objects)

            # Display image with bounding boxes
            st.subheader('Image with Bounding Boxes and Object Names')
            st.image(image_rgb, caption='Detected Objects in Image', use_column_width=True)

            # Display detection data
            display_detection_data(detected_objects)

            # Allow image download
            download_image(image_rgb)

if __name__ == "__main__":
    main()
