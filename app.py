import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI for the app
st.title('Object Detection and EDA with YOLO')

# Show file uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image as an array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Get image dimensions
    height, width, channels = image.shape
    
    # Load YOLO model if not already loaded
    config_path = 'yolov3.cfg'
    weights_path = 'yolov3.weights'
    coco_names_path = 'coco.names'
    
    net = cv2.dnn.readNet(weights_path, config_path)
    
    with open(coco_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Post-processing to extract detected objects
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

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=threshold, nms_threshold=0.4)
    indices = indices.flatten() if indices is not None else []

    # Collect detection data for the detected objects
    detected_objects = {
        'class_ids': [class_ids[i] for i in indices],
        'confidences': [confidences[i] for i in indices],
        'boxes': [boxes[i] for i in indices],
        'classes': [classes[class_ids[i]] for i in indices],
    }

    # If no objects detected, show a message
    if len(detected_objects['classes']) == 0:
        st.write("No objects detected in the image.")
    else:
        # Display image with bounding boxes and object names
        st.subheader('Image with Bounding Boxes and Object Names')

        for i in range(len(detected_objects['boxes'])):
            x, y, w, h = detected_objects['boxes'][i]
            label = detected_objects['classes'][i]
            confidence = detected_objects['confidences'][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert image to RGB for display in Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image with bounding boxes and object names
        st.image(image_rgb, caption='Detected Objects in Image', use_column_width=True)

        # Convert the detected object data to a DataFrame
        df = pd.DataFrame({
            'Class': detected_objects['classes'],
            'Confidence': detected_objects['confidences'],
            'Box': detected_objects['boxes'],
        })

        # Display basic information about the detected objects
        st.subheader('Basic Information of Detected Objects')
        st.write(df.describe())

        # Show class distribution for detected objects
        st.subheader('Class Distribution of Detected Objects')
        class_counts = df['Class'].value_counts()
        st.bar_chart(class_counts)

        # Show Confidence Score Distribution for the detected objects
        st.subheader('Confidence Score Distribution of Detected Objects')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Confidence'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title('Distribution of Confidence Scores for Detected Objects')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Bounding Box Dimensions (Width and Height)
        st.subheader('Bounding Box Dimensions (Width and Height) for Detected Objects')

        # Extract width and height from the bounding boxes
        df['Width'] = df['Box'].apply(lambda x: x[2])
        df['Height'] = df['Box'].apply(lambda x: x[3])

        # Plot bounding box width distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['Width'], bins=20, kde=True, color='orange', ax=ax)
        ax.set_title('Distribution of Bounding Box Widths for Detected Objects')
        ax.set_xlabel('Width')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Plot bounding box height distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['Height'], bins=20, kde=True, color='green', ax=ax)
        ax.set_title('Distribution of Bounding Box Heights for Detected Objects')
        ax.set_xlabel('Height')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Option to download the image with bounding boxes
        st.download_button(
            label="Download Image with Bounding Boxes",
            data=cv2.imencode('.jpg', image_rgb)[1].tobytes(),
            file_name='image_with_bounding_boxes.jpg',
            mime='image/jpeg'
        )
