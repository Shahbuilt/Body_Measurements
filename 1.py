from logging import root
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2

# Load the object detection model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
        serialized_graph = f.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def calculate_measurements(image_path):
    # Load the image and convert it to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the image through the object detection model
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Get handles to input and output tensors
            ops = detection_graph.get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = detection_graph.get_tensor_by_name(
                        tensor_name)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={
                                   image_tensor: np.expand_dims(image, 0)})

            # Filter out detections with low confidence scores
            boxes = output_dict['detection_boxes'][0]
            scores = output_dict['detection_scores'][0]
            classes = output_dict['detection_classes'][0]
            valid_indices = np.where(scores >= 0.5)[0]
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            classes = classes[valid_indices]

            # Calculate shoulder width
            shoulder_width = None
            person_indices = np.where(classes == 1)[0]
            if len(person_indices) > 0:
                person_box = boxes[person_indices[0]]
                shoulder_width = (
                    person_box[3] - person_box[1]) * image.shape[1]

            # Calculate waist approximation width
            waist_width = None
            if len(person_indices) > 0:
                person_box = boxes[person_indices[0]]
                waist_width = (person_box[2] - person_box[0]) * image.shape[1]

            # Return the measurements as a dictionary
            measurements = {'shoulder_width': shoulder_width,
                            'waist_width': waist_width}
            return measurements


def upload_image():
    # Open a file dialog to select an image
    filepath = filedialog.askopenfilename()
    if filepath:
        # Calculate measurements and display them
        measurements = calculate_measurements(filepath)
        message = f"Shoulder Width: {measurements['shoulder_width']:.2f}\nWaist Width: {measurements['waist_width']:.2f}"
        result_label.config(text=message)


# Create the main window
window = tk.Tk()
window.title("Body Measurements Calculator")

# Create the widgets
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
result_label = tk.Label(window, text="")

# Add the widgets to the window
upload_button.pack(pady=10)
result_label.pack(pady=10)

# Start the main event loop
root.mainloop()
