import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import os

# from src.deepLabModel import DeepLabModel
# from src.visualize import main


def estimate_body_measurements(image_path, height):
    # Initialize model
    MODEL_NAME = 'xception_coco_voctrainval'
    # MODEL = DeepLabModel(MODEL_NAME)

    # Load image
    image = cv2.imread(image_path)
    back = cv2.imread('sample_data/input/background.jpeg', cv2.IMREAD_COLOR)

    # Remove background using DeepLab model
    res_im, seg = MODEL.run(image)
    seg = cv2.resize(seg.astype(np.uint8), image.shape[:2][::-1])
    mask_sel = (seg == 15).astype(np.float32)
    mask = 255 * mask_sel.astype(np.uint8)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    res = cv2.bitwise_and(img, img, mask=mask)
    bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # Estimate body measurements using OpenCV functions
    gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour and fit ellipse
    max_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(max_contour)

    # Calculate waist and hip measurements
    waist, hip = ellipse[1]
    if waist > hip:
        waist, hip = hip, waist
    waist = int(waist)
    hip = int(hip)

    # Calculate other measurements based on height
    neck = int(0.27 * height)
    chest = int(0.75 * height)
    sleeve = int(0.3 * height)
    shoulder = int(0.35 * height)
    inseam = int(0.5 * height)

    # Visualize measurements on image
    measurements = {'Height': height, 'Waist': waist, 'Hip': hip, 'Neck': neck,
                    'Chest': chest, 'Sleeve': sleeve, 'Shoulder': shoulder, 'Inseam': inseam}
    main(bg_removed, height, measurements)

    return measurements


# Example usage
image_path = 'sample_data/input/arsalan2.jpg'
height = 180  # in cm
measurements = estimate_body_measurements(image_path, height)
print(measurements)
