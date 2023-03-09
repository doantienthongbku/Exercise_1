import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def detect(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    cropped_image = image[0:height, 0:int(height * 1.5)]
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    output = cropped_image.copy()
    
    edged = cv2.Canny(blur_image, 30, 130)
    
    circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 0.5, 30,
                            param1=50, param2=30, minRadius=20, maxRadius=50)
    if circles is not None:
        # get the circles with largest radius
        circles = np.round(circles[0, :]).astype("int")
        circle_max = np.argmax(circles[:, 2])

        (x, y, r) = circles[circle_max]
        
        # cropped rectangle
        obj_cropped = output[y - r:y + r, x - r:x + r]
        
        return obj_cropped
    else:
        return output
    