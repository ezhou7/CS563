import os
import cv2
from tracking.path import DataPath


def read_images():
    image_dir = DataPath.get_images_dir()
    image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]

    return [cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) for image_file in image_files]


def smooth_image(image):
    """
    Smooths a gray-scale image of cells to reduce false positive contours
    :param image: original image
    :return: smoothed image
    """
    gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bilateral_blurred = cv2.bilateralFilter(gaussian_blurred, 9, 75, 75)
    median_blurred = cv2.medianBlur(bilateral_blurred, 5)
    avg_blurred = cv2.blur(median_blurred, (5, 5))

    return avg_blurred
