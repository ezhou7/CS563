import os
import cv2
from tracking.path import DataPath


def read_image(image_num: int):
    """
    Read specified image.
    :param image_num: Which image to read.
    :return: image
    """
    path = DataPath.get_image_num(image_num)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def read_images():
    """
    Read all images from directory.
    :return: list of images
    """
    image_dir = DataPath.get_images_dir()
    image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]

    return [cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) for image_file in image_files]
