import cv2
import numpy as np
from tracking import cv_color


EDGE_CROP_SIZE = 5


def show_image(name: str, image: np.array):
    """
    Show an image.
    :param name: name of the image being displayed
    :param image: image to be displayed
    :return: none
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 800)
    cv2.moveWindow(name, 20, 20)

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess_image(image: np.array):
    """
    Pre-process an image by removing some noise.
    :param image: original image
    :return: versions of original image at different stages and any contours in the image
    """
    # crop image to remove excess whitespace border
    cropped = image[EDGE_CROP_SIZE:image.shape[0] - EDGE_CROP_SIZE, EDGE_CROP_SIZE:image.shape[1] - EDGE_CROP_SIZE]

    # smooth image
    smoothed = smooth_image(cropped)

    # binarize image to black and white
    max_color = cv_color.MAX_COLOR_DENSITY
    _, threshed = cv2.threshold(smoothed,  max_color * 0.4, max_color, cv2.THRESH_BINARY)

    # create contours of cells in binarized image
    contoured, contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return smoothed, contours


def smooth_image(image: np.array):
    """
    Smooths a gray-scale image of cells to reduce false positive contours.
    :param image: original image
    :return: smoothed image
    """
    gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bilateral_blurred = cv2.bilateralFilter(gaussian_blurred, 9, 75, 75)
    median_blurred = cv2.medianBlur(bilateral_blurred, 5)
    avg_blurred = cv2.blur(median_blurred, (5, 5))

    return avg_blurred


def get_image_corners(image: np.array) -> np.array:
    """
    Returns four corners of 2D image.
    1st +-------------+ 4th
        |             |
        |             |
    2nd +-------------+ 3rd
    :param image: image
    :return: four corners of image in specified order
    """
    r, c = image.shape
    return np.array([[0, 0], [r, 0], [r, c], [c, 0]]).astype("int32")
