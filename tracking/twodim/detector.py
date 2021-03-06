import cv2
import numpy as np
from typing import List


def register_cells(contours: List[np.array]) -> np.array:
    """
    return cell positions from given contour information
    :param contours: contours of cells
    :return: array of cells
    """
    def create_cell(contour: np.array) -> np.array:
        center, radius = cv2.minEnclosingCircle(contour)
        v = (0, 0)
        weight = -1

        return np.array([center[0], center[1], v[0], v[1], radius, weight])

    areas = np.array([cv2.contourArea(contour) for contour in contours]).astype("float32")
    blobs = np.array([create_cell(contour) for contour in contours]).astype("int32")

    return blobs[areas >= 100].astype("float32")
