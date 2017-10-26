import cv2
import numpy as np
from typing import List, Tuple
from tracking import cv_color
from tracking import bcell


def display_particle(image: np.array, particle: np.array, color: Tuple[int, int, int]):
    display_cell(image, particle, color)


def display_cell(image: np.array, cell: np.array, cell_color: Tuple[int, int, int]):
    center = tuple(cell[bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX].astype("int32").tolist())
    radius = cell[bcell.RADIUS_INDEX].astype("int32")

    cv2.circle(image, center, radius, cell_color, thickness=bcell.DEFAULT_THICKNESS)


def display_particles(image: np.array, particles: np.array,  color: Tuple[int, int, int]):
    for particle in particles:
        display_particle(image, particle, color)


def display_cells(image: np.array, cells: np.array, color: Tuple[int, int, int]):
    for cell in cells:
        display_cell(image, cell, color)


def display_cell_contours(image: np.array, cell_contours: List[np.array], cell_idx: int):
    cv2.drawContours(image, cell_contours, cell_idx, cv_color.RED, thickness=bcell.CONTOUR_THICKNESS)


def display_rect_bounds(image: np.array, cell_contours: List[np.array], cell_idx: int=-1):
    def display_bound(contour: np.array):
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + width, y + height), cv_color.GREEN, thickness=2)

    if cell_idx == -1:
        for cell_contour in cell_contours:
            display_bound(cell_contour)
    else:
        display_bound(cell_contours[cell_idx])


def display_circle_bounds(image: np.array, cell_contours: List[np.array], cell_idx: int=-1):
    def display_bound(contour: np.array):
        center, radius = cv2.minEnclosingCircle(contour)

        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        # show circle bound on cell
        cv2.circle(image, center, radius, cv_color.GREEN, thickness=2)

    if cell_idx == -1:
        for cell_contour in cell_contours:
            display_bound(cell_contour)
    else:
        display_bound(cell_contours[cell_idx])
