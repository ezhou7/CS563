import cv2
import numpy as np
from typing import List, Tuple
from tracking import cv_color
from tracking import bcell


def display_particle(image: np.array, particle: np.array):
    center = (particle[bcell.X_POS_INDEX].astype("int32"), particle[bcell.Y_POS_INDEX].astype("int32"))
    cv2.circle(image, center, bcell.PARTICLE_RADIUS, bcell.PARTICLE_COLOR, thickness=bcell.DEFAULT_THICKNESS)


def display_prediction(image: np.array, prediction: np.array):
    center = (prediction[bcell.X_POS_INDEX].astype("int32"), prediction[bcell.Y_POS_INDEX].astype("int32"))

    cv2.circle(image, center, bcell.PREDICTION_RADIUS, bcell.PREDICTION_COLOR, thickness=bcell.DEFAULT_THICKNESS)


def display_cell(image: np.array, cell: np.array):
    center = (cell[bcell.X_POS_INDEX].astype("int32"), cell[bcell.Y_POS_INDEX].astype("int32"))
    radius = cell[bcell.RADIUS_INDEX].astype("int32")

    cv2.circle(image, center, radius, bcell.CELL_COLOR, thickness=bcell.DEFAULT_THICKNESS)


def display_particles(image: np.array, particles: np.array):
    for particle in particles:
        display_particle(image, particle)


def display_cells(image: np.array, cells: np.array):
    for cell in cells:
        display_cell(image, cell)


def display_predictions(image: np.array, predictions: np.array):
    for prediction in predictions:
        display_prediction(image, prediction)


def display_particles_all(image: np.array, particles_all: np.array):
    for particles in particles_all:
        display_particles(image, particles)


def display_cell_contours(image: np.array, cell_contours: List[np.array], cell_idx: int):
    cv2.drawContours(image, cell_contours, cell_idx, cv_color.RED, thickness=bcell.CONTOUR_THICKNESS)


def display_rect_bounds(image: np.array, cell_contours: List[np.array], cell_idx: int=-1):
    def display_bound(contour: np.array):
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + width, y + height), cv_color.GREEN, thickness=bcell.DEFAULT_THICKNESS)

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
        cv2.circle(image, center, radius, cv_color.GREEN, thickness=bcell.DEFAULT_THICKNESS)

    if cell_idx == -1:
        for cell_contour in cell_contours:
            display_bound(cell_contour)
    else:
        display_bound(cell_contours[cell_idx])
