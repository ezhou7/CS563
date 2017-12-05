from typing import List

import cv2
import numpy as np

from tracking.threedim import bcell
from tracking.threedim import stain


def draw_particle(image: np.array, particle: np.array):
    center = (particle[bcell.X_POS_INDEX].astype("int32"), particle[bcell.Y_POS_INDEX].astype("int32"))
    cv2.circle(image, center, bcell.PARTICLE_RADIUS, bcell.PARTICLE_COLOR, thickness=bcell.DEFAULT_THICKNESS)


def draw_prediction(image: np.array, prediction: np.array):
    center = (prediction[bcell.X_POS_INDEX].astype("int32"), prediction[bcell.Y_POS_INDEX].astype("int32"))

    cv2.circle(image, center, bcell.PREDICTION_RADIUS, bcell.PREDICTION_COLOR, thickness=bcell.DEFAULT_THICKNESS)


def draw_cell(image: np.array, cell: np.array):
    center = (cell[bcell.X_POS_INDEX].astype("int32"), cell[bcell.Y_POS_INDEX].astype("int32"))
    radius = cell[bcell.RADIUS_INDEX].astype("int32")

    cv2.circle(image, center, radius, bcell.CELL_COLOR, thickness=bcell.DEFAULT_THICKNESS)


def draw_particles(image: np.array, particles: np.array):
    for particle in particles:
        draw_particle(image, particle)


def draw_cells(image: np.array, cells: np.array):
    for cell in cells:
        draw_cell(image, cell)


def draw_predictions(image: np.array, predictions: np.array):
    for prediction in predictions:
        draw_prediction(image, prediction)


def draw_particles_all(image: np.array, particles_all: np.array):
    for particles in particles_all:
        draw_particles(image, particles)


def draw_cell_contours(image: np.array, cell_contours: List[np.array], cell_idx: int):
    cv2.drawContours(image, cell_contours, cell_idx, stain.RED, thickness=bcell.CONTOUR_THICKNESS)


def draw_rect_bounds(image: np.array, cell_contours: List[np.array], cell_idx: int=-1):
    def display_bound(contour: np.array):
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + width, y + height), stain.GREEN, thickness=bcell.DEFAULT_THICKNESS)

    if cell_idx == -1:
        for cell_contour in cell_contours:
            display_bound(cell_contour)
    else:
        display_bound(cell_contours[cell_idx])


def draw_circle_bounds(image: np.array, cell_contours: List[np.array], cell_idx: int=-1):
    def display_bound(contour: np.array):
        center, radius = cv2.minEnclosingCircle(contour)

        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        # show circle bound on cell
        cv2.circle(image, center, radius, stain.GREEN, thickness=bcell.DEFAULT_THICKNESS)

    if cell_idx == -1:
        for cell_contour in cell_contours:
            display_bound(cell_contour)
    else:
        display_bound(cell_contours[cell_idx])
