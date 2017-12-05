import cv2
import numpy as np
from scipy import stats
from tracking.twodim.draw import draw_cells, draw_predictions
from tracking.twodim.image import get_image_corners, show_image
from tracking.twodim.pathutil import PathUtil

from tracking.twodim import bcell


def initial_state(image_dim: np.array, cells: np.array, num_particles: int):
    total_num_particles = num_particles * cells.shape[0]

    X = np.random.random_integers(0, image_dim[0], (total_num_particles,))
    Y = np.random.random_integers(0, image_dim[1], (total_num_particles,))

    VX = np.zeros(shape=(total_num_particles,))
    VY = np.zeros(shape=(total_num_particles,))

    radii = np.ones((num_particles, cells.shape[0])) * np.tile(cells[:, bcell.RADIUS_INDEX], (num_particles, 1))
    radii = radii.T.reshape((total_num_particles,))

    weights = np.zeros((total_num_particles,))

    particles = [X, Y, VX, VY, radii, weights]

    return np.array(particles).T.reshape((cells.shape[0], num_particles, len(particles))).astype("float32")


def update_state(image: np.array, curr_cells: np.array, particles_all: np.array):
    observations_all = get_dists_from_landmarks(image, curr_cells)
    observation_noise = 200

    particles_all[:, :, bcell.WEIGHT_INDEX] = 1

    for particles, observations in zip(particles_all, observations_all):
        weights = particles[:, bcell.WEIGHT_INDEX]

        particle_dists = get_dists_from_landmarks(image, particles)

        preds = stats.norm(particle_dists, observation_noise).pdf(observations)
        weights *= np.prod(preds, axis=1).astype("float32")

        weights /= np.sum(weights)


def get_dists_from_landmarks(image: np.array, cells: np.array):
    corners = get_image_corners(image)

    corners_mat = np.tile(corners, (cells.shape[0], 1))
    corners_mat = corners_mat.reshape((cells.shape[0], corners.shape[0], -1))

    cells_mat = np.tile(cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX], (corners.shape[0], 1, 1))
    cells_mat = np.swapaxes(cells_mat, 0, 1)
    cells_mat = cells_mat.reshape((cells.shape[0], corners.shape[0], -1))

    return np.linalg.norm(corners_mat - cells_mat, axis=2)


def resample_state(particles_all: np.array):
    new_particles_all = []

    for particles in particles_all:
        num_particles = particles.shape[0]
        weights = particles[:, bcell.WEIGHT_INDEX]

        marks = (np.random.rand() + np.arange(num_particles)) / num_particles
        weights_cdf = np.cumsum(weights)

        new_particles = np.zeros(particles.shape).astype("float32")

        i, j = 0, 0
        while i < num_particles:
            if marks[i] < weights_cdf[j]:
                new_particles[i] = particles[j]
                i += 1
            else:
                j += 1

        new_weights = new_particles[:, bcell.WEIGHT_INDEX]
        new_weights /= np.sum(new_weights)

        new_particles_all.append(new_particles)

    return np.array(new_particles_all).astype("float32")


def predict_state(particles_all):
    x_pos = particles_all[:, :, bcell.X_POS_INDEX]
    y_pos = particles_all[:, :, bcell.Y_POS_INDEX]

    weights = particles_all[:, :, bcell.WEIGHT_INDEX]

    x_avg = np.average(x_pos, weights=weights, axis=1).astype("int32")
    x_avg = x_avg.reshape((x_avg.shape[0], 1))

    y_avg = np.average(y_pos, weights=weights, axis=1).astype("int32")
    y_avg = y_avg.reshape((y_avg.shape[0], 1))

    return np.hstack((x_avg, y_avg))


def display_state(image: np.array, curr_cells: np.array, predictions):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    draw_cells(canvas, curr_cells)
    draw_predictions(canvas, predictions)

    show_image("smoothed image", canvas)


def write_state(time_step: int, image: np.array, curr_cells: np.array, predictions):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    draw_cells(canvas, curr_cells)
    draw_predictions(canvas, predictions)

    cv2.imwrite("../../../data/output/twodim/" + PathUtil.pad_zeros(time_step + 1, 3) + ".jpg", canvas)
