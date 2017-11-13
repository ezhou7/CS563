import numpy as np
from scipy import stats

from tracking import bcell

from tracking.image import get_image_corners


def initialize_state(image_dim: np.array, cells: np.array, num_particles: int):
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


def update_state(image: np.array, particles_all: np.array, observations_all, noise):
    particles_all[:, :, bcell.WEIGHT_INDEX] = 1

    for particles, observations in zip(particles_all, observations_all):
        weights = particles[:, bcell.WEIGHT_INDEX]

        particle_dists = get_dists_from_landmarks(image, particles)

        preds = stats.norm(particle_dists, noise).pdf(observations)
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


def resample(particles_all: np.array):
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
