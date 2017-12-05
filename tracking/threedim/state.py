import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from tracking.threedim import bcell
from tracking.threedim.image import get_frame_corners


def initial_state(frame_dim: np.array, cells: np.array, num_particles: int):
    total_num_particles = num_particles * cells.shape[0]

    X = np.random.random_integers(0, frame_dim[1], (total_num_particles,))
    Y = np.random.random_integers(0, frame_dim[2], (total_num_particles,))
    Z = np.random.random_integers(0, frame_dim[0], (total_num_particles,))

    VX = np.zeros(shape=(total_num_particles,))
    VY = np.zeros(shape=(total_num_particles,))
    VZ = np.zeros(shape=(total_num_particles,))

    radii = np.ones((num_particles, cells.shape[0])) * np.tile(cells[:, bcell.RADIUS_INDEX], (num_particles, 1))
    radii = radii.T.reshape((total_num_particles,))

    weights = np.zeros((total_num_particles,))

    particles = [X, Y, Z, VX, VY, VZ, radii, weights]

    return np.array(particles).T.reshape((cells.shape[0], num_particles, len(particles))).astype("float32")


def update_state(frame: np.array, curr_cells: np.array, particles_all: np.array):
    observations_all = get_dists_from_landmarks(frame, curr_cells)
    observation_noise = 100

    particles_all[:, :, bcell.WEIGHT_INDEX] = 1
    # print(particles_all[5, :, bcell.WEIGHT_INDEX])

    i = 0
    for particles, observations in zip(particles_all, observations_all):
        weights = particles[:, bcell.WEIGHT_INDEX]

        particle_dists = get_dists_from_landmarks(frame, particles)

        preds = stats.norm(particle_dists, observation_noise).pdf(observations)
        weights *= np.prod(preds, axis=1).astype("float32")
        # if i == 5:
        #     print(particle_dists)
        #     print(observations)
            # print(np.prod(preds, axis=1).astype("float32"))

        weights /= np.sum(weights).astype("float32")
        # if i == 5:
        #     print(weights)
        i += 1
        # print(weights)


def get_dists_from_landmarks(frame: np.array, cells: np.array):
    corners = get_frame_corners(frame)

    corners_mat = np.tile(corners, (cells.shape[0], 1))
    corners_mat = corners_mat.reshape((cells.shape[0], corners.shape[0], -1))

    cells_mat = np.tile(cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX], (corners.shape[0], 1, 1))
    cells_mat = np.swapaxes(cells_mat, 0, 1)
    cells_mat = cells_mat.reshape((cells.shape[0], corners.shape[0], -1))

    return np.linalg.norm(corners_mat - cells_mat, axis=2)


def resample_state(particles_all: np.array):
    new_particles_all = []
    # print(particles_all[5, :, bcell.WEIGHT_INDEX])

    # k = 0
    for particles in particles_all:
        num_particles = particles.shape[0]
        weights = particles[:, bcell.WEIGHT_INDEX]
        # print(k)
        # k += 1
        # print(weights)

        marks = (np.random.rand() + np.arange(num_particles)) / num_particles
        weights_cdf = np.cumsum(weights)

        new_particles = np.zeros(particles.shape).astype("float32")

        i, j = 0, 0
        while i < num_particles:
            # print(i, j)
            if marks[i] < weights_cdf[j]:
                new_particles[i] = particles[j]
                i += 1
            else:
                j += 1

        new_weights = new_particles[:, bcell.WEIGHT_INDEX]
        new_weights /= np.sum(new_weights).astype("float32")

        new_particles_all.append(new_particles)

    return np.array(new_particles_all).astype("float32")


def predict_state(particles_all):
    x_pos = particles_all[:, :, bcell.X_POS_INDEX]
    y_pos = particles_all[:, :, bcell.Y_POS_INDEX]
    z_pos = particles_all[:, :, bcell.Z_POS_INDEX]

    weights = particles_all[:, :, bcell.WEIGHT_INDEX]

    x_avg = np.average(x_pos, weights=weights, axis=1).astype("int32")
    x_avg = x_avg.reshape((x_avg.shape[0], 1))

    y_avg = np.average(y_pos, weights=weights, axis=1).astype("int32")
    y_avg = y_avg.reshape((y_avg.shape[0], 1))

    z_avg = np.average(z_pos, weights=weights, axis=1).astype("int32")
    z_avg = z_avg.reshape((z_avg.shape[0], 1))

    return np.hstack((x_avg, y_avg, z_avg))


def display_state(figure, curr_cells: np.array, predictions):
    # figure.clear()

    # figure.scatter(curr_cells[:, 0], curr_cells[:, 1], curr_cells[:, 2], color="red")
    # figure.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], color="blue")

    # n = 9
    # figure.scatter(curr_cells[n, 0], curr_cells[n, 1], curr_cells[n, 2], color="red")
    # figure.scatter(predictions[n, 0], predictions[n, 1], predictions[n, 2], color="blue")

    plt.draw()
    plt.pause(0.5)
