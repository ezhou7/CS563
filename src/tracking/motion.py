import numpy as np
from tracking import bcell


def many_to_one_dist(many: np.array, one: np.array) -> np.array:
    cell_vec_field = np.tile(one[bcell.BEG_POS_INDEX:bcell.END_POS_INDEX], (many.shape[bcell.X_POS_INDEX], 1))
    particle_centers = many[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]

    return np.linalg.norm(particle_centers - cell_vec_field, axis=1).astype("float32")


def get_velocities(prev_cells: np.array, curr_cells: np.array):
    return curr_cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] - prev_cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]


def move(obj: np.array):
    obj[bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] += obj[bcell.BEG_VEL_INDEX:bcell.END_VEL_INDEX]


def move_particles(particles: np.array, velocities: np.array, noise: np.array):
    noisy_velocities = velocities.reshape(velocities.shape[0], 1, velocities.shape[1]) + noise
    particles[:, :, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] += noisy_velocities
