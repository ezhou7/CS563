import numpy as np
from tracking import bcell


def many_to_one_dist(particles: np.array, cell: np.array) -> np.array:
    cell_vec_field = np.tile(cell[bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX], (particles.shape[bcell.X_POS_INDEX], 1))
    particle_centers = particles[:, bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX]

    return np.linalg.norm(particle_centers - cell_vec_field, axis=1).astype("float32")


def move_particles(particles: np.array, move_vec: np.array) -> np.array:
    move_vec_field = np.tile(move_vec, (particles.shape[bcell.X_POS_INDEX], 1))
    particle_centers = particles[:, bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX]

    particle_centers += move_vec_field
