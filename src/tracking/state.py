import numpy as np
from scipy import stats
from typing import List, Tuple
from tracking.cell import Cell, CellParticle


class State:
    def __init__(self, prev_particles: List[Tuple[Cell, float]]):
        pass

    def update_weights(self):
        pass


def initial_state_from_cell(image_2d_dims, cell: Cell, size: int):
    X = np.random.random_integers(0, image_2d_dims[0], (size, 1))
    Y = np.random.random_integers(0, image_2d_dims[1], (size, 1))

    radius = cell.get_radius()
    V = np.random.random_integers(-radius, radius, size=(size, 2))

    return [CellParticle(center=(x, y), radius=cell.get_radius(), velocity=(v[0], v[1])) for x, y, v in zip(X, Y, V)]


def update_state(curr_cell, prev_particles):
    curr_center = curr_cell.get_center()
    curr_velocity = curr_cell.get_velocity()

    for particle in prev_particles:
        prev_center = particle.get_center()
        prev_velocity = particle.get_velocity()

        row_magnitude_center = curr_center[0] - prev_center[0]
        col_magnitude_center = curr_center[1] - prev_center[0]
        center_salience = np.sqrt(row_magnitude_center ** 2 + col_magnitude_center ** 2)

        row_magnitude_velocity = curr_velocity[0] - prev_velocity[0]
        col_magnitude_velocity = curr_velocity[1] - prev_velocity[1]
        velocity_salience = np.sqrt(row_magnitude_velocity ** 2 + col_magnitude_velocity ** 2)

        salience = center_salience * velocity_salience

        prob = stats.norm()
