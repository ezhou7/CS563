import numpy as np
from scipy.stats import norm
from tracking import bcell
from tracking.motion import many_to_one_dist


WEIGHT_THRESHOLD = 0.38


def initial_state(image_dim: np.array, cell: np.array, num_particles: int):
    X = np.random.random_integers(0, image_dim[0], (num_particles,))
    Y = np.random.random_integers(0, image_dim[1], (num_particles,))

    radii = np.ones((num_particles,)) * cell[bcell.RADIUS_INDEX]
    weights = np.zeros((num_particles,))

    particles = [X, Y, radii, weights]

    return np.array(particles).T.astype("float32")


def update_state(particles: np.array, curr_cell: np.array):
    normal_distro = norm()

    cell_magnitude = np.linalg.norm(curr_cell[bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX], axis=0)
    dists = many_to_one_dist(particles, curr_cell)

    new_weights = [normal_distro.pdf(dist / cell_magnitude) for dist in dists]
    particles[:, bcell.WEIGHT_INDEX] = np.array(new_weights).astype("float32")


def prune_particles(particles: np.array, threshold: float=WEIGHT_THRESHOLD):
    return particles[particles[:, bcell.WEIGHT_INDEX] >= threshold, :]


def find_mean_particle(particles: np.array):
    position = np.mean(particles[:, bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX], axis=0)
    return np.concatenate([position, particles[0, bcell.RADIUS_INDEX:bcell.WEIGHT_INDEX]])


def populate_particles(pruned_particles: np.array, num_particles: int):
    mean = find_mean_particle(pruned_particles)[bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX]
    covar = np.cov(pruned_particles[:, bcell.X_POS_INDEX], pruned_particles[:, bcell.Y_POS_INDEX])

    positions = np.random.multivariate_normal(mean, covar, size=num_particles)

    radii = np.ones((num_particles, 1)) * pruned_particles[0, bcell.RADIUS_INDEX]
    weights = np.zeros((num_particles, 1))

    return np.concatenate([positions, radii, weights], axis=1)

