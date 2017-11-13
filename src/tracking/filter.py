import cv2
import numpy as np

from sklearn.utils.linear_assignment_ import _hungarian

from tracking import bcell

from tracking.detector import register_cells
from tracking.draw import display_cells, display_predictions
from tracking.image import preprocess_image, show_image
from tracking.motion import many_to_one_dist, get_velocities, move_particles, move
from tracking.reader import read_image
from tracking.state import initialize_state, update_state, resample, get_dists_from_landmarks


NUM_PARTICLES = 100


def is_empty(array: np.array):
    return True if array.shape[0] == 0 else False


def expected_value(particles_all):
    x_pos = particles_all[:, :, bcell.X_POS_INDEX]
    y_pos = particles_all[:, :, bcell.Y_POS_INDEX]

    weights = particles_all[:, :, bcell.WEIGHT_INDEX]

    x_avg = np.average(x_pos, weights=weights, axis=1).astype("int32")
    x_avg = x_avg.reshape((x_avg.shape[0], 1))

    y_avg = np.average(y_pos, weights=weights, axis=1).astype("int32")
    y_avg = y_avg.reshape((y_avg.shape[0], 1))

    return np.hstack((x_avg, y_avg))


def create_shadows_if_overlap(prev_cells, assignments):
    prev_assignments = assignments[:, 1]

    prev_set = set([i for i in range(len(prev_cells))])
    assigned_set = set(prev_assignments.tolist())
    diff_set = prev_set - assigned_set

    new_assignments = assignments.tolist()

    i = prev_assignments.shape[0]

    for unassigned in diff_set:
        new_assignments.append([i, unassigned])
        i += 1

    return np.array(new_assignments), diff_set


def track_cells():
    prev_cells = np.zeros((0,))
    particles_all = np.zeros((0,))

    display1 = read_image(1)
    display2 = cv2.cvtColor(display1, cv2.COLOR_GRAY2BGR)

    for i in range(20):
        original = read_image(i + 1)
        smoothed, contours = preprocess_image(original)

        cells = register_cells(contours)

        if is_empty(prev_cells):
            prev_cells = cells
            particles_all = initialize_state(smoothed.shape, cells, num_particles=NUM_PARTICLES)

            continue

        cost_matrix = np.array([many_to_one_dist(prev_cells, cell) for cell in cells])

        # current cells, previous cells
        assignments = _hungarian(cost_matrix)

        if prev_cells.shape[0] > assignments.shape[0]:
            assignments, unassigned_set = create_shadows_if_overlap(prev_cells, assignments)

            cells_list = cells.tolist()

            for unassigned in unassigned_set:
                move(prev_cells[unassigned])
                cells_list.append(prev_cells[unassigned].tolist())

            cells = np.array(cells_list)

        prev_cells = prev_cells[assignments[:, 1]]
        particles_all = particles_all[assignments[:, 1]]

        velocities = get_velocities(prev_cells, cells)
        cells[:, bcell.BEG_VEL_INDEX:bcell.END_VEL_INDEX] = velocities
        velocity_noise = np.random.randn(particles_all.shape[0], particles_all.shape[1], bcell.END_POS_INDEX) * 30

        move_particles(particles_all, velocities, velocity_noise)

        observations_all = get_dists_from_landmarks(smoothed, cells)
        observation_noise = 200

        update_state(smoothed, particles_all, observations_all, observation_noise)

        particles_all = resample(particles_all)
        mean_centers = expected_value(particles_all)

        canvas = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)

        # display_particles_all(canvas, particles_all)
        display_predictions(canvas, mean_centers)
        display_cells(canvas, cells)

        show_image("smoothed image", canvas)

        prev_cells = cells
