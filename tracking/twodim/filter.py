import numpy as np
from sklearn.utils.linear_assignment_ import _hungarian
from tracking.twodim.detector import register_cells
from tracking.twodim.image import preprocess_image
from tracking.twodim.matutil import many_to_many_dists
from tracking.twodim.reader import read_image_2d
from tracking.twodim.state import initial_state, update_state, resample_state, predict_state, display_state, write_state

from tracking.twodim import bcell


class ParticleFilter2D:
    def __init__(self, num_particles: int):
        self._num_particles = num_particles

        self._prev_cells = None
        self._curr_cells = None
        self._particles_all = None

        self._initialize_filter()

    def _initialize_filter(self):
        original = read_image_2d(1)
        smoothed, contours = preprocess_image(original)

        self._curr_cells = register_cells(contours)
        self._prev_cells = self._curr_cells
        self._particles_all = initial_state(smoothed.shape, self._curr_cells, num_particles=self._num_particles)

    def _create_shadow_cells(self, assignments):
        """
        If cells become indistinguishable from each other, create "shadow cells" for estimation

        :param assignments: assignments from hungarian algorithm matching current and previous cells
        :return: prev cells, curr cells and particles matched with each other
        """
        prev_assignments = assignments[:, 1]
        num_assigned = prev_assignments.shape[0]

        prev_set = {i for i in range(len(self._prev_cells))}
        assigned_set = set(prev_assignments)

        diff_set = prev_set - assigned_set

        new_assignments = np.zeros(shape=(self._prev_cells.shape[0],)).astype("int32")
        new_assignments[:num_assigned] = prev_assignments

        new_curr_cells = np.zeros(shape=self._prev_cells.shape)
        new_curr_cells[:self._curr_cells.shape[0]] = self._curr_cells

        i = num_assigned

        for unassigned in diff_set:
            new_assignments[i] = unassigned
            new_curr_cells[i] = self._prev_cells[unassigned]

            # move shadow cell (pos += vel)
            new_curr_cells[i, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] += \
                new_curr_cells[i, bcell.BEG_VEL_INDEX:bcell.END_VEL_INDEX]

            i += 1

        self._prev_cells = self._prev_cells[new_assignments]
        self._curr_cells = new_curr_cells
        self._particles_all = self._particles_all[new_assignments]

    def _sync_time_steps(self):
        # current cells, previous cells
        cost_matrix = many_to_many_dists(self._curr_cells, self._prev_cells)

        # current cells, previous cells
        assignments = _hungarian(cost_matrix)

        # if cells are indistinguishable
        if self._prev_cells.shape[0] > assignments.shape[0]:
            self._create_shadow_cells(assignments)
        else:
            self._prev_cells = self._prev_cells[assignments[:, 1]]
            self._particles_all = self._particles_all[assignments[:, 1]]

    def _move_particles(self):
        velocities = self._curr_cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] - \
                     self._prev_cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]
        velocity_noise = np.random.randn(self._particles_all.shape[0], self._particles_all.shape[1], bcell.END_POS_INDEX) * 30

        noisy_velocities = velocities.reshape(velocities.shape[0], 1, velocities.shape[1]) + velocity_noise

        self._curr_cells[:, bcell.BEG_VEL_INDEX:bcell.END_VEL_INDEX] = velocities
        self._particles_all[:, :, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] += noisy_velocities

    def _predict(self, time_step: int):
        original = read_image_2d(time_step + 1)
        smoothed, contours = preprocess_image(original)

        self._curr_cells = register_cells(contours)

        self._sync_time_steps()
        self._move_particles()

        update_state(smoothed, self._curr_cells, self._particles_all)

        self._particles_all = resample_state(self._particles_all)
        predictions = predict_state(self._particles_all)

        display_state(smoothed, self._curr_cells, predictions)
        # write_state(time_step, smoothed, self._curr_cells, predictions)

        self._prev_cells = self._curr_cells

    def track_cells_in_2d(self):
        for i in range(1, 20):
            self._predict(i)
