import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils.linear_assignment_ import _hungarian

from tracking.threedim.matutil import many_to_many_dists
from tracking.threedim.reader import read_frame, read_3d_cells_json
from tracking.threedim.state import initial_state, update_state, resample_state, predict_state, display_state

from tracking.threedim import bcell


def get_background_truth():
    path = "/Users/ezhou7/Documents/Emory/Senior/CS563/project/3D-truth/3d_t_tracking.mat"
    return spio.loadmat(path)["sTrue"][0]


class ParticleFilter3D:
    def __init__(self, num_particles: int, use_truth: bool):
        self._num_particles = num_particles
        self._use_truth = use_truth

        self._mat = get_background_truth() if use_truth else read_3d_cells_json()
        self._truth = get_background_truth()

        self._prev_cells = None
        self._curr_cells = None
        self._particles_all = None

        self._figure = None
        self._frame_dim = None

        self._initialize_filter()

    def _initialize_filter(self):
        # original = read_frame(1)
        # smoothed, contours = preprocess_image(original)
        #
        # self._curr_cells = register_cells(contours)
        # self._prev_cells = self._curr_cells
        # self._particles_all = initial_state(smoothed.shape, self._curr_cells, num_particles=self._num_particles)

        original = read_frame(1)

        self._curr_cells = np.zeros((len(self._mat[0]), bcell.MAX_INDICES))
        self._prev_cells = self._curr_cells
        self._particles_all = initial_state(original.shape, self._curr_cells, self._num_particles)

        self._curr_cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] = self._mat[0][:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]

        fig = plt.figure()
        self._figure = Axes3D(fig)

        frame_dim = original.shape
        self._frame_dim = frame_dim

        self._figure.set_xlim(0, frame_dim[1])
        self._figure.set_ylim(0, frame_dim[2])
        self._figure.set_zlim(0, frame_dim[0])

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
        # if self._use_truth:
        #     # current cells, previous cells
        #     cost_matrix = many_to_many_dists(self._curr_cells, self._prev_cells)
        #
        #     # current cells, previous cells
        #     assignments = _hungarian(cost_matrix)
        #
        #     # if cells are indistinguishable, create "shadow" cells as temporary replacements
        #     if self._prev_cells.shape[0] > assignments.shape[0]:
        #         self._create_shadow_cells(assignments)
        #     else:
        #         self._prev_cells = self._prev_cells[assignments[:, 1]]
        #         self._particles_all = self._particles_all[assignments[:, 1]]
        # else:

        # new_prev_cells = np.zeros(shape=self._curr_cells.shape, dtype="float32")
        # new_prev_cells[:self._prev_cells.shape[0]] = self._prev_cells
        # self._prev_cells = new_prev_cells

        # # previous cells, current cells
        # cost_matrix = many_to_many_dists(self._prev_cells, self._curr_cells)
        #
        # # previous cells, current cells
        # assignments = _hungarian(cost_matrix)
        # print(self._prev_cells.shape, self._curr_cells.shape)
        # print(assignments)
        # num_assigned = assignments.shape[0]

        # if cells are indistinguishable, create "shadow" cells as temporary replacements
        if self._curr_cells.shape[0] < self._prev_cells.shape[0]:
            cost_matrix = many_to_many_dists(self._curr_cells, self._prev_cells)
            assignments = _hungarian(cost_matrix)

            self._create_shadow_cells(assignments)
        else:
            # previous cells, current cells
            cost_matrix = many_to_many_dists(self._prev_cells, self._curr_cells)

            # previous cells, current cells
            assignments = _hungarian(cost_matrix)
            # print(self._prev_cells.shape, self._curr_cells.shape)
            # print(assignments)
            num_assigned = assignments.shape[0]

            if self._curr_cells.shape[0] > self._prev_cells.shape[0]:
                mask = np.zeros(shape=(self._curr_cells.shape[0],), dtype="bool")
                mask[assignments[:, 1]] = True
                # print(self._curr_cells[~mask])
                # print(self._prev_cells.shape, self._curr_cells.shape)

                new_curr_cells = np.zeros(shape=self._prev_cells.shape, dtype="float32")
                new_curr_cells[:num_assigned] = self._curr_cells[assignments[:, 1]]
                # new_curr_cells[num_assigned:] = self._curr_cells[~mask]

                self._curr_cells = new_curr_cells
                # print(self._curr_cells)
            else:
                self._curr_cells = self._curr_cells[assignments[:, 1]]
                # self._particles_all = self._particles_all[assignments[:, 1]]

        # print(self._curr_cells)

    def _move_particles(self):
        length = self._prev_cells.shape[0]

        velocities_curr = np.zeros((self._curr_cells.shape[0], bcell.DIMS))

        velocities_diff = self._curr_cells[:length, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] - \
                          self._prev_cells[:length, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]

        # if np.all(velocities_diff == 0):
        #     print(self._curr_cells is self._prev_cells)
        #     # velocities_diff = np.ones(velocities_diff.shape, dtype="float32")

        velocities_prev = self._prev_cells[:length, bcell.BEG_VEL_INDEX:bcell.END_VEL_INDEX]

        velocities_curr[:length] = velocities_diff

        velocity_noise = np.random.randn(self._particles_all.shape[0], self._particles_all.shape[1], bcell.END_POS_INDEX) * 30

        noisy_velocities = velocities_prev.reshape(velocities_prev.shape[0], 1, velocities_prev.shape[1]) + velocity_noise

        # print(velocities_diff)

        self._curr_cells[:, bcell.BEG_VEL_INDEX:bcell.END_VEL_INDEX] = velocities_curr
        self._particles_all[:, :, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] += noisy_velocities

        new_particles_all = initial_state(self._frame_dim, self._curr_cells, self._num_particles)
        new_particles_all[:length] = self._particles_all
        self._particles_all = new_particles_all

    def _predict(self, time_step: int):
        # original = read_image_2d(time_step + 1)
        # smoothed, contours = preprocess_image(original)
        #
        # self._curr_cells = register_cells(contours)
        #
        # self._sync_time_steps()
        # self._move_particles()
        #
        # update_state(smoothed, self._curr_cells, self._particles_all)
        #
        # self._particles_all = resample_state(self._particles_all)
        # predictions = predict_state(self._particles_all)
        #
        # display_state(smoothed, self._curr_cells, predictions)
        #
        # self._prev_cells = self._curr_cells

        original = read_frame(time_step + 1)

        num_detected = self._mat[time_step].shape[0]

        if num_detected > self._curr_cells.shape[0]:
            new_curr_cells = np.zeros((len(self._mat[time_step]), bcell.MAX_INDICES))
            new_curr_cells[:self._curr_cells.shape[0]] = self._curr_cells
            self._curr_cells = new_curr_cells

        self._curr_cells[:num_detected, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX] = self._mat[time_step][:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]
        # print(time_step)
        # t = 9
        #
        # if time_step == t:
        #     print(1)
        #     print(self._curr_cells)

        self._sync_time_steps()

        # if time_step == t:
        #     print(2)
        #     print(self._curr_cells)
        #     print(2.5)
        #     print(self._prev_cells[:, :3] == self._mat[9][:26, :3])

        self._move_particles()

        # if time_step == t:
        #     print(3)
        #     print(self._curr_cells)

        # print(self._particles_all[5, :, bcell.WEIGHT_INDEX])

        update_state(original, self._curr_cells, self._particles_all)
        # print(self._curr_cells.shape)
        # print(self._particles_all[5, :, bcell.WEIGHT_INDEX])
        # if time_step == 2:
        #     length = self._prev_cells.shape[0]
        #     print(self._curr_cells.shape[0], length)
        #     self._figure.scatter(self._curr_cells[length:, 0], self._curr_cells[length:, 1], self._curr_cells[length:, 2], color="red")
        #     plt.pause(10)

        self._particles_all = resample_state(self._particles_all)
        predictions = predict_state(self._particles_all)
        # print(self._curr_cells.shape, predictions.shape)

        # print(get_dists(self._curr_cells[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX], predictions))
        # print(time_step, target_effectiveness(self._curr_cells, predictions))

        display_state(self._figure, self._curr_cells, predictions)

        self._prev_cells = self._curr_cells

    def track_cells_in_3d(self):
        for i in range(1, 20):
            self._predict(i)


def get_dists(cells1: np.array, cells2: np.array):
    return np.linalg.norm(cells1 - cells2, axis=1)


def target_effectiveness(truth: np.array, predictions: np.array):
    dists = get_dists(truth[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX], predictions)
    dists = dists[dists <= 20]

    print(dists.shape[0], truth.shape[0])

    return dists.shape[0] / truth.shape[0]
