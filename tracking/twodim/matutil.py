import numpy as np

from tracking.twodim import bcell


def many_to_many_dists(group1: np.array, group2: np.array) -> np.array:
    p1 = group1[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]
    p2 = group2[:, bcell.BEG_POS_INDEX:bcell.END_POS_INDEX]

    p1 = p1.reshape((p1.shape[0], p1.shape[1], 1))
    p2 = p2.reshape((p2.shape[0], p2.shape[1], 1))

    p1_repmat = np.tile(p1, (1, 1, p2.shape[0]))
    p2_repmat = np.tile(p2, (1, 1, p1.shape[0])).swapaxes(0, 2)

    return np.linalg.norm(p1_repmat - p2_repmat, axis=1).astype("float32")
