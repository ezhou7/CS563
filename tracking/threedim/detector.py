import numpy as np
from typing import List

from tracking.twodim.detector import register_cells


def register_cells_3d(slices_of_contours: List[List[np.array]]):
    slices_of_cells = [register_cells(contours) for contours in slices_of_contours]

    for i, slice_of_cells in enumerate(slices_of_cells):
        slice_of_cells[:, 2] = i + 1

    return slices_of_cells
