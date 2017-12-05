import cv2
import json
import numpy as np

from tracking.threedim.path import DataPath


def read_image_3d(time_step: int, z_slice: int):
    path = DataPath.get_3d_image(time_step, z_slice)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def read_3d_images_with_time(time_step: int):
    return [cv2.imread(DataPath.get_3d_image(time_step, z + 1), cv2.IMREAD_GRAYSCALE) for z in range(100)]


def read_frame(time_step: int):
    return np.array(read_3d_images_with_time(time_step))


def read_3d_cells_json():
    with open(DataPath.get_3d_cells_path(), "r") as fin:
        cells_3d_data = json.load(fin)

    cells_3d = []

    for i in range(20):
        time_step_str = str(i + 1)
        cells_in_frame_json = cells_3d_data[time_step_str]

        xs = cells_in_frame_json["xs"]
        ys = cells_in_frame_json["ys"]
        zs = cells_in_frame_json["zs"]

        cells_in_frame = np.array([xs, ys, zs], dtype="float32").T

        cells_3d.append(cells_in_frame)

    return cells_3d
