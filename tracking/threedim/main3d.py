from tracking.threedim.filter import ParticleFilter3D

from tracking.threedim.reader import read_frame
import numpy as np
import json
import scipy.io


NUM_PARTICLES = 100


def main():
    pfilter = ParticleFilter3D(NUM_PARTICLES, False)
    pfilter.track_cells_in_3d()

    # json_filepath = "/Users/ezhou7/Downloads/2d_blobs.json"
    #
    # with open(json_filepath, "r") as f:
    #     data = json.load(f)
    #
    # data_array = np.zeros(shape=(20, 100, 150, 4), dtype="float32")
    #
    # for i in range(20):
    #     time_step_str = str(i + 1)
    #     slices = data[time_step_str]
    #
    #     for j in range(100):
    #         slice_str = str(j + 1)
    #         slice_img = slices[slice_str]
    #
    #         xs = slice_img["xs"]
    #         ys = slice_img["ys"]
    #         areas = slice_img["areas"]
    #         colors = slice_img["colors"]
    #
    #         data_array[i, j, :len(xs), 0] = np.array(xs)
    #         data_array[i, j, :len(ys), 1] = np.array(ys)
    #         data_array[i, j, :len(areas), 2] = np.array(areas)
    #         data_array[i, j, :len(colors), 3] = np.array(colors)
    #
    # # print(data_array[5, 34])
    #
    # scipy.io.savemat("/Users/ezhou7/Downloads/2d_blobs.mat", mdict={"data_array": data_array})


if __name__ == "__main__":
    main()
