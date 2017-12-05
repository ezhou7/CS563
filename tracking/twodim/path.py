import os

from tracking.twodim.pathutil import PathUtil


class DataPath:
    @staticmethod
    def get_2d_images_dir():
        return os.path.expanduser("~/Documents/Emory/Senior/CS563/project/2D_images/")

    @staticmethod
    def get_2d_image_num(num: int=0):
        file_base_name = PathUtil.pad_zeros(num, 3)
        return DataPath.get_2d_images_dir() + file_base_name + ".jpg"

    @staticmethod
    def get_3d_images_dir():
        return os.path.expanduser("~/Documents/Emory/Senior/CS563/project/3D_images/")

    @staticmethod
    def get_3d_image(time_step: int, z_slice: int):
        time_step_str = PathUtil.pad_zeros(time_step, 2)
        z_slice_str = PathUtil.pad_zeros(z_slice, 3)

        file_base_name = "t={}_z={}".format(time_step_str, z_slice_str)
        return DataPath.get_3d_images_dir() + file_base_name + ".jpg"
