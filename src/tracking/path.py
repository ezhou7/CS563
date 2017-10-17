import os
from tracking.path_util import PathUtil


class DataPath:
    @staticmethod
    def get_images_dir():
        return os.path.expanduser("~/Documents/Emory/Senior/CS563/hw2/2D_images/")

    @staticmethod
    def get_image_num(num: int=0):
        file_base_name = PathUtil.pad_zeros(num, 3)
        return DataPath.get_images_dir() + file_base_name + ".jpg"
