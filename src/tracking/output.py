import json
from tracking.cell_detector import register_cells
from tracking.image import preprocess_image
from tracking.reader import read_image


def output_cell_centers(image_num: int):
    original = read_image(image_num)

    cropped, smoothed, threshed, contoured, contours = preprocess_image(original)

    cells, cell_contours = register_cells(contours)

    fout = open("../../resources/cell-centers-00%d.json" % image_num, "w")

    cell_dict = {"centers": [{"row": float(cell[0]), "col": float(cell[1])} for cell in cells]}

    json.dump(cell_dict, fout)


if __name__ == "__main__":
    output_cell_centers(1)
    output_cell_centers(2)
