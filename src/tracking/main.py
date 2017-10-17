import cv2
from tracking import color
from tracking.cell_detector import find_cells
from tracking.debug import show_image
from tracking.draw import draw_cell
from tracking.image import smooth_image
from tracking.path import DataPath
from tracking.state import initial_state_from_cell


def process_image_file(filepath: str):
    # read original image into memory
    original = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # smooth image
    smoothed = smooth_image(original)

    # binarize image to black and white
    _, threshed = cv2.threshold(smoothed, color.MAX_COLOR_DENSITY / 2 + 1, color.MAX_COLOR_DENSITY, cv2.THRESH_BINARY)

    # create contours of cells in binarized image
    contoured_image, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # detect cells in image
    cells_image, cells, cell_contours = find_cells(contoured_image, contours)


def main():
    # sample image path
    path = DataPath.get_image_num(1)

    # read original image into memory
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    cropped = original[5:original.shape[0] - 5, 5:original.shape[1] - 5]

    # smooth image
    smoothed = smooth_image(cropped)

    # binarize image to black and white
    _, threshed = cv2.threshold(smoothed, color.MAX_COLOR_DENSITY * 2 / 5, color.MAX_COLOR_DENSITY, cv2.THRESH_BINARY)

    # create contours of cells in binarized image
    contoured_image, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # detect cells in image
    cells_image, cells, cell_contours = find_cells(contoured_image, contours)

    print(len(cell_contours))

    # draw cell contours
    cv2.drawContours(cells_image, cell_contours, -1, (0, 0, 255), 3)

    # show image of drawn contours
    show_image("contoured_image", cells_image)

    # create sample cell particles
    cell_particles = initial_state_from_cell(cells_image.shape, cells[0], 100)

    # create color image for display
    particle_image = cv2.cvtColor(smoothed.copy(), cv2.COLOR_GRAY2BGR)

    # draw cell particles
    for particle in cell_particles:
        draw_cell(particle_image, particle, color.GREEN)

    draw_cell(particle_image, cells[0], color.RED)

    # display cell particles
    show_image("particle_image", particle_image)


if __name__ == "__main__":
    main()
