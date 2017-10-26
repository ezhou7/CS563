import cv2
from tracking import bcell
from tracking import cv_color
from tracking.cell_detector import register_cells
from tracking.draw import display_cell, display_cells, display_particles
from tracking.image import preprocess_image, show_image
from tracking.motion import move_particles
from tracking.reader import read_image
from tracking.state import initial_state, update_state, prune_particles, find_mean_particle, populate_particles


NUM_PARTICLES = 100


def track_cells():
    for i in range(3):
        original = read_image(i)
        cropped, smoothed, threshed, contoured, contours = preprocess_image(original)

        cells, _ = register_cells(contours)


def main():
    # sample image path
    original = read_image(1)

    # process image
    cropped, smoothed, threshed, contoured, contours = preprocess_image(original)

    # detect cells in image
    cells, cell_contours = register_cells(contours)

    # draw cell contours
    cells_image = cv2.cvtColor(contoured, cv2.COLOR_GRAY2BGR)

    # display_cell_contours(cells_image, cell_contours, cell_idx=-1)
    # display_rect_bounds(cells_image, cell_contours)
    # display_circle_bounds(cells_image, cell_contours)

    # show image of drawn contours
    # show_image("contoured_image", cells_image)

    # create sample cell particles
    cell_particles = initial_state(cells_image.shape, cells[0], num_particles=100)

    # create color image for display
    particle_image = cv2.cvtColor(smoothed.copy(), cv2.COLOR_GRAY2BGR)

    # draw cell particles
    display_particles(particle_image, cell_particles, bcell.PARTICLE_COLOR)

    display_cell(particle_image, cells[0], bcell.CELL_COLOR)

    # display cell particles
    # show_image("particle_image", particle_image)

    # Image 2

    original2 = read_image(2)

    cropped2, smoothed2, threshed2, contoured2, contours2 = preprocess_image(original2)

    cells2, cell_contours2 = register_cells(contours2)

    prev_c = cells[0, bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX]
    curr_c = cells2[0, bcell.FIRST_POS_INDEX:bcell.LAST_POS_INDEX]
    mov_vec = curr_c - prev_c

    move_particles(cell_particles, mov_vec)

    update_state(cell_particles, cells2[0])

    pruned_particles = prune_particles(cell_particles)

    mean_particle = find_mean_particle(pruned_particles)

    display_cells(particle_image, pruned_particles, cv_color.LIGHT_ORANGE)

    display_cell(particle_image, cells2[0], bcell.CELL_COLOR)

    display_cell(particle_image, mean_particle, cv_color.DEEP_SKY_BLUE)

    # show_image("particle_image", particle_image)

    cell_particles2 = populate_particles(pruned_particles, num_particles=100)

    particle_image2 = cv2.cvtColor(smoothed2.copy(), cv2.COLOR_GRAY2BGR)

    display_particles(particle_image2, cell_particles2, bcell.PARTICLE_COLOR)
    display_cell(particle_image2, cells2[0], bcell.CELL_COLOR)

    show_image("particle_image_2", particle_image2)


if __name__ == "__main__":
    main()
