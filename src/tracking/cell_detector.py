import cv2
from tracking import color
from tracking.cell import Cell


def find_cells(src, contours):
    color_image = cv2.cvtColor(src.copy(), cv2.COLOR_GRAY2BGR)

    contour_areas = [cv2.contourArea(contour) for contour in contours]

    print(contour_areas)

    cells = []
    cell_contours = []
    for contour, area in zip(contours, contour_areas):
        if area < 100:
            continue

        cell_contours.append(contour)

        # show rectangle bound on cell
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(color_image, (x, y), (x + width, y + height), color.GREEN, thickness=2)

        center, radius = cv2.minEnclosingCircle(contour)

        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        # show circle bound on cell
        cv2.circle(color_image, center, radius, color.GREEN, thickness=2)

        cells.append(Cell(center, radius, area))

    return color_image, cells, cell_contours
