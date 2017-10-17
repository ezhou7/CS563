import cv2
from tracking.cell import AbstractCell


def draw_cell(image, cell: AbstractCell, color):
    center = cell.get_center()
    radius = cell.get_radius()
    velocity = cell.get_velocity()

    edge_point = (int(center[0] + velocity[0]), int(center[1] + velocity[1]))

    cv2.circle(image, center, radius, color, thickness=2)
    cv2.arrowedLine(image, center, edge_point, color, thickness=2)


