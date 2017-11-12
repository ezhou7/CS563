from tracking import cv_color


# ---- Cell information indices ---- #

# Total size
MAX_INDICES = 4

# Boundary indices
BEG_POS_INDEX = 0
END_POS_INDEX = 2

# Individual indices
X_POS_INDEX = 0
Y_POS_INDEX = 1

RADIUS_INDEX = 2
WEIGHT_INDEX = 3


# ---- Cell graphics information ---- #

# Color
CELL_COLOR = cv_color.RED
PARTICLE_COLOR = cv_color.LIGHT_ORANGE
PREDICTION_COLOR = cv_color.SKY_BLUE

CONTOUR_COLOR = cv_color.RED
BOUND_COLOR = cv_color.GREEN

# THICKNESS
DEFAULT_THICKNESS = 2
CONTOUR_THICKNESS = 3

# SIZE
PARTICLE_RADIUS = 20
PREDICTION_RADIUS = 40
