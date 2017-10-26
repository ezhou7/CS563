from tracking import cv_color


# ---- Cell information indices ---- #

# Boundary indices
FIRST_POS_INDEX = 0
LAST_POS_INDEX = 2

# FIRST_VEL_INDEX = 2
# LAST_VEL_INDEX = 3
#
# FIRST_ACCEL_INDEX = 4
# LAST_ACCEL_INDEX = 5

# Individual indices
X_POS_INDEX = 0
Y_POS_INDEX = 1

# X_VEL_INDEX = 2
# Y_VEL_INDEX = 3
#
# X_ACCEL_INDEX = 4
# Y_ACCEL_INDEX = 5

RADIUS_INDEX = 2
WEIGHT_INDEX = 3


# ---- Cell graphics information ---- #

# Color
CONTOUR_COLOR = cv_color.RED
CELL_COLOR = cv_color.RED
PARTICLE_COLOR = cv_color.GREEN
BOUND_COLOR = cv_color.GREEN

# THICKNESS
DEFAULT_THICKNESS = 2
CONTOUR_THICKNESS = 3
