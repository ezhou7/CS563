from tracking.twodim import stain


# ---- Cell information indices ---- #

# Total size
MAX_INDICES = 6

# Boundary indices
BEG_POS_INDEX = 0
END_POS_INDEX = 2

BEG_VEL_INDEX = 2
END_VEL_INDEX = 4

# Individual indices
X_POS_INDEX = 0
Y_POS_INDEX = 1

X_VEL_INDEX = 2
Y_VEL_INDEX = 3

RADIUS_INDEX = 4
WEIGHT_INDEX = 5


# ---- Cell graphics information ---- #

# Color
CELL_COLOR = stain.LIGHT_ORANGE
PARTICLE_COLOR = stain.LIGHT_ORANGE
PREDICTION_COLOR = stain.SKY_BLUE

CONTOUR_COLOR = stain.RED
BOUND_COLOR = stain.GREEN

# THICKNESS
DEFAULT_THICKNESS = 2
CONTOUR_THICKNESS = 3

# SIZE
PARTICLE_RADIUS = 20
PREDICTION_RADIUS = 40
