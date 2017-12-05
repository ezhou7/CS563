from tracking.threedim import stain


# ---- Cell information indices ---- #

# Spatial Dimension
DIMS = 3

# Total size
MAX_INDICES = 8

# Boundary indices
BEG_POS_INDEX = 0
END_POS_INDEX = 3

BEG_VEL_INDEX = 3
END_VEL_INDEX = 6

# Individual indices
X_POS_INDEX = 0
Y_POS_INDEX = 1
Z_POS_INDEX = 2

X_VEL_INDEX = 3
Y_VEL_INDEX = 4
Z_VEL_INDEX = 5

RADIUS_INDEX = 6
WEIGHT_INDEX = 7


# ---- Cell graphics information ---- #

# Color
CELL_COLOR = stain.RED
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
