
from enum import Enum


from .Backend import backend as bd 


"""
Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, 
Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω
"""
# ==================================================================
""" ======================= Consts and flags =================== """
# ==================================================================


# Global flag for developing and debugging features 
DEVELOPER_MODE = True 


# Gloabl flag if the ray path feature is enabled
ENABLE_RAYPATH = True
# Recommend to turn this off for image propagation to save space. 


# Data type for the operation 
PRECISION_TYPE = bd.float32


# Placeholder focus/object distance 
KNOB_DISTANCE = 1500 
# Name and value from leica lenses which has a focus knob,
# the lens will focus at 1.2 to 1.5 meter when the knob point down 


# The default radiant value for the raybatch
NORMAL_RADIANT = bd.sqrt(.5)
# This is set so that the 2 directions of the radiant have a normalized value of 1.


# initial phase difference for the two directions of radiant 
INIT_PHASE_DIFF = 0


# The threashold by which a raybatch will no longer propagate 
RADIANT_KILL = bd.array(0.001)
# Changing this could increase accuracy, at the cost of increase time 


INFINITY = bd.array(bd.inf)


# Placeholder varible for arguments 
SOME_BIG_CONST = 1024


# Value that can be viewed as the same zero 
NEAR_ZERO = 1e-10


# Value that can be viewed as on an axis when comparing 
AXIAL_ZERO = 1e-3


# Far distance is treated as 200m away from the lens
FAR_DISTANCE = bd.array(200000.0)
# This is treated as a pesudo infinity for the lens system.


# Scalar contants for Cupy compatibility
ZERO = bd.array(0.0)
ONE = bd.array(1.0)
TWO = bd.array(2.0)


# Global variable for per spot sampling in image formation. 
PER_POINT_MAX_SAMPLE = 100
# This can be used to estimate and normalize the colors. 




# ==================================================================
""" ========================== Vectors ========================= """
# ==================================================================


# Origin point for the lens system 
ORIGIN = bd.array([0, 0, 0])


# Default object facing direction
OBJ_FACING = bd.array([0, 0, -1])


# Up direction 
UP_DIR = bd.array([0, 1, 0])


# ==================================================================
""" ======================== Definitions ======================= """
# ==================================================================


# Fraunhofer symbols used in imager wavelength-RGB conversion 
# and material RI and Abbe calculation 
LambdaLines = {
    "i" : 365.01,  # Default UV cut 
    "h" : 404.66, 
    "g" : 435.84,  # Default B
    "F'": 479.99, 
    "F" : 486.13,  # Default secondary 
    "e" : 546.07,  # Default G
    "d" : 587.56,  
    "D" : 589.3,   # Default secondary
    "C'": 643.85,  # Default R 
    "C" : 656.27, 
    "r" : 706.52, 
    "A'": 768.2,   # Default IR cut 
    "s" : 852.11,
    "t" : 1013.98,
}


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2




# ==================================================================
""" =========================== Enums ========================== """
# ==================================================================


# Ways to fit a gate when input and gate has different aspect ratio 
class Fit(Enum):
    FILL = 1  # Stretch both horizontal and vertical direction to fill the entire gate 
    FIT = 2   # Proportionally scale the image to fit the longer axis while ensuring the image is not cropped 



# ==================================================================
""" ============================ RNG =========================== """
# ==================================================================

# Creates a RNG for the entire program to use 
RANDOM_SEED = 42 
RNG = bd.random.default_rng(seed=RANDOM_SEED)

def RefreshRNG():
    """Refresh the RNG with a new seed generated using itself"""
    newSeed = int(bd.random.random_integers(1))
    RNG = bd.random.default_rng(seed = newSeed)