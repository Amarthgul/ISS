
from enum import Enum
from Backend import backend as bd 


"""
Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, 
Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω
"""
# ==================================================================
""" ============= Consts, flags, and definitions =============== """
# ==================================================================


# Global flag for developing and debugging features 
DEVELOPER_MODE = True 


# Gloabl flag if the ray path feature is enabled
ENABLE_RAYPATH = True
# Recommend to turn this off for image propagation to save space. 


# Placeholder focus/object distance 
KNOB_DISTANCE = 1500 
# Name and value from leica lenses which has a focus knob,
# the lens will focus at 1.2 to 1.5 meter when the knob point down 


# Creates a RNG for the entire program to use 
RANDOM_SEED = 42 
rng = bd.random.default_rng(seed=RANDOM_SEED)

def RefreshRNG():
    """Refresh the RNG with a new seed generated using itself"""
    newSeed = bd.random.random_integers(1)
    rng = bd.random.default_rng(seed=newSeed)


# The threashold by which a raybatch will no longer propagate 
RADIANT_KILL = 0.001
# Changing this could increase accuracy, at the cost of increase time 


# Global variable for per spot sampling in image formation. 
PER_POINT_MAX_SAMPLE = 100
# This can be used to estimate and normalize the colors. 


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


# Ways to fit a gate when input and gate has different aspect ratio 
class Fit(Enum):
    FILL = 1  # Stretch both horizontal and vertical direction to fill the entire gate 
    FIT = 2   # Proportionally scale the image to fit one axis and ensure the image is not cropped 


# Placeholder varible for arguments 
SOME_BIG_CONST = 1024
SOME_SML_CONST = 1e-10
