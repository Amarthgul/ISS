
import numpy

"""
Modified from: 

diffractsim - https://github.com/rafael-fuente/diffractsim 

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


global JAX_AVAILABLE
global CUPY_CUDA_AVAILABLE

try:
    import cupy
    CUPY_CUDA_AVAILABLE = True
except ImportError:
    CUPY_CUDA_AVAILABLE = False


global backend
backend = numpy
global backend_name
backend_name = 'numpy'

def set_backend(name: str):
    """ Set the backend for the simulations
    This way, all methods of the backend object will be replaced.
    Args:
        name: name of the backend. Allowed backend names:
            - ``CPU``
            - ``CUDA``
    """
    # perform checks
    if name == "CUDA" and not CUPY_CUDA_AVAILABLE:
        raise RuntimeError(
            "Cupy CUDA backend is not available.\n"
            "Do you have a GPU on your computer?\n"
            "Is Cupy with CUDA support installed?"
        )
    global backend
    global backend_name

    # change backend
    if name == "CPU":
        backend = numpy
        backend_name = 'numpy'

    elif name == "CUDA":
        backend = cupy
        backend_name = 'cupy'
    else:
        raise RuntimeError(f'unknown backend "{name}"')


def GetBackend():
    global backend    
    print(backend)

    