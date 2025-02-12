


from Util.Backend import backend as bd
from Util.Globals import ONE




def ModifyEllipse(A, v, add=True):
    """
    Modify an ellipse to expand or contract in the direction of vector `v`.
    
    :param A: 2x2 positive definite matrix defining the original ellipse (ndarray).
    :param v: 2D vector specifying direction and magnitude (ndarray). 
    :param add: when set to True, ellipse will be expanded, otherwise contracted. 

    :return: bd.ndarray: New 2x2 matrix defining the modified ellipse
    """

    v = bd.array(v, dtype=float)
    # assert A.shape == (2, 2), "A must be 2x2 matrix"
    # assert v.shape == (2,), "v must be 2D vector"
    
    # Compute direction angle
    theta = bd.arctan2(v[1], v[0])
    c, s_rot = bd.cos(theta), bd.sin(theta)
    R = bd.array([[c, -s_rot], [s_rot, c]])  # Rotation matrix
    
    m = bd.linalg.norm(v)  # Vector magnitude
    if (add):
        s_factor = bd.sqrt(v @ A @ v)
    else:
        # Calculate original length in direction v
        L = m / bd.sqrt(v @ A @ v)
        # Calculate new length after subtraction
        L_new = L - m
        # if L_new <= 0:
        #     raise ValueError(f"Cannot subtract - vector magnitude {m:.2f} exceeds ellipse extent {L:.2f}")
        s_factor = (L_new / L)  # Scaling factor

    # Transformation matrix components
    S = bd.diag([int(s_factor), 1])      # Scaling matrix
    M_inv = R @ bd.linalg.inv(S) @ R.T  # Combined transformation
    
    # Compute new ellipse matrix
    A_new = M_inv.T @ A @ M_inv
    
    return A_new



def main():
    A = bd.eye(2)

    # Vector specifying direction and magnitude
    v = bd.array([2, 0])

    # Get modified ellipse
    A_new = ModifyEllipse(A, v)

    # Verify new ellipse properties
    print("New ellipse matrix:")
    print(A_new)

if __name__ == "__main__":
    main()