import numpy as np

def eval_basis_function(elmnt_nodes, point_x, point_y):
    """ Calculate barycentric coordinates for a point within a triangle and use them as basis function values. """
    x1, y1 ,x2, y2, x3,y3 = elmnt_nodes.flatten()
    # Any point (x,y) inside the triangle, can be expressed using barycentric coordinates as: 
    # (x , y ) = λ1(x1​,y1​) + λ2(x2​,y2​) + λ3(x3​,y3​) and we can use a point to obtain the barycentric coordinates. 
    # The barycentric coordinates are the basis function values.
   
    # Construct matrix to solve for barycentric coordinates
    A = np.array([
        [x1, x2, x3],
        [y1, y2, y3],
        [1, 1, 1]
    ])
    
    # Construct vector b from the point to close the system of equation
    b = np.array([point_x, point_y, 1])
    
    # Solve for barycentric coordinates (which are the basis function values)
    lambdas = np.linalg.solve(A, b)
    return lambdas

# Example usage
vertices = np.array([[0, 0], [1, 0], [0, 1]])

# # Compute and print basis function values at the point
lambdas = eval_basis_function(vertices,0.33, 0.33)
print("Basis Function Values at point (0.33, 0.33):", lambdas)


