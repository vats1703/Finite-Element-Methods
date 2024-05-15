import numpy as np
# Compute basis functions


def compute_basis_functions(points, k_triangle):
    # k_triangle contains the index of the nodes for each triangle
    # Extract vertices of each triangle 
    x1, y1 ,x2, y2, x3,y3 = points[k_triangle].flatten()
    # x2, y2 = points[triangle[1]]
    # x3, y3 = points[triangle[2]]
    
    # Set up the coeficient matrix for the basis functions N_i(x,y) = a_i + b_i*x + c_i*y
    A = np.array([[1, x1, y1],
                  [1, x2, y2],
                  [1, x3, y3]])
    
    # Inverse of matrix A to calculate the coefficients of the basis functions 
    A_inv = np.linalg.inv(A)
    
    # Coefficients for the basis functions
    # Each column of A_inv corresponds to the coefficients of one basis function
    return A_inv

def evaluate_basis_function(coeffs, x, y):
    # coeffs is one column of A_inv
    return coeffs[0] + coeffs[1] * x + coeffs[2] * y

# # Example usage for a single triangle
# points = np.array([[0, 0], [1, 0], [0, 1]])
# coeffs = compute_basis_functions(points)

# # Evaluate basis function N1 at the center of the triangle
# N1_at_center = evaluate_basis_function(coeffs[0], 0.33, 0.33)
# print("N1 at (0.33, 0.33):", N1_at_center)