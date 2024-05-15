"""
This module provides functions for computing basis functions using barycentric coordinates.

Module Functions:
    compute_basis_function(elmnt_nodes, point_x, point_y): Calculate barycentric coordinates for a point within a triangle and use them as basis function values.
"""

import numpy as np

def compute_basis_function(el_nodes, point_x, point_y):
    """
    Calculate barycentric coordinates for a point within a triangle and use them as basis function values.

    Parameters:
        el_nodes (numpy.ndarray): Array of shape (3, 2) containing the coordinates of the triangle nodes.
        point_x (float): x-coordinate of the point.
        point_y (float): y-coordinate of the point.

    Returns:
        lambdas (numpy.ndarray): Array of shape (3,) containing the barycentric coordinates (basis function values).

    Notes:
        Any point (x, y) inside the triangle can be expressed using barycentric coordinates as:
        (x, y) = λ1(x1, y1) + λ2(x2, y2) + λ3(x3, y3),
        where λ1, λ2, and λ3 are the barycentric coordinates.

        The barycentric coordinates are the basis function values.

    """
    # Unpack the coordinates of the nodes within a triangular element nodes
    x1, y1, x2, y2, x3, y3 = el_nodes.flatten()

    # Construct matrix to solve for barycentric coordinates
    A = np.array([
        [x1, x2, x3],
        [y1, y2, y3],
        [1, 1, 1]
    ])

    # Construct vector b from the point to close the system of equations
    b = np.array([point_x, point_y, 1])

    # Solve for barycentric coordinates (which are the basis function values)
    lambdas = np.linalg.solve(A, b)
    return lambdas

# Example usage
# Define the vertices of the triangle
# vertices = np.array([[0, 0], [1, 0], [0, 1]])

# Compute and print basis function values at the point
# lambdas = compute_basis_function(vertices,0.33, 0.33)
# print("Basis Function Values at point (0.33, 0.33):", lambdas)


