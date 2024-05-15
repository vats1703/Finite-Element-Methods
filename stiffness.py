"""
This module provides functions for computing  grad_basis, local and global stiffness matrices using barycentric coordinates.

Module Functions:
    triangle_area(elmnt_nodes): Compute the area of a triangle given its vertices.
    compute_grad(elmnt_nodes): Compute the gradients of the barycentric basis functions.
    local_stiffness(elmnt_nodes): Assemble the local stiffness matrix for a triangular element.
    calculate_global_stiffness(nodes, elements, num_nodes, num_elements): Calculate the global stiffness matrix for a finite element model.
"""

import numpy as np
from scipy.sparse import lil_matrix
import basis as cart_basis

def triangle_area(elmnt_nodes):
    """
    Compute the area of a triangle given its vertices.

    Parameters:
    elmnt_nodes (ndarray): An array containing the coordinates of the triangle vertices in the form [x1, y1, x2, y2, x3, y3].

    Returns:
    float: The area of the triangle.

    """

    # Unpack the coordinates of the nodes within a triangular element nodes
    x1, y1, x2, y2, x3, y3 = elmnt_nodes.flatten()
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

def compute_grad(elmnt_nodes):
    """
    Compute the gradients of the barycentric basis functions.

    Parameters:
        elmnt_nodes (ndarray): An array containing the coordinates of the triangle vertices in the form [x1, y1, x2, y2, x3, y3].

    Returns:
        ndarray: Array of shape (3, 2) containing the gradients of the barycentric basis functions.

    Notes:
        We use the grad of a reference triangle to compute the grad of the actual triangle.
        Such reference is a triangle with vertices at (0, 0), (1, 0), and (0, 1).

    """
    x1, y1, x2, y2, x3, y3 = elmnt_nodes.flatten()
    
    # Calculate the Jacobian matrix and its inverse transpose
    J = np.array([ [x2 - x1, x3 - x1],
                [y2 - y1, y3 - y1]])
    J_inv_T = np.linalg.inv(J).T
    
    # Gradients of barycentric coordinates in reference triangle
    grad_lambda_ref = np.array([
        [-1, -1],
        [ 1,  0],
        [ 0,  1]
    ])
    
    # Gradients in the actual triangle
    grad_lambda = grad_lambda_ref @ J_inv_T
    
    return grad_lambda

def local_stiffness(elmnt_nodes):
    """Assemble the local stiffness matrix for a triangular element.

    Parameters:
        elmnt_nodes (array-like): The coordinates of the element nodes.

    Returns:
        array-like: The local stiffness matrix.
    """
    area = triangle_area(elmnt_nodes)
    grad_lambda = compute_grad(elmnt_nodes)
    
    # Initialize local stiffness matrix. We use zeros as we know that the global matrix is sparse one.
    A_local = np.zeros((3, 3))
    
    # Compute the local stiffness matrix
    for i in range(3):
        for j in range(3):
            A_local[i, j] = area * (grad_lambda[i] @ grad_lambda[j])
    return A_local

# Initialize the global stiffness matrix


def calculate_global_stiffness(nodes, elements, num_nodes, num_elements):
    """
    Calculate the global stiffness matrix for a finite element model.

    Parameters:
        nodes (ndarray): The coordinates of the nodes.
        elements (ndarray): The indices of the vertices that form every triangular element.
        num_nodes (int): The number of nodes.
        num_elements (int): The number of elements.

    Returns:
        A_global (lil_matrix): The global stiffness matrix.

    Notes:
        The function assumes that the local stiffness matrix is computed by the `local_stiffness` function.
        We use lil matrix format for the global stiffness matrix to allow for efficient sparse matrix assembly.
    """
    
    # Initialize the global stiffness matrix
    A_global = lil_matrix((num_nodes, num_nodes))

    # Iterate over all triangular elements
    for element in elements:
        # Get the indices of the nodes that form the element
        nodes_indices = element
        # Get the coordinates of the nodes that form the element
        element_vertices = nodes[nodes_indices]
        # Compute the local stiffness matrix
        A_local = local_stiffness(element_vertices)
        for i in range(3):
            for j in range(3):
                # Assemble the global stiffness matrix
                A_global[nodes_indices[i], nodes_indices[j]] += A_local[i, j]
    return A_global














## CALCULATE STIFFNESS MATRIX IN CARTESIAN COORDINATES (NEEDS TO BE IMPROVEDS)

def compute_gradients(vertices):
    coeffs1, coeffs2, coeffs3 = cart_basis.compute_basis_functions_cart(vertices)
    
    # The gradients are the coefficients of x and y in the basis functions
    grad_N1 = coeffs1[1:],  # b1, c1
    grad_N2 = coeffs2[1:],  # b2, c2
    grad_N3 = coeffs3[1:]   # b3, c3
    
    return np.array([grad_N1, grad_N2, grad_N3])

def local_stiffness_matrix_cartesian(vertices):
    area = triangle_area(vertices)
    gradients = compute_gradients(vertices)
    
    # Initialize local stiffness matrix
    A_local = np.zeros((3, 3))
    
    # Compute the local stiffness matrix
    for i in range(3):
        for j in range(3):
            A_local[i, j] = area * np.dot(gradients[i], gradients[j])
    
    return A_local