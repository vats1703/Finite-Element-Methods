
import numpy as np
from scipy.sparse import csr_matrix



def identify_b_nodes_by_coord(nodes, bounds):
    """
    Identifies boundary nodes based on their coordinates.

    Parameters:
        nodes (ndarray)(N x 2): An array containing the coordinates of the triangle vertices in the form [[x1, y1], [x2, y2],...[ xn, yn]].
        bounds (list): List containing the minimum and maximum values for x and y coordinates in the form (xmin, xmax, ymin, ymax).

    Returns:
        list: List of indices of the boundary nodes.

    """
    xmin, xmax, ymin, ymax = bounds
    boundary_nodes = []
    for idx, (x, y) in enumerate(nodes):
        if x == xmin or x == xmax or y == ymin or y == ymax:
            boundary_nodes.append(idx)
    return boundary_nodes



def apply_dirichlet(A, F, boundary_conditions):
    """
    Apply Dirichlet boundary conditions to the stiffness matrix and load vector.
    
    Parameters:
        A (ndarray): The global stiffness matrix.
        F (ndarray): The global load vector.
        boundary_conditions (list): A dictionary where keys are the node indices
                                    subject to Dirichlet conditions and values
                                    are the prescribed values at these nodes.
    
    Returns:
        A_modified (csr_matrix): Modified global stiffness matrix in csr format. Useful for sparse matrices.
        F_modified (ndarray): Modified global load vector.
    """
    A_modified = A.copy()
    F_modified = F.copy()
    
    for node_index in boundary_conditions:
        # Set all entries in the row and column to zero
        A_modified[node_index, :] = 0
        A_modified[:, node_index] = 0
        
        # Set the diagonal entry for this node to 1
        A_modified[node_index, node_index] = 1
        
        # Set the corresponding entry in the load vector to the prescribed value
        F_modified[node_index] = 0
    
    return csr_matrix(A_modified), F_modified


