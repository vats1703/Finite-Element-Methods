import numpy as np
from scipy.sparse import lil_matrix

def triangle_area(elmnt_nodes):
    x1, y1, x2, y2, x3, y3= elmnt_nodes.flatten()
    # Compute the area of a triangle given its vertices
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

def compute_grad(elmnt_nodes):
    """Compute the gradients of the barycentric basis functions."""
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
    """Assemble the local stiffness matrix for a triangular element (barycentric coordinates)."""
    area = triangle_area(elmnt_nodes)
    grad_lambda = compute_grad(elmnt_nodes)
    
    # Initialize local stiffness matrix. We use zeros as we know that the global matrix is sparse one.
    A_local = np.zeros((3, 3))
    
    # Compute the local stiffness matrix
    for i in range(3):
        for j in range(3):
            A_local[i, j] = area * (grad_lambda[i] @ grad_lambda[j])
    return A_local

# # Example usage
# vertices = np.array([[0, 0], [1, 0], [0, 1]])
# A_local_barycentric = local_stiffness(vertices)
# print("Local Stiffness Matrix (Barycentric):\n", A_local_barycentric)



# Example parameters


# Initialize the global stiffness matrix

def calculate_global_stiffness(nodes, elements, num_nodes, num_elements):
    
    # elements =  indices of the vertices that form every triangular element.
    # nodes =  coordinates of the nodes.
    # num_nodes = len(vertices)
    # num_elements = len(elements)
    A_global = lil_matrix((num_nodes, num_nodes))
    for element in elements:
        # Get the nodes indices for the current element. For example, if the element is [0, 1, 2], then the nodes are 0, 1, and 2.
        nodes_indices = element

        # Get the vertex coordinates for the current element
        element_vertices = nodes[nodes_indices]

        # Compute the local stiffness matrix for this element
        A_local = local_stiffness(element_vertices)

        # Insert the local stiffness matrix into the global matrix
        for i in range(3):
            for j in range(3):
                A_global[nodes_indices[i], nodes_indices[j]] += A_local[i, j]
    return A_global
















def triangle_area(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

def compute_basis_function_coeffs(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    
    # Set up the system of equations
    A = np.array([
        [1, x1, y1],
        [1, x2, y2],
        [1, x3, y3]
    ])
    
    # Right-hand side for each basis function
    b1 = np.array([1, 0, 0])
    b2 = np.array([0, 1, 0])
    b3 = np.array([0, 0, 1])
    
    # Solve for the coefficients
    coeffs1 = np.linalg.solve(A, b1)
    coeffs2 = np.linalg.solve(A, b2)
    coeffs3 = np.linalg.solve(A, b3)
    
    return coeffs1, coeffs2, coeffs3

def compute_gradients(vertices):
    coeffs1, coeffs2, coeffs3 = compute_basis_function_coeffs(vertices)
    
    # The gradients are the coefficients of x and y in the basis functions
    grad_N1 = coeffs1[1:],  # b1, c1
    grad_N2 = coeffs2[1:],  # b2, c2
    grad_N3 = coeffs3[1:]   # b3, c3
    
    return np.array([grad_N1, grad_N2, grad_N3])

def local_stiffness_matrix_cartesian(vertices):
    area = triangle_area(vertices)
    gradients = compute_gradients(vertices)
    
    # Initialize local stiffness matrix
    K_local = np.zeros((3, 3))
    
    # Compute the local stiffness matrix
    for i in range(3):
        for j in range(3):
            K_local[i, j] = area * np.dot(gradients[i], gradients[j])
    
    return K_local