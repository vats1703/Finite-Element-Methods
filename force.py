import numpy as np
import basis_baryo as b_basis
from stiffness import triangle_area as triag_area


def integrate_over_triangle(nodes, source_function):
    """
    Integrate the source function over a triangle using a simple method.

    Parameters:
        nodes (ndarray): The coordinates of the triangle vertices.
        source_function (function): The source function to be integrated.

    Returns:
        float: The approximate integral value over the triangle. Will be the same for all three vertices.
                While we increase the number of elements, the integral value will be more accurate.
    """
    # Calculate the area of the triangle using determinant formula
    area = triag_area(nodes)
    
    # Use the centroid (average of vertices) for a simple midpoint rule approximation
    x1, y1, x2, y2, x3, y3 = nodes.flatten()
    centroid_x = (x1 + x2 + x3) / 3
    centroid_y = (y1 + y2 + y3) / 3
    # centroid = np.mean(nodes, axis=0)
    value_at_centroid = source_function(centroid_x , centroid_y)
    
    # Approximate integral over the triangle
    integral_value = value_at_centroid * area
    return integral_value


def assemble_load_vector(nodes, elements, source_function):
    """
    Assemble the global load vector for a mesh of triangles.

    Parameters:
        nodes (ndarray): The coordinates of all nodes in the mesh.
        elements (ndarray): The list of element indices.
        source_function (function): The function to be integrated.

    Returns:
        ndarray: The global load vector.
    """
    num_nodes = len(nodes)
    f_global = np.zeros(num_nodes)

    # Loop over all triangular elements
    for element in elements:
        vertices = nodes[element]
        
        # Integrate over the current triangle
        # We will get the same value for the three nodes of the triangle
        local_integrals = [integrate_over_triangle(vertices, source_function) for _ in element]
        
        # Add contributions to the global load vector
        for i, node_index in enumerate(element):
            f_global[node_index] += local_integrals[i]

    return f_global