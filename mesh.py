"""
Raw Finite Element implementation to solve Poisson equation in a 2D rectangular domain.

This module provides functions to generate a triangular mesh for a rectangular domain and plot the mesh.

Functions:
- generate_mesh(a, b, nx, ny): Generate a triangular mesh for a rectangular domain.
- plot_mesh(nodes, triang_elements): Plot the generated mesh.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def generate_mesh(a, b, nx, ny):
    """
    Generate a triangular mesh for a rectangular domain.

    Parameters:
    a (float): The width of the rectangle.
    b (float): The height of the rectangle.
    nx (int): The number of divisions along the x-axis.
    ny (int): The number of divisions along the y-axis.

    Returns:
    nodes_pos (ndarray): N x 2 matrix of nodes positions.
    triangles (ndarray): N x 3 matrix of elements. Each row contains the indices of the nodes of a triangle.
    num_nodes (int): The number of nodes in the mesh.
    num_elements (int): The number of elements in the mesh.
    X (ndarray): The X coordinates of the nodes in the mesh.
    Y (ndarray): The Y coordinates of the nodes in the mesh.
    """

    # Generate a grid of points in x and y direction
    x = np.linspace(0, a, nx + 1)
    y = np.linspace(0, b, ny + 1)

    # Generate the grid using meshgrid. 
    X , Y = np.meshgrid(x, y)

    # Stack the x and y coordinates of the points to form a matrix. Each row is a (x,y) point on the mesh
    # We use ravel() in this case also flatten() works. But ravel() is faster since it creates a view of the original array.
    nodes_pos = np.vstack([X.ravel(), Y.ravel()]).T  # N x 2 matrix of nodes positions

    triangles = []  # N x 3 matrix of elements
    for j in range(ny):
        for i in range(nx):
            # Calculate the indices of the nodes for each triangle
            node1 = j * (nx + 1) + i
            node2 = node1 + nx + 1

            # First triangle
            triangles.append([node1, node2, node1 + 1])

            # Second triangle
            triangles.append([node1 + 1, node2, node2 + 1])

    num_nodes = len(nodes_pos)
    num_elements = len(triangles)

    return nodes_pos, np.array(triangles), num_nodes, num_elements, X , Y

def plot_mesh(nodes, triang_elements):
    """    Plot the generated mesh.

    Parameters:
    nodes (ndarray): N x 2 matrix of nodes positions.
    triang_elements (ndarray): N x 3 matrix of elements.

    Returns:
    None
    """
    plt.figure(figsize=(8, 4))
    plt.triplot(nodes[:, 0], nodes[:, 1], triang_elements, 'k.-')
    plt.gca().set_aspect('equal')
    plt.title('Finite Element Mesh for our rectangle')
    plt.xlabel('x')
    plt.ylabel('y[x]')
    plt.show()
    return None
