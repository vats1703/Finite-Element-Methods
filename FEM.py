## Raw Finite Element implementation  to solve Poisson equation in a 2D rectangular domain.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
#sys.path.append('/Users/alex/Desktop/Finite Element Methods/')
import mesh as mesh
import basis as c_basis
import basis_baryo as b_basis
import stiffness as stiffness
import force as force
import c_boundaries as boundaries
from scipy.sparse.linalg import spsolve
import final_eval as final_eval



def plot_solution(X,Y,u_h):
    """
    Plot the finite element solution.
    
    Parameters:
        X (ndarray): The X coordinates of the nodes in the mesh.
        Y (ndarray): The Y coordinates of the nodes in the mesh.
        u_h (ndarray): The finite element solution.
    """
    plt.figure(figsize=(8, 8))
    plt.tricontourf(X, Y, u_h.reshape(X.shape()), levels=20, cmap='viridis')
    plt.colorbar()
    plt.title('Finite Element Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return None 

def FEM_solver(w, h, nx, ny, source_function):
    """
    Solves a finite element problem using the FEM method.

    Parameters:
    w (float): Width of the domain.
    h (float): Height of the domain.
    nx (int): Number of elements in the x-direction.
    ny (int): Number of elements in the y-direction.
    source_function (function): Function that defines the source term.

    Returns:
    xi (nparray): Solution vector.

    """
    # Generate the mesh
    nodes, triang_elements, num_nodes , num_elements, X, Y = mesh.generate_mesh(w, h, nx, ny)

    # Plot the mesh and its triangles
    mesh.plot_mesh(nodes, triang_elements)

    # Calculate the global stiffness matrix
    A_global = stiffness.calculate_global_stiffness(nodes, triang_elements, num_nodes, num_elements)

    # Assemble the global load vector
    F = force.assemble_load_vector(nodes, triang_elements, source_function)

    # Apply Dirichlet boundary conditions
    
    # Define the boundary of the mesh
    mesh_bounds = [0, w, 0, h]   # [x_min, x_max, y_min, y_max]

    # Identify the boundary nodes
    boundary_nodes = boundaries.identify_b_nodes_by_coord(nodes, mesh_bounds)

    # Apply Dirichlet boundary conditions
    A_final, F_final = boundaries.apply_dirichlet(A_global, F, boundary_nodes)

    # Solve the linear system
    xi = spsolve(A_final, F_final)
    
    # Plot the solution
    plot_solution(X,Y,xi)
    return  A_final, F_final
