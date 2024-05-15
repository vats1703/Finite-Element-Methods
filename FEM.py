## Raw Finite Element implementation  to solve Poisson equation in a 2D rectangular domain.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
#sys.path.append('/Users/alex/Desktop/Finite Element Methods/')
import mesh as m
import basis as c_basis
import basis_baryo as b_basis
import stiffness as stiffness
import force as force
import c_boundaries as boundaries
from scipy.sparse.linalg import spsolve
import final_eval as final_eval

# Parameters
a = 1  # width of the rectangle
b = 1  # height of the rectangle
nx = 20  # divisions along x
ny = 20  # divisions along y

# # # Generate mesh
nodes, triang_elements, num_nodes , num_elements  = m.generate_mesh(a, b, nx, ny)

#Plotting the mesh
plt.figure(figsize=(8, 4))
plt.triplot(nodes[:, 0], nodes[:, 1], triang_elements, 'k.-')
plt.gca().set_aspect('equal')
plt.title('Finite Element Mesh')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# print("Number of nodes:", num_nodes)
# print("Number of elements:", num_elements)
# print("Nodes positions:\n", nodes)
# print("Elements (triangles):\n", triang_elements)

# print("Nodes index of first element:\n",nodes[triang_elements[0]])
# Compute basis functions for the first triangular element.

#coeffs = c_basis.compute_basis_functions(points, triangles[0])

#basis_function_values = b_basis.eval_basis_function(nodes[triang_elements[0]], 0.33, 0.33)

# # Evaluate basis function N1 at the center of the triangle
# N1 = c_basis.evaluate_basis_function(coeffs, 0.33, 0.33)
# # print("")
# print(basis_function_values,basis_function_values.sum())
# print(N1,N1.sum())

# A_local = stiffness.local_stiffness(points[triangles[0]])
# print("Local Stiffness Matrix (Barycentric):\n", A_local)

#num_nodes, num_elements = len(points), len(triangles)
A_global = stiffness.calculate_global_stiffness(nodes, triang_elements, num_nodes, num_elements)
# print("Assembled Global Stiffness Matrix:\n", A_global)

#print("Global Stiffness Matrix:\n", A_global)

F, lambdas = force.assemble_load_vector(nodes, triang_elements)
# print("Assembled Force Vector:", F)

#local_f = force.integrate_over_triangle(nodes[triang_elements[0]])
mesh_bounds = [0, a, 0, b]
boundary_nodes = boundaries.identify_b_nodes_by_coord(nodes, mesh_bounds)

# Apply Dirichlet boundary conditions
final_A, final_F = boundaries.apply_dirichlet(A_global, F, boundary_nodes)  
# print(type(nodes[triang_elements[0]]))
# print(nodes[triang_elements[0]])

# print("Final Stiffness Matrix:\n", final_A)
# print("Final Load Vector:\n", final_F)
# print("Final lambda:\n", lambdas)

xi = spsolve(final_A, final_F)

# print("Solution (xi):\n", xi)

x = np.linspace(0, a, 2*nx)
y = np.linspace(0, b, 2*ny)
X, Y = np.meshgrid(x, y)

# Flatten the meshgrid for use in the interpolation function
points = np.vstack([X.ravel(), Y.ravel()]).T

u_h = final_eval.interpolate_solution(nodes, triang_elements, xi, points)

# Plot the solution
plt.figure(figsize=(8, 8))
plt.contourf(points, u_h, levels = 20, cmap = 'viridis')
plt.colorbar()
plt.title('Finite Element Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


