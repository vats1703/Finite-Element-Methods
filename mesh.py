## Raw Finite Element implementation  to solve Poisson equation in a 2D rectangular domain.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def generate_mesh(a, b, nx, ny):
    # a, b: dimensions of the rectangle
    # nx, ny: number of divisions on each axis. Please recall that this is not the number of nodes but the number of divisions.

    # Generate a grid of points
    x = np.linspace(0, a, nx + 1 )
    y = np.linspace(0, b, ny + 1)

    # Instead of using inneficient loops to generate the points connecting x and y, we  take advantage of vectorization implemented
    # within Numpy  to generate the grid. The meshgrid function generates a grid of points in the form of a matrix. The first matrix
    # contains the x-coordinates of the points and the second matrix contains the y-coordinates of the points. 

    X, Y = np.meshgrid(x, y)
    # X contains the x-coordinates of the points
    # Y containts the y-coordinates of the points

    # The ith component of nodes_pos mathrix contains the x and y coordinates of the ith node.
    # We then used vertical stack function to stack the x and y coordinates of the points to form a matrix.
    # We use ravel() in this case also flatten() works. But ravel() is faster
    nodes_pos = np.vstack([X.ravel(), Y.ravel()]).T  # N x 2 matrix of nodes positions

    #elements = []
    triangles = [] # Return the triang_element formed  by referencing three nodes from the nodes_pos matrix. # N x 3 matrix of elements
    for j in range(ny):
        for i in range(nx):
            #Each row (say, i-th row) in the elements matrix defines the i-th element by referencing three nodes from the nodes matrix.
            node1 = j * (nx + 1) + i
            node2 = node1 + nx + 1 # "height" of the triangle
            #node3 = node1 + 1
            #node4 = node2 + 1
           # print(node1,node2,node3,node4)
            
            # First triangle
            triangles.append([node1, node2, node1 + 1])
            #elements.append([nodes_pos[node1],nodes_pos[node2], nodes_pos[node3]])
            #print(triangles)
            # Second triangle
            triangles.append([node1 + 1, node2, node2 + 1])
            #elements.append([nodes_pos[node3],nodes_pos[node2], nodes_pos[node4]])

    return nodes_pos, np.array(triangles), len(nodes_pos), len(triangles)

# # Parameters
# a = 2  # width of the rectangle
# b = 2  # height of the rectangle
# nx = 2  # divisions along x
# ny = 2 # divisions along y

# # # # Generate mesh
# points, triangles, n_nodes, n_elements = generate_mesh(a, b, nx, ny)
# print("Number of nodes:", n_nodes)
# print("Number of elements:", n_elements)
# print("Nodes positions:\n", points)
# print("Elements (triangles):\n", triangles)

#Plotting the mesh
# plt.figure(figsize=(8, 4))
# plt.triplot(points[:, 0], points[:, 1], triangles, 'k.-')
# plt.gca().set_aspect('equal')
# plt.title('Finite Element Mesh')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
