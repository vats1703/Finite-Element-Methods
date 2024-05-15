import numpy as np
import basis_baryo as b_basis

def interpolate_solution(nodes, elements, xi, points):
    """Interpolate the finite element solution at a set of points."""


    u_h = np.zeros(len(points))

    for i, point in enumerate(points):
        for element in elements:
            vertex_indices = element
            # Check if point is inside the current triangle using barycentric coordinates
            # print("vertex_indices",vertex_indices)
            # print("nodes[vertex_indices]",nodes[vertex_indices])
            # print("point[1][0]",point[0])
            # print("point[1][1]",point[1])
            lambdas = b_basis.eval_basis_function(nodes[vertex_indices], point[0], point[1])
            if np.all(lambdas >= 0) and np.all(lambdas <= 1):  # The point lies within the triangle
                # Interpolate solution at this point
                u_h[i] = np.dot(lambdas, xi[vertex_indices])
                break
    return u_h