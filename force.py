import numpy as np
import basis_baryo as b_basis
def test_function(x,y):
    """Example test function f(x) = x + y."""
    return - (x**2 + y**2)


def triangle_area(vertices):
    x1, y1, x2, y2, x3, y3 = vertices.flatten()
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))


def integrate_over_triangle(nodes):
    # Calculate the centroid of the triangle
    x1, y1, x2, y2, x3, y3 = nodes.flatten()
    centroid_x = (x1 + x2 + x3) / 3
    centroid_y = (y1 + y2 + y3) / 3

    # Calculate the area of the triangle
    area = triangle_area(nodes)

    # Evaluate the force function at the centroid
    force_centroid = test_function(centroid_x, centroid_y)

    # Get barycentric coordinates at the centroid
    lambdas = b_basis.eval_basis_function(nodes, centroid_x, centroid_y)

    # Load vector contributions for each vertex
    load_vector = [area * force_centroid * lam for lam in lambdas]
    
    return load_vector, lambdas

def assemble_load_vector(nodes, elements):
    num_nodes = len(nodes)
    F = np.zeros(num_nodes)
    lambdas = np.zeros(num_nodes)

    for element in elements:
        node_indices = element
        #vertices = [nodes[idx] for idx in node_indices]
        #print
        #print(type(vertices),type(node_indices))
        # Integrate over the current triangle))
        local_loads, lambdas_loc = integrate_over_triangle(nodes[node_indices])
        
        # Add contributions to the global load vector
        for i, node_idx in enumerate(node_indices):
            F[node_idx] += local_loads[i]
            lambdas[node_idx] += lambdas_loc[i]
    return F, lambdas
