# Finite-Element-Methods

 Here I present a very basic implementation of Finite Element methods to solve Poisson's equation in 2D. 

 I have separated in every module each part of the implementation that consists on

 1) Defining the mesh (mesh.py)
 2) Construct basis functions in either cartesian (base.py) or barycentric coordinates (baryo.py)
 3) Construct the local stiffness matrix and assembly the global one (stiffness.py)
 4) Assembly the loading vector (force.py)
 5) Apply boundary conditions in the stiffness matrix and loading vector (c_boundaries.py)
 6) Solve the linear system and construct a general FEMSolver(FEM.py)
 7) Implement convergence theory for mesh refinement(to be added)

Additionally, you can find in the jupyter notebook a short summary of the implementation with some examples and the mathematical description of the theory

 
