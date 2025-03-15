
import casadi as ca

# Define extended collocation matrix
C_ext = ca.DM([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Extract the second 2x2 block (C1)
C1 = C_ext[0:2, 2:4]

# Print C1
print("C1:")
print(C1)