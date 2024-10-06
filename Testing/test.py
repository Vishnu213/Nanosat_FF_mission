# import casadi as ca
# import numpy

# def extract_non_zero_columns(M):
#     # Sum the absolute values of each column in the matrix
#     column_sums = ca.sum1(ca.fabs(M))

#     # Create a mask to identify non-zero columns (this is symbolic)
#     non_zero_columns_mask = column_sums > 0

#     # Create an empty list to store non-zero column indices
#     non_zero_column_indices = []

#     # Loop through the mask and store indices of non-zero columns
#     for i in range(M.shape[1]):
#         # Append the index if the column is non-zero, -1 otherwise
#         non_zero_column_indices.append(ca.if_else(non_zero_columns_mask[i], i, -1))  # -1 as a placeholder for zero columns

#     # Create a CasADi function to evaluate the non-zero column indices
#     non_zero_indices_func = ca.Function('non_zero_indices_func', [M], [ca.vertcat(*non_zero_column_indices)])

#     return non_zero_indices_func

# def filter_symbolic_matrix(M, M_values):
#     # Get the function to compute non-zero column indices
#     non_zero_indices_func = extract_non_zero_columns(M)

#     # Evaluate the non-zero column indices using the numeric matrix
#     non_zero_indices = non_zero_indices_func(M_values)

#     # Convert the result to a list of valid indices (remove -1 which signifies zero columns)
#     non_zero_indices_list = [int(idx) for idx in non_zero_indices.full().flatten() if idx != -1]

#     # Extract the non-zero columns from the symbolic matrix
#     filtered_matrix = ca.horzcat(*[M[:, i] for i in non_zero_indices_list])

#     # Create a CasADi function to evaluate the filtered symbolic matrix
#     filtered_matrix_func = ca.Function('filtered_matrix_func', [M], [filtered_matrix])

#     # Evaluate the filtered matrix with M_values
#     filtered_matrix_evaluated = filtered_matrix_func(M_values)

#     return filtered_matrix_evaluated

# # Example Usage
# if __name__ == "__main__":
#     # Example symbolic matrix M
#     M = ca.MX.sym('M', 4, 6)  # Create a 4x6 symbolic matrix

#     # Example numeric values for M
#     M_values = numpy.array([[1, 0, 0, 2, 0, 0],
#                       [3, 0, 0, 4, 0, 0],
#                       [5, 0, 0, 6, 0, 0],
#                       [7, 0, 0, 8, 0, 0]])

#     # Filter the symbolic matrix based on non-zero columns
#     filtered_matrix_evaluated = filter_symbolic_matrix(M, M_values)

#     # Output the result
#     print("Filtered Matrix:")
#     print(filtered_matrix_evaluated)
import casadi as ca
import numpy as np

def extract_non_zero_columns_symbolic(M_symbolic):
    # Sum the absolute values of each column in the symbolic matrix
    column_sums = ca.sum1(ca.fabs(M_symbolic))

    # Create a mask to identify non-zero columns (this is symbolic)
    non_zero_columns_mask = column_sums > 1e-6  # Use a small threshold to check non-zero values

    # Initialize an empty list for storing valid columns
    valid_columns = []

    # Loop through columns and add non-zero columns to the list
    for i in range(M_symbolic.shape[1]):
        column = M_symbolic[:, i]
        condition = non_zero_columns_mask[i]

        # Only append the column if the condition is true
        valid_columns.append(ca.if_else(condition, column, column))  # Keep original column for true

    # Concatenate all valid columns
    filtered_matrix = ca.horzcat(*valid_columns)

    # Return the filtered symbolic matrix
    return filtered_matrix

# Example Usage
if __name__ == "__main__":
    # Define the symbolic matrix for CasADi operations
    M_symbolic = ca.MX.sym('M', 4, 6)

    # Extract non-zero columns symbolically
    filtered_matrix_symbolic = extract_non_zero_columns_symbolic(M_symbolic)

    # Create a CasADi function for symbolic extraction
    filter_func = ca.Function('filter_func', [M_symbolic], [filtered_matrix_symbolic])

    # Example numeric matrix to test
    M_values = np.array([[1, 0, 0, 2, 0, 0],
                         [3, 0, 0, 4, 0, 0],
                         [5, 0, 0, 6, 0, 0],
                         [7, 0, 0, 8, 0, 0]])

    # Convert M_values to a CasADi DM matrix before calling the function
    M_values_dm = ca.DM(M_values)

    # Evaluate the CasADi function numerically using the test matrix
    filtered_matrix_evaluated = filter_func(M_values_dm)

    # Output the result
    print("Filtered Matrix:")
    print(filtered_matrix_evaluated)
