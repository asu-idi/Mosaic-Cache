import math


def find_grid(space_dim, total_cells=10000):
    """
    Given the dimensions of a space (space_dim) as a list or tuple of three numbers,
    find integers n1, n2, n3 such that:
        n1 * n2 * n3 = total_cells
    and the ratios n1/space_dim[0], n2/space_dim[1], and n3/space_dim[2]
    are as close as possible.

    Parameters:
        space_dim (list or tuple): The dimensions of the space, e.g. [2560, 960, 3456]
        total_cells (int): Total number of cells to split the space into (default 10000)

    Returns:
        tuple: (n1, n2, n3) representing the number of divisions along each axis.
    """
    best_error = None
    best_tuple = None

    # Iterate through all possible divisors for n1
    for n1 in range(1, total_cells + 1):
        if total_cells % n1 != 0:
            continue
        rem = total_cells // n1
        # Iterate through divisors for n2 (n3 is then determined)
        for n2 in range(1, rem + 1):
            if rem % n2 != 0:
                continue
            n3 = rem // n2

            # Compute the normalized ratios with respect to each dimension
            r1 = n1 / space_dim[0]
            r2 = n2 / space_dim[1]
            r3 = n3 / space_dim[2]

            # Use an error metric: sum of absolute differences between these ratios.
            error = abs(r1 - r2) + abs(r1 - r3) + abs(r2 - r3)

            # Update best if this triplet has a smaller error
            if best_error is None or error < best_error:
                best_error = error
                best_tuple = (n1, n2, n3)
    return best_tuple


def cell_size(space_dim, n1, n2, n3):
    """
    Given the dimensions of a space (space_dim) as a list or tuple of three numbers,
    and the number of divisions along each axis (n1, n2, n3), compute the size of each cell
    along each axis.

    Parameters:
        space_dim (list or tuple): The dimensions of the space, e.g. [2560, 960, 3456]
        n1 (int): Number of divisions along the first axis
        n2 (int): Number of divisions along the second axis
        n3 (int): Number of divisions along the third axis

    Returns:
        tuple: (s1, s2, s3) representing the size of each cell along each axis.
    """
    s1 = math.ceil(space_dim[0] / n1)
    s2 = math.ceil(space_dim[1] / n2)
    s3 = math.ceil(space_dim[2] / n3)
    return s1, s2, s3


# Example usage:
space_dim = [2560, 960, 3456]
n1, n2, n3 = find_grid(space_dim)
s1, s2, s3 = cell_size(space_dim, n1, n2, n3)
print("Optimal grid divisions:", n1, n2, n3)
print("Total cells:", n1 * n2 * n3)
print("Cell sizes:", s1, s2, s3)
