import numpy as np
# import cupy as cp
import math


def calculate_space_size(space_dim):
    """Calculate the total size of the space given the dimensions."""
    return np.prod(space_dim)


def calculate_overlap_ratio(queries, space_dim, backend='cpu'):
    """Calculate the overlap ratio using a matrix-based approach."""

    # Initialize the space matrix with 0
    if backend == 'cpu':
        matrix = np.zeros(space_dim, dtype=np.uint16)
    # elif backend == 'gpu':
        # matrix = cp.zeros(space_dim, dtype=cp.uint16)

    # Update the matrix for each query region
    for start, count in queries:
        slices = tuple(slice(start[i], start[i] + count[i]) for i in range(len(space_dim)))
        matrix[slices] += 1  # Increment overlap count in the region

    # Calculate the total overlapped volume
    overlap_space = 0
    for start, count in queries:
        slices = tuple(slice(start[i], start[i] + count[i]) for i in range(len(space_dim)))

        # Count overlapped cells in this query's region (where matrix > 1)
        if backend == 'cpu':
            overlap_space += np.sum(matrix[slices] > 1)
        # elif backend == 'gpu':
            # overlap_space += cp.sum(matrix[slices] > 1).get()


    # Calculate total volume of all queries
    total_volume = sum(math.prod(count) for _, count in queries)

    overlap_ratio = overlap_space / total_volume if total_volume > 0 else 0

    # Return the overlap ratio
    return overlap_ratio, overlap_space, total_volume, matrix


def update_overlap_ratio(matrix, old_query, new_query, overlap_space, shrink=False, backend='cpu'):
    old_slices = tuple(slice(old_query[0][i], old_query[0][i] + old_query[1][i]) for i in range(len(old_query[0])))
    new_slices = tuple(slice(new_query[0][i], new_query[0][i] + new_query[1][i]) for i in range(len(new_query[0])))

    if backend == 'cpu':
        # Compute boolean masks once
        old_mask = matrix[old_slices] > 1
        new_mask = matrix[new_slices] > 1

        # Compute the initial overlap sums
        old_overlap_before = np.sum(matrix[old_slices][old_mask])
        new_overlap_before = np.sum(matrix[new_slices][new_mask])

        # Update matrix efficiently
        matrix[old_slices] -= 1
        matrix[new_slices] += 1

        # Recompute masks only for modified regions
        old_mask_after = matrix[old_slices] > 1
        new_mask_after = matrix[new_slices] > 1

        # Compute the new overlap sums
        old_overlap_after = np.sum(matrix[old_slices][old_mask_after])
        new_overlap_after = np.sum(matrix[new_slices][new_mask_after])
    elif backend == 'gpu':
        # old_overlap = cp.count_nonzero(matrix[old_slices] > 1).get()
        # new_overlap = cp.count_nonzero(matrix[new_slices] > 0).get()
        pass

    overlap_change = np.add(
        np.subtract(old_overlap_after, old_overlap_before, dtype=np.int64),
        np.subtract(new_overlap_after, new_overlap_before, dtype=np.int64),
        dtype=np.int64
    )
    if overlap_change < 0 and shrink:
        overlap_space += overlap_change
        return True, overlap_space
    elif overlap_change > 0 and not shrink:
        overlap_space += overlap_change
        return True, overlap_space
    else:
        # reverse the matrix update
        matrix[old_slices] += 1
        matrix[new_slices] -= 1
        return False, overlap_space


def calculate_occupancy(matrix, backend='cpu'):
    """Calculate the occupancy ratio using a matrix-based approach."""

    # Count the number of occupied cells
    if backend == 'cpu':
        occupied_space = np.sum(matrix > 0)
    # elif backend == 'gpu':
        # occupied_space = cp.sum(matrix > 0).get()

    return occupied_space


def test_overlap_ratio_2D():
    queries = [
        ((0, 0), (3, 3)),
        ((1, 1), (3, 3)),
        ((2, 2), (2, 2)),
        ((4, 4), (2, 2))
    ]
    space_dim = [6, 6]

    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim)

    print(matrix)
    assert abs(overlap_ratio - 0.5769) < 1e-4
    assert overlap_space == 15
    assert total_volume == 26

    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim, backend='gpu')
    print(matrix)
    assert abs(overlap_ratio - 0.5769) < 1e-4
    assert overlap_space == 15
    assert total_volume == 26


def test_overlap_ratio_3D():
    queries = [
        ((0, 0, 0), (3, 3, 3)),  # 3x3x3 region
        ((1, 1, 1), (3, 3, 3)),  # Overlaps with first region
        ((2, 2, 2), (2, 2, 2)),  # Overlaps with first two regions
        ((4, 4, 4), (2, 2, 2))   # Separate region, no overlap
    ]
    space_dim = [6, 6, 6]  # 6x6x6 space

    # CPU version
    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim)

    print("CPU Matrix:\n", matrix)
    print(f"CPU Overlap Ratio: {overlap_ratio:.4f}, Overlap Space: {overlap_space}, Total Volume: {total_volume}")

    # Expected values (these values should be verified based on actual calculations)
    assert abs(overlap_ratio - 0.4429) < 1e-4  # Adjust expected value if necessary
    assert overlap_space == 31  # Adjust based on manual calculation
    assert total_volume == 70   # Adjust based on manual calculation

    # GPU version
    overlap_ratio_gpu, overlap_space_gpu, total_volume_gpu, matrix_gpu = calculate_overlap_ratio(queries, space_dim, backend='gpu')

    print("GPU Matrix:\n", matrix_gpu)
    print(f"GPU Overlap Ratio: {overlap_ratio_gpu:.4f}, Overlap Space: {overlap_space_gpu}, Total Volume: {total_volume_gpu}")

    assert abs(overlap_ratio_gpu - 0.4429) < 1e-4
    assert overlap_space_gpu == 31
    assert total_volume_gpu == 70


def test_update_overlap_ratio():
    queries = [
        ((0, 0), (3, 3)),
        ((1, 1), (3, 3)),
        ((2, 2), (2, 2)),
        ((4, 4), (2, 2))
    ]
    space_dim = [6, 6]

    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim)
    print(overlap_ratio, overlap_space, total_volume)
    print(matrix)

    # **Test Case 1: Overlap Decreases (shrink=True)**
    old_query = ((1, 1), (3, 3))  # Old region
    new_query = ((0, 0), (1, 1))  # New region, overlap decreases
    result, overlap_space = update_overlap_ratio(matrix, old_query, new_query, overlap_space, shrink=True)
    print(result, overlap_space)
    print(matrix)

    assert result is True

    queries.remove(((1, 1), (3, 3)))
    queries.append(((0, 0), (1, 1)))
    overlap_ratio_verify, overlap_space_verify, total_volume_verify, matrix_verify = calculate_overlap_ratio(queries, space_dim)
    print("------------------------------")
    print(overlap_ratio_verify, overlap_space_verify, total_volume_verify)
    print(matrix_verify)

    assert overlap_space == overlap_space_verify  # Overlap reduced to 4

    # **Test Case 2: Overlap Increases (shrink=False)**
    old_query = ((0, 0), (1, 1))  # Old region
    new_query = ((2, 2), (3, 3))  # New region, overlap increases
    result, overlap_space = update_overlap_ratio(matrix, old_query, new_query, overlap_space, shrink=False)

    assert result is True
    queries.remove(((0, 0), (1, 1)))
    queries.append(((2, 2), (3, 3)))
    overlap_ratio_verify, overlap_space_verify, total_volume_verify, matrix_verify = calculate_overlap_ratio(queries, space_dim)
    print("------------------------------")
    print(overlap_ratio_verify, overlap_space_verify, total_volume_verify)
    print(matrix_verify)

    assert overlap_space == overlap_space_verify  # Overlap increased to 9

    # **Test Case 3: Overlap Unchanged**
    old_query = ((2, 2), (3, 3))  # Old region
    new_query = ((2, 2), (2, 2))  # Same region
    result, overlap_space = update_overlap_ratio(matrix, old_query, new_query, overlap_space, shrink=False)

    assert result is False  # Overlap unchanged
    print(matrix)

    print("All tests passed!")

def test_update_overlap_ratio_v2():
    queries = [
        ((0, 0), (3, 3)),
        ((1, 1), (3, 3)),
        ((2, 2), (2, 2)),
        ((4, 4), (2, 2))
    ]

    space_dim = [6, 6]
    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim)
    print(overlap_ratio, overlap_space, total_volume)
    print(matrix)

    old_query = ((1, 1), (3, 3))
    new_query = ((0, 0), (1, 1))
    shrink = True
    updated, overlap_space = update_overlap_ratio(matrix, old_query, new_query, overlap_space, shrink)
    print(updated, overlap_space)
    print(matrix)

    queries.remove(((1, 1), (3, 3)))
    queries.append(((0, 0), (1, 1)))
    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim)
    print(overlap_ratio, overlap_space, total_volume)
    print(matrix)
