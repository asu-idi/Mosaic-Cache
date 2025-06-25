import csv
import random
import time
import os
from calculate_overlap import *
from find_grid import *


generated_folder = "generated_queries"
space_dim = [2560, 960, 3456]
grid_num = find_grid(space_dim)
grid_size = cell_size(space_dim, n1, n2, n3)


def check_query_validity(start, count, space_dim):
    """Check if query is within space_dim."""
    return all(0 <= start[i] < space_dim[i] and 0 < start[i] + count[i] <= space_dim[i] for i in range(len(space_dim)))


def rescale_query(query_size, count, space_dim):
    current_size = calculate_space_size(count)
    scale_factor = (query_size / current_size) ** (1 / len(space_dim))
    count = [max(1, int(c * scale_factor)) for c in count]
    return count


def generate_queries(space_dim, query_size_ratio, total_query_num, exact_match_ratio, partial_match_ratio):
    space_size = calculate_space_size(space_dim)
    query_size = int(space_size * query_size_ratio)
    exact_match_count = int(total_query_num * exact_match_ratio)
    random_query_num = total_query_num - exact_match_count

    queries = set()
    generated_num = 0

    print("Generating random queries...")
    start_time = time.perf_counter()  # Timer for query generation

    # Generate base random queries ensuring size approximation with variation
    while generated_num < random_query_num:
        start = [random.randint(0, space_dim[i] - 1) for i in range(len(space_dim))]
        count = [max(1, int((query_size / len(space_dim)) * random.uniform(0.5, 2.0))) for i in range(len(space_dim))]
        count = rescale_query(query_size, count, space_dim)
        if check_query_validity(start, count, space_dim):
            queries.add((tuple(start), tuple(count)))
            generated_num += 1
    end_time = time.perf_counter()
    print(f"Random query generation completed in {end_time - start_time:.4f} seconds")

    print("Calculating initial overlap ratio...")
    start_time = time.perf_counter()  # Timer for overlap calculation
    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim)
    print(f"Initial overlap ratio: {overlap_ratio:.4f}, overlap space: {overlap_space}, total volume: {total_volume}")
    end_time = time.perf_counter()
    print(f"Initial overlap calculation completed in {end_time - start_time:.4f} seconds")

    # print("[GPU]Calculating initial overlap ratio...")
    # start_time = time.perf_counter()  # Timer for overlap calculation
    # overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim, backend='gpu')
    # end_time = time.perf_counter()
    # print(f"[GPU]Initial overlap calculation completed in {end_time - start_time:.4f} seconds")
    # print(f"Initial overlap ratio: {overlap_ratio:.4f}")

    print("Adjusting queries to achieve target partial overlap ratio...")
    start_time = time.perf_counter()  # Timer for adjustment loop
    attempt_num, update_num = 1, 1
    adjust_factor = 1000
    # Adjust queries to achieve the target partial overlap ratio
    while abs(overlap_ratio - partial_match_ratio) > 0.01 * partial_match_ratio:
        attempt_num += 1
        old_query = random.sample(queries, 1)[0]
        new_query = (list(old_query[0]), list(old_query[1]))
        shrink = True if overlap_ratio > partial_match_ratio else False
        if shrink:
            # move start to boundary, while keeping count constant and the query valid
            for i in range(len(space_dim)):
                if attempt_num % adjust_factor == 0:
                    # random a cell of the space to reduce the overlap
                    new_query[0][i] = random.randint(0, grid_num[i]) * grid_size[i]
                    new_query[1][i] = grid_size[i]
                else:
                    if random.random() < 0.3:
                        continue
                    new_query[0][i] = max(0, min(space_dim[i] - 1, int(old_query[0][i] * (random.random() + 0.8)) if old_query[0][i] > space_dim[i] // 2 else int(old_query[0][i] * (random.random() + 0.3))))
        else:
            # move start to center, while keeping count constant and the query valid
            for i in range(len(space_dim)):
                if attempt_num % adjust_factor == 0:
                    # slightly adjust start and count
                    new_query[0][i] = max(0, min(space_dim[i] - 1, int(old_query[0][i] * (random.random() * 0.02 + 0.99))))
                    new_query[1][i] = max(1, int(old_query[1][i] * (random.random() * 0.02 + 0.99)))
                else:
                    if random.random() < 0.3:
                        continue
                    new_query[0][i] = max(0, min(space_dim[i] - 1, int(old_query[0][i] * (random.random() + 0.3)) if old_query[0][i] > space_dim[i] // 2 else int(old_query[0][i] * (random.random() + 0.8))))

        rescale_query(query_size, new_query[1], space_dim)
        if attempt_num % adjust_factor == 0:
            old_query = random.sample(queries, 1)[0]

        if check_query_validity(new_query[0], new_query[1], space_dim):
            updated, overlap_space = update_overlap_ratio(matrix, old_query, new_query, overlap_space, shrink)
            if updated:
                queries.remove((tuple(old_query[0]), tuple(old_query[1])))
                queries.add((tuple(new_query[0]), tuple(new_query[1])))
                overlap_ratio = overlap_space / total_volume
                update_num += 1
                print(f"Overlap ratio: {overlap_ratio:.4f}, Attempt: {attempt_num}, Update: {update_num}")

    end_time = time.perf_counter()
    print(f"Query adjustment completed in {end_time - start_time:.4f} seconds")

    occupy_space = calculate_occupancy(matrix)
    occupancy_ratio = occupy_space / space_size

    overlap_ratio_verify, overlap_space_verify, total_volume_verify, matrix_verify = calculate_overlap_ratio(queries, space_dim)
    print(f"Final overlap ratio: {overlap_ratio_verify:.4f}, overlap space: {overlap_space_verify}, total volume: {total_volume_verify}")

    return queries, occupancy_ratio


def generator(space_dim, query_size_ratio, total_query_num, exact_match_ratio, partial_match_ratio):
    print("Generating queries...", "space_dim: ", space_dim, ", query_size_ratio: ", query_size_ratio, ", total_queries: ", total_query_num, ", exact_match_ratio: ", exact_match_ratio, ", partial_match_ratio: ", partial_match_ratio)
    queries = False
    # for i in range(3):
    queries, occupancy_ratio = generate_queries(space_dim, query_size_ratio, total_query_num, exact_match_ratio, partial_match_ratio)
        # if queries:
            # break

    if not queries:
        print("Failed to generate queries")
        return False

    filename = f"dim{'x'.join(map(str, space_dim))}_ratio{query_size_ratio}_num{total_query_num}_exact{exact_match_ratio}_partial{partial_match_ratio}_occupancy{occupancy_ratio:.4f}.csv"

    # Write to CSV
    with open(f"{generated_folder}/{filename}", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "count"])
        for start, count in queries:
            writer.writerow([start, count])

        # Generate exact match queries by selecting from existing queries
        for _ in range(int(total_query_num * exact_match_ratio)):
            # queries.add(random.sample(queries, 1)[0])
            start, count = random.sample(queries, 1)[0] # Select a random query
            writer.writerow([start, count])

    print(f"Queries saved to {filename}")
    return True


if __name__ == "__main__":
    # Example usage
    # query_size_ratio = 0.0001
    # total_query_num = 10000
    # exact_match_ratio = 0.1
    # partial_match_ratio = 0.3
    #
    # res = generator(space_dim, query_size_ratio, total_query_num, exact_match_ratio, partial_match_ratio)

    # Generate queries with different parameters
    # 100 KB, 1 MB, 10 MB queries
    query_size_ratio_list = [1.5e-6, 1.5e-5, 1.5e-4]
    total_query_num_list = [100, 1000, 10000]
    exact_match_ratio_list = [0.75, 0.5, 0.25, 0]
    partial_match_ratio_list = [0.75, 0.5, 0.25]
    # Matrix: record all overlap information
    # First query loop: For each query, the corresponding cell Matrix[i][j] += 1
    # Second query loop: sum(overlap size of each query) / sum(each query size)
    # update -> shrink or expand, remove old random query, add new random query
    # partial_match_ratio converge to target value

    for query_size_ratio in query_size_ratio_list:
        for total_query_num in total_query_num_list:
            for exact_match_ratio in exact_match_ratio_list:
                for partial_match_ratio in partial_match_ratio_list:

                    already_generated = False
                    already_generated_files = os.listdir(generated_folder)
                    filename_prefix = f"dim{'x'.join(map(str, space_dim))}_ratio{query_size_ratio}_num{total_query_num}_exact{exact_match_ratio}_partial{partial_match_ratio}"

                    for filename in already_generated_files:
                        if filename.startswith(filename_prefix):
                            already_generated = True
                            break
                    if already_generated:
                        print(f"Queries for {filename_prefix} already generated")
                        continue

                    print(f"Generating queries for {filename_prefix}...")

                    res = generator(space_dim, query_size_ratio, total_query_num, exact_match_ratio, partial_match_ratio)
