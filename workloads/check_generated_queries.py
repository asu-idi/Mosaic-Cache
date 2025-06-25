import os
import ast
import pandas as pd
from calculate_overlap import *

def verify_query_file(file_path):
    # parse the file name to get the parameters
    # f"dim{'x'.join(map(str, space_dim))}_ratio{query_size_ratio}_num{total_query_num}_exact{exact_match_ratio}_partial{partial_match_ratio}_occupancy{occupancy_ratio:.4f}.csv"
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    file_name_parts = file_name.split('_')
    space_dim = file_name_parts[0][3:].split('x')
    # convert the space dimensions to integers
    space_dim = [int(dim) for dim in space_dim]
    query_size_ratio = float(file_name_parts[1][5:])
    total_query_num = float(file_name_parts[2][3:])
    exact_match_ratio = float(file_name_parts[3][5:])
    partial_match_ratio = float(file_name_parts[4][7:])
    occupancy_ratio = float(file_name_parts[5][9:])

    # Read the queries from the file
    queries = pd.read_csv(file_path, header=0, index_col=False)
    print(f"Read {len(queries)} queries from {file_path}")

    assert len(queries) == int(total_query_num), f"Expected {total_query_num} queries, but found {len(queries)} queries"

    # deduplicate the queries
    queries.drop_duplicates(inplace=True)
    print(f"Removed duplicates. Remaining queries: {len(queries)}")
    assert len(queries) == int(total_query_num * (1 - exact_match_ratio)), f"Expected {total_query_num} queries, but found {len(queries)} queries"

    queries_set = set()
    # check the queries for validity
    for index, row in queries.iterrows():
        start = eval(row['start'])
        count = eval(row['count'])
        assert len(start) == len(space_dim), f"Query {index} has invalid start dimensions"
        assert len(count) == len(space_dim), f"Query {index} has invalid count dimensions"
        for i in range(len(space_dim)):
            assert start[i] >= 0 and start[i] < int(space_dim[i]), f"Query {index} has invalid start dimensions"
            assert count[i] > 0 and count[i] <= int(space_dim[i]), f"Query {index} has invalid count dimensions"
        queries_set.add((tuple(start), tuple(count)))
    print(f"All queries are valid")

    # check the overlap ratio
    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries_set, space_dim)
    print(f"Overlap ratio calculated successfully")

    assert abs(overlap_ratio - partial_match_ratio) < 0.01 * partial_match_ratio, f"Expected overlap ratio {partial_match_ratio}, but found {overlap_ratio}"
    print(f"Overlap ratio matches expected value")

    # check the occupancy ratio
    occupy_space = calculate_occupancy(matrix)
    occupancy_ratio_cal = occupy_space / calculate_space_size(space_dim)
    print(f"Occupancy ratio calculated successfully")

    assert abs(occupancy_ratio_cal - occupancy_ratio) < 1e-4, f"Expected occupancy ratio {occupancy_ratio}, but found {occupancy_ratio_cal}"
    print(f"Occupancy ratio matches expected value")

    # the query space is valid
    space_size = calculate_space_size(space_dim)
    query_size = int(space_size * query_size_ratio)
    print(f"Space size: {space_size}, Query size: {query_size}")

    total_query_size = 0
    for query in queries_set:
        total_query_size += calculate_space_size(query[1])
    print(f"Total query size: {total_query_size}")
    average_query_size = total_query_size / len(queries_set)

    print(f"Average query size: {average_query_size}")
    assert abs(average_query_size / query_size - 1) < 0.05, f"Expected average query size {query_size}, but found {average_query_size}"


def calculate_overall_ratio(file_path):
    # Read the queries from the file
    query_df = pd.read_csv(file_path, header=0, index_col=False)
    print(f"Read {len(query_df)} queries from {file_path}")

    # Convert string tuples to actual tuples
    query_df['start'] = query_df['start'].apply(ast.literal_eval)
    query_df['count'] = query_df['count'].apply(ast.literal_eval)

    # Convert dataframe to list of tuples
    queries = list(zip(query_df['start'], query_df['count']))

    print(f"Calculating overlap ratio for {len(queries)} queries")

    # Call the function
    overlap_ratio, overlap_space, total_volume, matrix = calculate_overlap_ratio(queries, space_dim=[2560, 960, 3456])

    print(f"Overlap ratio: {overlap_ratio:.4f}, Overlap space: {overlap_space}, Total volume: {total_volume}")
    print()


if __name__ == '__main__':
    folder = "generated_queries"
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            # verify_query_file(file_path)
            calculate_overall_ratio(file_path)
            print(f"Verified {file_path}")