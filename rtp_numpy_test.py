import numpy as np
import time
import os

import fun

SHARD_COUNT = 100  # Number of LMDB databases (one per zero count)
LMDB_PATH_TEMPLATE = "RTP_shards/shard_{:02d}.lmdb"  # LMDB file path pattern
MAP_SIZE = 10**8  # 10GB per shard (adjust as needed)

prob_maps_dir = "test_100chars\\"
prob_maps_condensed_dir = "test_100chars_condensed\\"
n_files = 100_000


# Function to determine the shard index
def get_shard_index(arr):
    zero_count = np.count_nonzero(arr == 0)  # Count zeros in the array
    return zero_count  # This determines which LMDB shard to use

# Function to retrieve a stored array

def retrieve_array(search_array):
    shard_index = get_shard_index(search_array)
    file_name = "_" + str(shard_index) + ".npy"
    path = prob_maps_condensed_dir + file_name
    condensed_library = os.listdir(prob_maps_condensed_dir)
    if file_name in condensed_library:
        numpy_database = np.load(path, allow_pickle=True)
    else:
        return None
    board_state_100chars_code = fun.update_board_state_100chars_code(search_array)

    result = numpy_database[numpy_database['Label'] == board_state_100chars_code]['Array']

    if result.size > 0:
        return result
    else:
        return None


# Define structured array dtype: one string field and one object field for the array
custom_dtype = [('Label', 'U100'), ('Array', 'O')]  # 'U10' for string (up to 10 chars), 'O' for object (array)


# WRITE CONDENSED
write_start_time_condensed = time.time()
for i in range(n_files):  # Example: Insert 1k records

    condensed_library = os.listdir(prob_maps_condensed_dir)

    # Generate a random original array
    original_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)

    # Compute derived array
    computed_array = original_array.astype(np.int32) * 2

    # Determine the shard index
    shard_index = get_shard_index(original_array)

    file_name = "_" + str(shard_index) + ".npy"
    path = prob_maps_condensed_dir + file_name
    if file_name not in condensed_library:
        numpy_database = np.empty(0, dtype=custom_dtype)
    else:
        numpy_database = np.load(path, allow_pickle=True)

    board_state_100chars_code = fun.update_board_state_100chars_code(original_array)
    new_entry = np.array([(board_state_100chars_code, computed_array)], dtype=custom_dtype)  # Create new structured array with one entry

    numpy_database = np.append(numpy_database, new_entry)

    np.save(path, numpy_database)

    print(f"saved array number: {i}")

write_total_time_condensed = time.time() - write_start_time_condensed



# RETRIEVE FOR CONDENSED
total_retrieved_condensed = 0
read_start_time_condensed = time.time()
for i in range(n_files):  # Example: Insert 1k records

    # Generate a random original array
    search_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
    retrieved_data = retrieve_array(search_array)

    if retrieved_data is not None:
        computed = retrieved_data
        print("Retrieved Computed Array:\n", computed)
        total_retrieved_condensed += 1
read_total_time_condensed = time.time() - read_start_time_condensed



# WRITE SIMPLE
write_start_time_simple = time.time()
for i in range(n_files):  # Example: Insert 1k records

    # Generate a random original array
    original_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)

    # Compute derived array
    computed_array = original_array.astype(np.int32) * 2

    board_state_100chars_code = fun.update_board_state_100chars_code(original_array)
    prob_map_adv_100chars_file_name = board_state_100chars_code + '.npy'
    fun.write_to_library(prob_maps_dir, computed_array, prob_map_adv_100chars_file_name)

write_total_time_simple = time.time() - write_start_time_simple


# RETRIEVE FOR SIMPLE
read_start_time_simple = time.time()
total_retrieved_simple = 0

for i in range(n_files):  # Example: Insert 1k records
    prob_map_library = os.listdir(prob_maps_dir)
    # Generate a random original array
    original_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)

    board_state_100chars_code = fun.update_board_state_100chars_code(original_array)
    prob_map_100chars_file_name = board_state_100chars_code + '.npy'

    if prob_map_100chars_file_name in prob_map_library:
        full_path = prob_maps_dir + prob_map_100chars_file_name
        prob_table = np.load(full_path, allow_pickle=True)
        print (prob_table)
        total_retrieved_simple += 1

read_total_time_simple = time.time() - read_start_time_simple




print(f'write_total_time_condensed: {write_total_time_condensed}')
print(f'read_total_time_condensed: {read_total_time_condensed}')
print(f'total_retrieved_condensed: {total_retrieved_condensed}')


print(f'write_total_time_simple: {write_total_time_simple}')
print(f'read_total_time_simple: {read_total_time_simple}')
print(f'total_retrieved_simple: {total_retrieved_simple}')