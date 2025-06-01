import lmdb
import numpy as np
import hashlib
import pickle
import blosc  # Faster compression
import time
import os

from timeit import timeit
from timeit import Timer

import fun

SHARD_COUNT = 100  # Number of LMDB databases (one per zero count)
LMDB_PATH_TEMPLATE = "RTP_shards_v2/shard_{:02d}.lmdb"  # LMDB file path pattern
MAP_SIZE = 10**6  # 0.1MB per shard (adjust as needed)

prob_maps_dir = "test_100chars\\"
n_files = 10_000


# possible_field_vals = [0, 1, 2, 3, 4, 5, 7]
# possible_shards = [str(field5) + str(field6) for field5 in possible_field_vals for field6 in possible_field_vals]
# Function to compute hash for uniqueness
def hash_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

# Function to determine the shard index
def get_shard_index(arr):
    zero_count = np.count_nonzero(arr == 0)  # Count zeros in the array
    return zero_count  # This determines which LMDB shard to use

# Function to retrieve a stored array

def retrieve_array(search_array):
    shard_index = get_shard_index(search_array)
    search_hash = hash_array(search_array).encode("utf-8")

    # Open correct shard
    env = lmdb.open(LMDB_PATH_TEMPLATE.format(shard_index), readonly=True)


    with env.begin() as txn:
        stats = txn.stat()
        num_records = stats["entries"]
        compressed_data = txn.get(search_hash)
        print(f"Shard {shard_index:02d} contains {num_records} records.")
        if compressed_data:
            original_array, computed_array = pickle.loads(blosc.decompress(compressed_data))
            print(f"Array found in shard {shard_index}!")
            return original_array, computed_array
        else:
            print("Array not found.")
            return None



# Initialize LMDB environments (one per shard)
shard_envs = {i: lmdb.open(LMDB_PATH_TEMPLATE.format(i), map_size=MAP_SIZE, writemap=True) for i in range(SHARD_COUNT)}

write_start_time_lmdb = time.time()
# Insert data in batches
for i in range(n_files):  # Example: Insert 1k records

    # Generate a random original array
    original_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)

    # Compute derived array
    computed_array = original_array.astype(np.int32) * 2

    # Determine the shard index
    shard_index = get_shard_index(original_array)

    # Compute hash as key
    array_hash = hash_array(original_array).encode("utf-8")

    # Serialize & compress data
    compressed_data = blosc.compress(pickle.dumps((original_array, computed_array)))

    with shard_envs[shard_index].begin(write=True) as txn:
        txn.put(array_hash, compressed_data)
    print(f"saved array number: {i}")

write_total_time_lmdb = time.time() - write_start_time_lmdb



# RETRIEVE FOR LMDB
total_retrieved_lmdb = 0
read_start_time_lmdb = time.time()
for i in range(n_files):  # Example: Insert 1k records

    # Generate a random original array
    search_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
    retrieved_data = retrieve_array(search_array)

    if retrieved_data:
        original, computed = retrieved_data
        print("Retrieved Original Array:\n", original)
        print("Retrieved Computed Array:\n", computed)
        total_retrieved_lmdb += 1
read_total_time_lmdb = time.time() - read_start_time_lmdb



# # WRITE SIMPLE
# write_start_time_simple = time.time()
# for i in range(n_files):  # Example: Insert 1k records
#
#     # Generate a random original array
#     original_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
#
#     if i == 99:
#         original_array = np.zeros((10, 10), dtype=np.uint8)
#         original_array[3, 3] = 2  # A known array
#
#     # Compute derived array
#     computed_array = original_array.astype(np.int32) * 2
#
#     board_state_100chars_code = fun.update_board_state_100chars_code(original_array)
#     prob_map_adv_100chars_file_name = board_state_100chars_code + '.npy'
#     fun.write_to_library(prob_maps_dir, computed_array, prob_map_adv_100chars_file_name)
#
# write_total_time_simple = time.time() - write_start_time_simple
#
#
# # RETRIEVE FOR SIMPLE
# read_start_time_simple = time.time()
# total_retrieved_simple = 0
#
# for i in range(n_files):  # Example: Insert 1k records
#     prob_map_library = os.listdir(prob_maps_dir)
#     # Generate a random original array
#     original_array = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
#
#     board_state_100chars_code = fun.update_board_state_100chars_code(original_array)
#     prob_map_100chars_file_name = board_state_100chars_code + '.npy'
#
#     if prob_map_100chars_file_name in prob_map_library:
#         full_path = prob_maps_dir + prob_map_100chars_file_name
#         prob_table = np.load(full_path, allow_pickle=True)
#         print (prob_table)
#         total_retrieved_simple += 1
#
# read_total_time_simple = time.time() - read_start_time_simple




print(f'write_total_time_lmdb: {write_total_time_lmdb}')
print(f'read_total_time_lmdb: {read_total_time_lmdb}')
print(f'total_retrieved_lmdb: {total_retrieved_lmdb}')


# print(f'write_total_time_simple: {write_total_time_simple}')
# print(f'read_total_time_simple: {read_total_time_simple}')
# print(f'total_retrieved_simple: {total_retrieved_simple}')