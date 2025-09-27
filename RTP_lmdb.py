import lmdb
import numpy as np
import hashlib
import pickle
import blosc  # Faster compression
import time
import os

import fun

def initialize_RTP_lmdb(SHARD_COUNT, LMDB_PATH_TEMPLATE, MAP_SIZE):
    """
    Initialize LMDB environments (one per shard)
    """
    shard_envs =\
        {i: lmdb.open(LMDB_PATH_TEMPLATE.format(i), map_size=MAP_SIZE, writemap=True) for i in range(SHARD_COUNT+1)}
    return shard_envs


def get_shard_index(arr):
    zero_count = np.count_nonzero(arr == 0)  # Count zeros in the array
    return zero_count  # This determines which LMDB shard to use


def hash_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_to_RTP_lmdb(known_board, occurances, shard_envs):
    """
    Save known board and computed occurances array into RTP library (lmdb format).
    No overwriting allowed - if the given hash already exists in database, no new data is written.
    """
    shard_index = get_shard_index(known_board)
    array_hash = hash_array(known_board).encode("utf-8")

    # Serialize & compress data
    compressed_data = blosc.compress(pickle.dumps((known_board, occurances)))

    with shard_envs[shard_index].begin(write=True) as txn:
        inserted = txn.put(array_hash, compressed_data, overwrite=False)
        if not inserted:
            print(f"Key already exists. Skipping.")
        else:
            print(f"saved array to RTP")


def load_array_from_RTP_lmdb(known_board, LMDB_PATH_TEMPLATE):
    """
    Retrieve computed occurances array from RTP for given known board state

    :return: occurances array
    """
    shard_index = get_shard_index(known_board)
    search_hash = hash_array(known_board).encode("utf-8")

    # Open correct shard
    env = lmdb.open(LMDB_PATH_TEMPLATE.format(shard_index), readonly=True)

    with env.begin() as txn:
        stats = txn.stat()
        num_records = stats["entries"]
        compressed_data = txn.get(search_hash)
        print(f"Shard {shard_index:02d} contains {num_records} records.")
        if compressed_data:
            original_array, computed_array = pickle.loads(blosc.decompress(compressed_data))
            print(f"Array found in shard {shard_index}")
            return computed_array
        else:
            print("Array not found.")
            return None

if __name__ == "__main__":
    SHARD_COUNT = 100  # Number of LMDB database shards
    LMDB_PATH_TEMPLATE = "prob_maps_adv_lmdb\\shard_{:02d}.lmdb"  # LMDB file path pattern
    MAP_SIZE = 10 ** 8  # 100MB per shard (can be increased at any time)
    shard_envs = initialize_RTP_lmdb(SHARD_COUNT, LMDB_PATH_TEMPLATE, MAP_SIZE)

    #board_state_100chars_code = "0700755555777777777774733372777477777270747000777774700000707770000007000000000077770000103337000070"
    board_state_100chars_code = "0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000"
    #board_state_100chars_code = "0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000"
    known_board = np.zeros((100), dtype="int16")
    for id, val in enumerate(board_state_100chars_code):
        known_board[id] = int(val)
    known_board = known_board.reshape(10, 10)

    occurances = load_array_from_RTP_lmdb(known_board, LMDB_PATH_TEMPLATE)
    print(occurances)
    if occurances is None:
        occurances = np.array([0,2,5])
        #save_to_RTP_lmdb(known_board, occurances, shard_envs)











