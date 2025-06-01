import numpy as np
import os
import lmdb

import RTP_lmdb as rtp


SHARD_COUNT = 100  # Number of LMDB databases (one per zero count)
LMDB_PATH_TEMPLATE = "prob_density_maps_mc_lmdb\\shard_{:02d}.lmdb"  # LMDB file path pattern
MAP_SIZE = 10 ** 8  # 100MB per shard (can be increased at any time)


source_dir = "prob_density_maps_mc_100chars\\"


prob_map_library = os.listdir(source_dir )
tot_files = len(prob_map_library)

shard_envs = rtp.initialize_RTP_lmdb(SHARD_COUNT, LMDB_PATH_TEMPLATE, MAP_SIZE)

for n, prob_map_file_name in enumerate(prob_map_library):
    board_state_100chars_code = prob_map_file_name.strip('.npy')
    # if board_state_100chars_code != "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000":
    #     continue
    known_board = np.zeros((100), dtype = "int16")
    for id, val in enumerate(board_state_100chars_code):
        known_board[id] = int(val)
    known_board = known_board.reshape(10, 10)

    occurances = np.load(source_dir + prob_map_file_name)

    rtp.save_to_RTP_lmdb(known_board, occurances, shard_envs)
    print(f' Done {n+1} out of {tot_files}')


all_records = 0
for shard_index in shard_envs:
    env = lmdb.open(LMDB_PATH_TEMPLATE.format(shard_index), readonly=True)
    with env.begin() as txn:
        stats = txn.stat()
        num_records = stats["entries"]
        print (num_records)
        all_records = all_records + num_records

print(f'all records = {all_records}')