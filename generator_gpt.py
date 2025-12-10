# save as generate_uniform_boards.py
import random
import pandas as pd

BOARD_SIZE = 6
SHIP_SIZES = [3, 2, 2]

def check_spacing(board, coords):
    for (x, y) in coords:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    if (nx, ny) not in coords and board[nx][ny] != 0:
                        return False
    return True

def list_valid_placements(board, size):
    placements = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            # rightwards (horizontal)
            if y + size <= BOARD_SIZE:
                coords = [(x, y + i) for i in range(size)]
                if all(board[cx][cy] == 0 for cx, cy in coords) and check_spacing(board, coords):
                    placements.append(('H', x, y))
            # downwards (vertical)
            if x + size <= BOARD_SIZE:
                coords = [(x + i, y) for i in range(size)]
                if all(board[cx][cy] == 0 for cx, cy in coords) and check_spacing(board, coords):
                    placements.append(('V', x, y))
    return placements

def apply_placement(board, placement, size):
    orientation, x, y = placement
    nb = [row[:] for row in board]
    if orientation == 'H':
        for i in range(size):
            nb[x][y+i] = size
    else:
        for i in range(size):
            nb[x+i][y] = size
    return nb

def backtrack_enumerate(board, ship_index, all_boards):
    if ship_index >= len(SHIP_SIZES):
        s = ''.join(str(board[x][y]) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE))
        all_boards.append(s)
        return
    size = SHIP_SIZES[ship_index]
    placements = list_valid_placements(board, size)
    for p in placements:
        nb = apply_placement(board, p, size)
        backtrack_enumerate(nb, ship_index + 1, all_boards)

def main(output_csv_path="bias_checks\\random_boards_bruteforce_uniform.csv", sample_size=30_000):
    empty = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    all_boards = []
    backtrack_enumerate(empty, 0, all_boards)
    print("Total full boards enumerated:", len(all_boards))

    if sample_size > len(all_boards):
        raise ValueError("Requested more samples than distinct full boards")
    sampled = random.sample(all_boards, sample_size)
    df = pd.DataFrame(sampled, columns=["board"])
    df.to_csv(output_csv_path, index=False)
    print("Saved", sample_size, "boards to", output_csv_path)

if __name__ == "__main__":
    main()


