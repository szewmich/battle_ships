# Combined module: generate_uniform_and_biased_boards.py
# Contains the full uniform board enumerator and biased sampler

import random
import math
import csv
from collections import deque

# -------------------------
# UNIFORM ENUMERATION PART (your original module)
# -------------------------

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

# -------------------------
# RELATIONAL BIAS PART
# -------------------------

def find_2_ship_components(board_str):
    grid = [[board_str[r*BOARD_SIZE + c] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]
    visited = [[False]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    comps = []
    from collections import deque
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r][c] == '2' and not visited[r][c]:
                q = deque([(r,c)])
                visited[r][c] = True
                comp = []
                while q:
                    rr, cc = q.popleft()
                    comp.append((cc, rr))  # store as (x,y)
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if not visited[nr][nc] and grid[nr][nc] == '2':
                                visited[nr][nc] = True
                                q.append((nr, nc))
                comps.append(comp)
    return comps

def min_distance_between_2ships(board_str):
    comps = find_2_ship_components(board_str)
    if len(comps) < 2:
        return None
    a, b = comps[0], comps[1]
    return min(abs(x1-x2) + abs(y1-y2) for (x1,y1) in a for (x2,y2) in b)

def generate_biased_sample(all_boards, N=10000, alpha=0.5, out_csv="bias_checks\\biased_2seg_close_10000.csv"):
    distances = []
    weights = []
    for s in all_boards:
        d = min_distance_between_2ships(s)
        if d is None:
            d = 99
            w = 0.001
        else:
            w = math.exp(-alpha * d)   # encourage closeness
        distances.append(d)
        weights.append(w)

    sampled = random.choices(all_boards, weights=weights, k=N)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for board in sampled:
            writer.writerow([board])

    print("Biased sample saved to", out_csv)
    return sampled

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    empty = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    all_boards = []
    backtrack_enumerate(empty, 0, all_boards)
    print("Full uniform enumeration count:", len(all_boards))

    sampled = generate_biased_sample(all_boards, N=10000)
    print("Example biased sample size:", len(sampled))
