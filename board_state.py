class BoardState:
    def __init__(self, size=6):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.highlights = {}  # Store highlight rectangles

    def is_within_bounds(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def can_place_ship(self, row, col, length, is_vertical):
        if not is_vertical:
            # Check horizontal placement
            if col + length > self.size:
                return False
            # Check if any cell in the ship's path is occupied
            for c in range(col, col + length):
                if not self.is_cell_available(row, c):
                    return False
        else:
            # Check vertical placement
            if row + length > self.size:
                return False
            # Check if any cell in the ship's path is occupied
            for r in range(row, row + length):
                if not self.is_cell_available(r, col):
                    return False
        return True

    def is_cell_available(self, row, col):
        # Check if cell is within bounds
        if not self.is_within_bounds(row, col):
            return False
            
        # Check the cell itself
        if self.grid[row][col] != 0:
            return False
            
        # Check adjacent cells (including diagonals)
        for r in range(max(0, row-1), min(self.size, row+2)):
            for c in range(max(0, col-1), min(self.size, col+2)):
                cell_value = self.grid[r][c]
                if cell_value != 0 and cell_value != -1:  # Allow temporary markers
                    return False
        return True

    def place_ship(self, row, col, length, ship_type, is_vertical):
        if not self.can_place_ship(row, col, length, is_vertical):
            return False
            
        # First mark adjacent cells to prevent other ships from touching
        if not is_vertical:
            for c in range(col, col + length):
                # Mark cells above and below
                if row > 0:
                    self.grid[row-1][c] = -1
                if row < self.size-1:
                    self.grid[row+1][c] = -1
                # Mark diagonal cells
                if c > 0:
                    if row > 0:
                        self.grid[row-1][c-1] = -1
                    if row < self.size-1:
                        self.grid[row+1][c-1] = -1
                if c < self.size-1:
                    if row > 0:
                        self.grid[row-1][c+1] = -1
                    if row < self.size-1:
                        self.grid[row+1][c+1] = -1
        else:
            for r in range(row, row + length):
                # Mark cells to left and right
                if col > 0:
                    self.grid[r][col-1] = -1
                if col < self.size-1:
                    self.grid[r][col+1] = -1
                # Mark diagonal cells
                if r > 0:
                    if col > 0:
                        self.grid[r-1][col-1] = -1
                    if col < self.size-1:
                        self.grid[r-1][col+1] = -1
                if r < self.size-1:
                    if col > 0:
                        self.grid[r+1][col-1] = -1
                    if col < self.size-1:
                        self.grid[r+1][col+1] = -1
                        
        # Then place the ship
        if not is_vertical:
            for c in range(col, col + length):
                self.grid[row][c] = ship_type
        else:
            for r in range(row, row + length):
                self.grid[r][col] = ship_type
                
        return True

    def remove_ship(self, ship_type):
        # Create a copy of the grid to avoid modifying while iterating
        to_clear = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] == ship_type:
                    to_clear.append((row, col))
        
        # Clear all cells of this ship
        for row, col in to_clear:
            self.grid[row][col] = 0
            
        # Clear adjacent cells as well to prevent "ghost" occupied spaces
        for row, col in to_clear:
            for r in range(max(0, row-1), min(self.size, row+2)):
                for c in range(max(0, col-1), min(self.size, col+2)):
                    if self.grid[r][c] == -1:  # Clear temporary markers
                        self.grid[r][c] = 0

    def get_state_string(self):
        # Convert temporary markers (-1) to empty spaces (0) when generating the string
        return ''.join('0' if cell == -1 else str(cell) for row in self.grid for cell in row)

    def is_complete(self):
        # Count ships of each type
        ship_counts = {2: 0, 3: 0}
        for row in self.grid:
            for cell in row:
                if cell in ship_counts:
                    ship_counts[cell] += 1
                    
        # Check if we have correct number of ship segments
        return ship_counts[2] == 4 and ship_counts[3] == 3  # 2x2-segment + 1x3-segment