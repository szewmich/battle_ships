import tkinter as tk
from tkinter import ttk
import csv
import os
from board_state import BoardState

class Ship:
    def __init__(self, canvas, x, y, length, tag, board):
        self.canvas = canvas
        self.length = length
        self.tag = tag
        self.board = board
        self.is_vertical = False
        self.width = 40 * length
        self.height = 40
        self.ship_type = 3 if length == 3 else 2
        self.target_row = None
        self.target_col = None
        
        # Create ship rectangle
        self.shape = canvas.create_rectangle(
            x, y, x + self.width, y + self.height,
            fill='gray', tags=(tag, 'ship')
        )
        
        # Bind events
        canvas.tag_bind(tag, '<Button-1>', self.start_drag)
        canvas.tag_bind(tag, '<B1-Motion>', self.drag)
        canvas.tag_bind(tag, '<ButtonRelease-1>', self.stop_drag)
        canvas.tag_bind(tag, '<Button-3>', self.rotate)
        
        self.start_x = None
        self.start_y = None
        self.origin_x = x
        self.origin_y = y
        self.placed = False

    def start_drag(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
        # Remove ship from board state if it was placed
        if self.placed:
            coords = self.canvas.coords(self.shape)
            col = int((coords[0] - self.board.board_start_x) // self.board.cell_size)
            row = int((coords[1] - self.board.board_start_y) // self.board.cell_size)
            self.board.board_state.remove_ship(self.ship_type)
            self.placed = False
            self.board.save_button.config(state='disabled')

    def drag(self, event):
        if self.start_x is not None and self.start_y is not None:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            self.canvas.move(self.tag, dx, dy)
            self.start_x = event.x
            self.start_y = event.y

    def stop_drag(self, event):
        self.start_x = None
        self.start_y = None
        
        # Use the target position from highlighting
        if hasattr(self, 'target_row') and hasattr(self, 'target_col'):
            row = self.target_row
            col = self.target_col
        else:
            # Fallback to current position if no target (shouldn't happen)
            coords = self.canvas.coords(self.shape)
            board_x = coords[0] - self.board.board_start_x
            board_y = coords[1] - self.board.board_start_y
            col = int(board_x // self.board.cell_size)
            row = int(board_y // self.board.cell_size)
        
        # Check if the position is valid
        if (0 <= row < 6 and 0 <= col < 6 and 
            self.board.board_state.can_place_ship(row, col, self.length, self.is_vertical)):
            # Snap to grid
            new_x = self.board.board_start_x + col * self.board.cell_size
            new_y = self.board.board_start_y + row * self.board.cell_size
            self.canvas.coords(self.shape, 
                             new_x, new_y,
                             new_x + self.width,
                             new_y + self.height)
            
            # Update board state
            if self.placed:
                self.board.board_state.remove_ship(self.ship_type)
            self.board.board_state.place_ship(row, col, self.length, self.ship_type, self.is_vertical)
            self.placed = True
        else:
            # Invalid position - return to origin
            self.reset_position()
            if self.placed:
                self.board.board_state.remove_ship(self.ship_type)
                self.placed = False
        
        # Check if all ships are placed correctly
        if self.board.board_state.is_complete():
            self.board.save_button.config(state='normal')
        else:
            self.board.save_button.config(state='disabled')

    def rotate(self, event):
        # Always remove ship from board state before rotation
        if self.placed:
            self.board.board_state.remove_ship(self.ship_type)
            self.placed = False
            self.board.save_button.config(state='disabled')
        
        coords = self.canvas.coords(self.shape)
        center_x = (coords[0] + coords[2]) / 2
        center_y = (coords[1] + coords[3]) / 2
        
        # Store original orientation in case we need to revert
        original_vertical = self.is_vertical
        
        # Rotate
        self.is_vertical = not self.is_vertical
        if self.is_vertical:
            self.width, self.height = self.height, self.width
        else:
            self.width, self.height = self.height, self.width
            
        # Apply new coordinates
        self.canvas.coords(
            self.shape,
            center_x - self.width/2,
            center_y - self.height/2,
            center_x + self.width/2,
            center_y + self.height/2
        )
        
        # Only try to place the ship if it's over the board
        new_coords = self.canvas.coords(self.shape)
        col = int((new_coords[0] - self.board.board_start_x) // self.board.cell_size)
        row = int((new_coords[1] - self.board.board_start_y) // self.board.cell_size)
        
        if (0 <= row < 6 and 0 <= col < 6):
            # Update target position for potential placement
            self.target_row = row
            self.target_col = col
            
            if self.board.board_state.can_place_ship(row, col, self.length, self.is_vertical):
                self.board.board_state.place_ship(row, col, self.length, self.ship_type, self.is_vertical)
                self.placed = True
                
                # Check if all ships are placed correctly
                if self.board.board_state.is_complete():
                    self.board.save_button.config(state='normal')
            else:
                # Only revert rotation if we're actually over the board and placement is invalid
                self.is_vertical = original_vertical
                self.width, self.height = self.height, self.width
                self.canvas.coords(
                    self.shape,
                    coords[0], coords[1],
                    coords[2], coords[3]
                )

    def reset_position(self):
        current_coords = self.canvas.coords(self.shape)
        dx = self.origin_x - current_coords[0]
        dy = self.origin_y - current_coords[1]
        self.canvas.move(self.tag, dx, dy)
        if self.is_vertical:
            self.rotate(None)

class BattleshipBoard:
    def __init__(self, root):
        self.root = root
        self.root.title("Battleship Board Generator")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas for the board and ships
        self.canvas = tk.Canvas(self.main_frame, width=600, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        # Initialize board state
        self.board_state = BoardState()
        
        # Create board grid (6x6)
        self.cell_size = 40
        self.board_start_x = 50
        self.board_start_y = 50
        self.create_board()
        
        # Create ships
        self.ships = []
        ship_start_x = 400
        self.create_ships(ship_start_x)
        
        # Create Save button
        self.save_button = ttk.Button(
            self.main_frame, 
            text="Save and reset",
            command=self.save_and_reset,
            state='disabled'
        )
        self.save_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Bind canvas motion for highlighting
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.active_highlights = []

    def create_board(self):
        # Draw 6x6 grid
        for i in range(7):
            # Vertical lines
            self.canvas.create_line(
                self.board_start_x + i * self.cell_size,
                self.board_start_y,
                self.board_start_x + i * self.cell_size,
                self.board_start_y + 6 * self.cell_size
            )
            # Horizontal lines
            self.canvas.create_line(
                self.board_start_x,
                self.board_start_y + i * self.cell_size,
                self.board_start_x + 6 * self.cell_size,
                self.board_start_y + i * self.cell_size
            )

    def create_ships(self, start_x):
        # Create 3-segment ship
        self.ships.append(Ship(
            self.canvas, start_x, 50, 3, 'ship3', self
        ))
        
        # Create two 2-segment ships
        self.ships.append(Ship(
            self.canvas, start_x, 150, 2, 'ship2a', self
        ))
        self.ships.append(Ship(
            self.canvas, start_x, 250, 2, 'ship2b', self
        ))
        
    def on_mouse_move(self, event):
        # Clear previous highlights
        for highlight in self.active_highlights:
            self.canvas.delete(highlight)
        self.active_highlights.clear()
            
        # Find the currently dragged ship
        dragged_ship = None
        for ship in self.ships:
            if ship.start_x is not None:  # Ship is being dragged
                dragged_ship = ship
                break
                
        if dragged_ship:
            # Get ship's current position
            coords = self.canvas.coords(dragged_ship.shape)
            ship_center_x = (coords[0] + coords[2]) / 2 - self.board_start_x
            ship_center_y = (coords[1] + coords[3]) / 2 - self.board_start_y
            
            # Convert to grid coordinates based on ship center
            col = int(ship_center_x // self.cell_size)
            row = int(ship_center_y // self.cell_size)
            
            # Check if ship center is over the board
            if not (0 <= col < 6 and 0 <= row < 6):
                return
                
            # Check if placement would be valid
            valid = self.board_state.can_place_ship(
                row, col, dragged_ship.length, dragged_ship.is_vertical
            )
            
            # Store the target position for snapping
            dragged_ship.target_row = row
            dragged_ship.target_col = col
            
            # Highlight cells
            color = 'green' if valid else 'red'
            alpha = 0.3
            
            if dragged_ship.is_vertical:
                for r in range(row, min(6, row + dragged_ship.length)):
                    highlight = self.create_highlight(r, col, color, alpha)
                    self.active_highlights.append(highlight)
            else:
                for c in range(col, min(6, col + dragged_ship.length)):
                    highlight = self.create_highlight(row, c, color, alpha)
                    self.active_highlights.append(highlight)
                    
    def create_highlight(self, row, col, color, alpha):
        x1 = self.board_start_x + col * self.cell_size
        y1 = self.board_start_y + row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        return self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=color,
            stipple='gray50',
            tags='highlight'
        )

    def save_and_reset(self):
        # Get board state and save to CSV
        board_state = self.get_board_state()
        self.save_to_csv(board_state)
        
        # Reset ships to original positions
        for ship in self.ships:
            ship.reset_position()
        
        # Disable save button
        self.save_button.config(state='disabled')

    def get_board_state(self):
        return self.board_state.get_state_string()

    def save_to_csv(self, board_state):
        filename = "generated_boards_36_chars.csv"
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['board_state'])
            writer.writerow([board_state])
            
        # Reset board state
        self.board_state = BoardState()
        for ship in self.ships:
            ship.placed = False

def main():
    root = tk.Tk()
    app = BattleshipBoard(root)
    root.mainloop()

if __name__ == "__main__":
    main()