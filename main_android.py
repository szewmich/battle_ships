from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Color, Rectangle, Line
from kivy.core.window import Window
from kivy.clock import Clock
from functools import partial
import csv
import os
from board_state import BoardState

class Ship(Widget):
    def __init__(self, length, ship_type, **kwargs):
        super(Ship, self).__init__(**kwargs)
        self.length = length
        self.ship_type = ship_type
        self.is_vertical = False
        self.placed = False
        self.origin_pos = self.pos
        self.selected = False
        
        with self.canvas:
            Color(0.5, 0.5, 0.5, 1)  # Gray color
            self.rect = Rectangle(pos=self.pos, size=self.size)
            
        self.bind(pos=self._update_rect, size=self._update_rect)
        
    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
        
    def rotate(self):
        if self.is_vertical:
            self.size = (self.height, self.width)
        else:
            self.size = (self.width, self.height)
        self.is_vertical = not self.is_vertical
        
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Right click (two-finger touch on mobile) to rotate
            if hasattr(touch, 'button') and touch.button == 'right':
                self.rotate()
            else:
                self.selected = True
                touch.grab(self)
            return True
        return super(Ship, self).on_touch_down(touch)
        
    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.center = touch.pos
            return True
        return super(Ship, self).on_touch_move(touch)
        
    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self.selected = False
            return True
        return super(Ship, self).on_touch_up(touch)

class BattleshipBoard(RelativeLayout):
    def __init__(self, **kwargs):
        super(BattleshipBoard, self).__init__(**kwargs)
        self.board_state = BoardState()
        self.cell_size = min(Window.width / 8, Window.height / 8)
        self.board_offset_x = self.cell_size
        self.board_offset_y = self.cell_size
        
        # Create ships
        self.ships = []
        self.create_ships()
        
        # Create save button
        self.save_button = Button(
            text='Save and reset',
            size_hint=(None, None),
            size=(self.cell_size * 2, self.cell_size / 2),
            pos=(self.board_offset_x, self.board_offset_y - self.cell_size),
            disabled=True
        )
        self.save_button.bind(on_release=self.save_and_reset)
        self.add_widget(self.save_button)
        
        # Schedule the drawing
        Clock.schedule_once(self.draw_board, 0)
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        
    def create_ships(self):
        # Create 3-segment ship
        ship3 = Ship(
            length=3,
            ship_type=3,
            size=(self.cell_size * 3, self.cell_size),
            pos=(self.board_offset_x + self.cell_size * 7, self.board_offset_y + self.cell_size * 4)
        )
        self.ships.append(ship3)
        self.add_widget(ship3)
        
        # Create two 2-segment ships
        for i in range(2):
            ship2 = Ship(
                length=2,
                ship_type=2,
                size=(self.cell_size * 2, self.cell_size),
                pos=(self.board_offset_x + self.cell_size * 7, self.board_offset_y + self.cell_size * (1 + i * 2))
            )
            self.ships.append(ship2)
            self.add_widget(ship2)
            
    def draw_board(self, dt):
        with self.canvas:
            Color(0, 0, 0, 1)
            for i in range(7):
                # Vertical lines
                Line(points=[
                    self.board_offset_x + i * self.cell_size,
                    self.board_offset_y,
                    self.board_offset_x + i * self.cell_size,
                    self.board_offset_y + 6 * self.cell_size
                ])
                # Horizontal lines
                Line(points=[
                    self.board_offset_x,
                    self.board_offset_y + i * self.cell_size,
                    self.board_offset_x + 6 * self.cell_size,
                    self.board_offset_y + i * self.cell_size
                ])
                
    def update(self, dt):
        # Clear previous highlights
        self.canvas.before.clear()
        
        # Update highlights for selected ships
        for ship in self.ships:
            if ship.selected:
                # Calculate grid position
                col = int((ship.center_x - self.board_offset_x) // self.cell_size)
                row = int((ship.center_y - self.board_offset_y) // self.cell_size)
                
                # Check if position is valid
                valid = (0 <= row < 6 and 0 <= col < 6 and
                        self.board_state.can_place_ship(row, col, ship.length, ship.is_vertical))
                
                # Draw highlights
                with self.canvas.before:
                    Color(0, 1, 0, 0.3) if valid else Color(1, 0, 0, 0.3)
                    if ship.is_vertical:
                        for r in range(row, min(6, row + ship.length)):
                            Rectangle(
                                pos=(self.board_offset_x + col * self.cell_size,
                                     self.board_offset_y + r * self.cell_size),
                                size=(self.cell_size, self.cell_size)
                            )
                    else:
                        for c in range(col, min(6, col + ship.length)):
                            Rectangle(
                                pos=(self.board_offset_x + c * self.cell_size,
                                     self.board_offset_y + row * self.cell_size),
                                size=(self.cell_size, self.cell_size)
                            )
                            
                # Snap to grid on release
                if not ship.selected and valid:
                    ship.pos = (
                        self.board_offset_x + col * self.cell_size,
                        self.board_offset_y + row * self.cell_size
                    )
                    self.board_state.place_ship(row, col, ship.length, ship.ship_type, ship.is_vertical)
                    ship.placed = True
                    
        # Update save button state
        self.save_button.disabled = not self.board_state.is_complete()
        
    def save_and_reset(self, instance):
        # Save current state
        board_state = self.board_state.get_state_string()
        filename = "generated_boards_36_chars.csv"
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['board_state'])
            writer.writerow([board_state])
            
        # Reset board
        self.board_state = BoardState()
        for ship in self.ships:
            ship.pos = ship.origin_pos
            ship.placed = False
            if ship.is_vertical:
                ship.rotate()
        self.save_button.disabled = True

class BattleshipApp(App):
    def build(self):
        return BattleshipBoard()

if __name__ == '__main__':
    BattleshipApp().run()