import pyautogui
import time
import random
import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
import chess
from dataclasses import dataclass
from utilities import char_to_num


@dataclass
class GameState:
    """Container for game state variables that affect mouse movement"""
    critical_time_pressure: bool = False
    severe_time_pressure: bool = False
    in_time_pressure: bool = False
    just_blundered: bool = False
    we_are_winning_big: bool = False
    consecutive_good_moves: int = 0
    our_confidence_level: float = 1.0
    moves_in_time_pressure: int = 0
    game_phase: str = "opening"  # opening, middlegame, endgame
    session_duration: float = 0.0  # Time since session started
    total_moves_made: int = 0
    recent_move_times: list = None

    def __post_init__(self):
        if self.recent_move_times is None:
            self.recent_move_times = []


class MouseController:
    """Handles all mouse movements with enhanced human-like behaviors"""

    def __init__(self, grabber, mouse_latency: float = 0.05, enable_human_delays: bool = True):
        self.grabber = grabber
        self.mouse_latency = mouse_latency
        self.enable_human_delays = enable_human_delays
        self.is_white = None

        # Session tracking
        self.session_start_time = time.time()
        self.movement_history = []  # Store recent movements for muscle memory
        self.fatigue_factor = 0.0

        # Movement parameters
        self.base_speed = 0.3  # Base movement duration
        self.tremor_amplitude = 2.0  # Micro-tremor strength
        self.overshoot_probability = 0.15
        self.curve_factor = 0.3  # How much movements curve

    def set_player_color(self, is_white: bool):
        """Set the player's color for coordinate calculations"""
        self.is_white = is_white

    def update_fatigue(self, game_state: GameState):
        """Update fatigue factor based on session duration and move count"""
        session_duration = time.time() - self.session_start_time

        # Fatigue increases over time and with more moves
        time_fatigue = min(session_duration / 7200, 1.0)  # Max fatigue after 2 hours
        move_fatigue = min(game_state.total_moves_made / 200, 1.0)  # Max after 200 moves

        # Time pressure reduces fatigue effects (adrenaline)
        if game_state.critical_time_pressure:
            self.fatigue_factor = 0.1
        elif game_state.severe_time_pressure:
            self.fatigue_factor = max(0.2, (time_fatigue + move_fatigue) / 2 * 0.5)
        else:
            self.fatigue_factor = (time_fatigue + move_fatigue) / 2

    def generate_ballistic_curve(self, start: Tuple[float, float], end: Tuple[float, float],
                                 num_points: int = 20) -> list:
        """Generate a ballistic curve path between two points"""
        points = []

        # Calculate distance and angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Base curve parameters
        control_distance = distance * self.curve_factor * random.uniform(0.8, 1.2)

        # Random perpendicular offset for curve
        angle = math.atan2(dy, dx)
        perpendicular_angle = angle + math.pi / 2

        # Control point for quadratic Bezier curve
        curve_direction = random.choice([-1, 1])
        control_x = (start[0] + end[0]) / 2 + math.cos(perpendicular_angle) * control_distance * curve_direction
        control_y = (start[1] + end[1]) / 2 + math.sin(perpendicular_angle) * control_distance * curve_direction

        # Generate points along curve with ballistic speed profile
        for i in range(num_points):
            t = i / (num_points - 1)

            # Ballistic profile: fast start, slow end
            speed_t = 1 - math.pow(1 - t, 2.5)

            # Quadratic Bezier curve
            x = (1 - speed_t) ** 2 * start[0] + 2 * (1 - speed_t) * speed_t * control_x + speed_t ** 2 * end[0]
            y = (1 - speed_t) ** 2 * start[1] + 2 * (1 - speed_t) * speed_t * control_y + speed_t ** 2 * end[1]

            points.append((x, y))

        return points

    def add_micro_tremors(self, x: float, y: float, amplitude: float = None) -> Tuple[float, float]:
        """Add subtle micro-tremors to coordinates"""
        if amplitude is None:
            amplitude = self.tremor_amplitude * (1 + self.fatigue_factor)

        # Higher frequency tremors when nervous/fatigued
        tremor_x = random.gauss(0, amplitude * 0.3)
        tremor_y = random.gauss(0, amplitude * 0.3)

        return x + tremor_x, y + tremor_y

    def calculate_movement_duration(self, distance: float, game_state: GameState) -> float:
        """Calculate movement duration based on distance and game state"""
        if not self.enable_human_delays:
            return 0.1

        # Base duration scales with distance (Fitts's Law)
        base_duration = self.base_speed * (1 + math.log(1 + distance / 100))

        # Emotional state modifiers
        confidence_modifier = 1.5 - game_state.our_confidence_level * 0.5

        # Time pressure dramatically speeds up movements
        if game_state.critical_time_pressure:
            time_modifier = 0.2
        elif game_state.severe_time_pressure:
            time_modifier = 0.4
        elif game_state.in_time_pressure:
            time_modifier = 0.7
        else:
            time_modifier = 1.0

        # Fatigue slows movements
        fatigue_modifier = 1 + self.fatigue_factor * 0.3

        # Recent blunder makes movements more careful
        blunder_modifier = 1.5 if game_state.just_blundered else 1.0

        # Winning big = more casual/fast movements
        winning_modifier = 0.8 if game_state.we_are_winning_big else 1.0

        final_duration = base_duration * confidence_modifier * time_modifier * fatigue_modifier * blunder_modifier * winning_modifier

        # Add some randomness
        final_duration *= random.uniform(0.8, 1.2)

        return max(0.05, min(final_duration, 2.0))

    def simulate_overshoot(self, target: Tuple[float, float], distance: float,
                           game_state: GameState) -> Optional[Tuple[float, float]]:
        """Simulate overshooting the target (more likely when rushed or fatigued)"""
        overshoot_chance = self.overshoot_probability

        # Increase overshoot probability based on state
        if game_state.critical_time_pressure:
            overshoot_chance *= 3.0
        elif game_state.severe_time_pressure:
            overshoot_chance *= 2.0

        overshoot_chance *= (1 + self.fatigue_factor)
        overshoot_chance *= (1.5 - game_state.our_confidence_level)

        if random.random() < overshoot_chance:
            # Overshoot distance proportional to movement speed and distance
            overshoot_distance = random.uniform(5, 20) * (1 + distance / 500)

            # Random angle for overshoot
            angle = random.uniform(0, 2 * math.pi)
            overshoot_x = target[0] + math.cos(angle) * overshoot_distance
            overshoot_y = target[1] + math.sin(angle) * overshoot_distance

            return (overshoot_x, overshoot_y)

        return None

    def move_to_screen_pos(self, square: str) -> Tuple[float, float]:
        """Convert chess square notation to screen coordinates"""
        print(f"=== Calculating coordinates for square: {square} ===")

        canvas_x_offset, canvas_y_offset = self.grabber.get_top_left_corner()
        print(f"Canvas offset: x={canvas_x_offset}, y={canvas_y_offset}")

        board_location_x = self.grabber.get_board().location["x"]
        board_location_y = self.grabber.get_board().location["y"]
        print(f"Board location (relative): x={board_location_x}, y={board_location_y}")

        board_x = canvas_x_offset + board_location_x
        board_y = canvas_y_offset + board_location_y
        print(f"Board absolute position: x={board_x}, y={board_y}")

        board_width = self.grabber.get_board().size['width']
        board_height = self.grabber.get_board().size['height']
        square_size = board_width / 8
        print(f"Board size: {board_width}x{board_height}, Square size: {square_size}")

        file_num = char_to_num(square[0])
        rank_num = int(square[1])
        print(f"Target square: {square} = file {file_num}, rank {rank_num}")

        if self.is_white:
            x = board_x + square_size * (file_num - 1) + square_size / 2
            y = board_y + square_size * (8 - rank_num) + square_size / 2
            print(f"White perspective calculation:")
        else:
            x = board_x + square_size * (8 - file_num) + square_size / 2
            y = board_y + square_size * (rank_num - 1) + square_size / 2
            print(f"Black perspective calculation:")

        # Add slight randomness to avoid hitting exact center every time
        x += random.uniform(-square_size * 0.2, square_size * 0.2)
        y += random.uniform(-square_size * 0.2, square_size * 0.2)

        print(f"Final coordinates: x={x}, y={y}")
        print(f"=== End coordinate calculation ===")
        return x, y

    def get_move_positions(self, move: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get start and end positions for a chess move"""
        start_pos = self.move_to_screen_pos(move[0:2])
        end_pos = self.move_to_screen_pos(move[2:4])
        return start_pos, end_pos

    def simulate_mouse_hesitation(self, move: str, board: chess.Board, game_state: GameState):
        """Simulate human-like hesitation before making a move"""
        if not self.enable_human_delays:
            return

        start_pos, end_pos = self.get_move_positions(move)

        # Base hesitation probability
        hesitation_chance = 0.2

        # Modify based on game state
        if game_state.just_blundered:
            hesitation_chance *= 2.0  # More hesitation after blunder
        if game_state.critical_time_pressure:
            hesitation_chance *= 0.1  # Almost no hesitation in time scramble
        if game_state.consecutive_good_moves > 5:
            hesitation_chance *= 0.5  # Less hesitation when in flow

        # Hover over piece before grabbing
        if random.random() < hesitation_chance:
            hover_duration = random.uniform(0.05, 0.2) * (1 + self.fatigue_factor)

            # Generate hover position near piece
            hover_x = start_pos[0] + random.uniform(-10, 10)
            hover_y = start_pos[1] + random.uniform(-10, 10)

            # Smooth movement to hover position
            self._move_to_position(hover_x, hover_y, game_state, hover_duration)
            time.sleep(random.uniform(0.03, 0.1))

        # Rarely almost grab wrong piece (muscle memory error)
        wrong_piece_chance = 0.03 * (1 + self.fatigue_factor) * (2 - game_state.our_confidence_level)

        if random.random() < wrong_piece_chance and not game_state.critical_time_pressure:
            # Move toward adjacent piece
            wrong_x = start_pos[0] + random.choice([-50, 50])
            wrong_y = start_pos[1] + random.choice([-50, 50])

            # Start moving toward wrong piece
            self._move_to_position(wrong_x, wrong_y, game_state, 0.15)
            time.sleep(0.05)
            print("🤦 Almost grabbed wrong piece!")

    def _move_to_position(self, x: float, y: float, game_state: GameState, duration: float = None):
        """Internal method to move mouse to a position with human-like physics"""
        current_x, current_y = pyautogui.position()
        distance = math.sqrt((x - current_x) ** 2 + (y - current_y) ** 2)

        if duration is None:
            duration = self.calculate_movement_duration(distance, game_state)

        # Generate ballistic curve path
        num_points = max(5, int(distance / 20))
        path_points = self.generate_ballistic_curve((current_x, current_y), (x, y), num_points)

        # Execute movement along path
        start_time = time.time()

        for i, (px, py) in enumerate(path_points):
            # Add micro-tremors
            if i > 0 and i < len(path_points) - 1:  # Not at start or end
                px, py = self.add_micro_tremors(px, py)

            # Calculate timing for this segment
            progress = i / (len(path_points) - 1)
            target_time = start_time + duration * progress

            # Wait if needed to maintain timing
            current_time = time.time()
            if current_time < target_time:
                time.sleep(target_time - current_time)

            pyautogui.moveTo(px, py, _pause=False)

        # Ensure we end at exact position
        pyautogui.moveTo(x, y, _pause=False)

    def make_move(self, move: str, board: chess.Board, game_state: GameState):
        """Execute a chess move with human-like mouse movements"""
        print(f"Making move: {move}")

        # Update fatigue based on current game state
        self.update_fatigue(game_state)

        # Simulate pre-move hesitation
        self.simulate_mouse_hesitation(move, board, game_state)

        start_pos, end_pos = self.get_move_positions(move)
        print(f"Start position: {start_pos}, End position: {end_pos}")

        # Move to start position
        print(f"Moving to start position: {start_pos}")
        self._move_to_position(start_pos[0], start_pos[1], game_state)
        time.sleep(self.mouse_latency)

        # Click and hold
        print("Clicking and holding at start position")
        pyautogui.mouseDown()

        # Human-like hold duration varies with state
        if game_state.critical_time_pressure:
            hold_time = random.uniform(0.02, 0.05)
        elif self._is_obvious_move(move, board, game_state):
            hold_time = random.uniform(0.05, 0.1)
        else:
            hold_time = random.uniform(0.1, 0.2) * (1 + self.fatigue_factor * 0.5)

        time.sleep(hold_time)

        # Drag to end position
        print(f"Dragging to end position: {end_pos}")

        # Calculate drag duration based on move type and state
        drag_duration = self._calculate_drag_duration(move, board, game_state, start_pos, end_pos)

        # Check for overshoot
        distance = math.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
        overshoot_pos = self.simulate_overshoot(end_pos, distance, game_state)

        if overshoot_pos and not game_state.critical_time_pressure:
            # Move to overshoot position first
            self._move_to_position(overshoot_pos[0], overshoot_pos[1], game_state, drag_duration * 0.8)
            time.sleep(0.03)
            # Then correct to actual position
            self._move_to_position(end_pos[0], end_pos[1], game_state, drag_duration * 0.2)
        else:
            # Direct movement
            self._move_to_position(end_pos[0], end_pos[1], game_state, drag_duration)

        time.sleep(0.05)

        # Release mouse
        print("Releasing mouse at end position")
        pyautogui.mouseUp()

        # Post-move delay
        post_delay = self._calculate_post_move_delay(move, board, game_state)
        time.sleep(post_delay)

        # Handle promotion
        if len(move) == 5:
            self._handle_promotion(move, game_state)

        # Store movement in history for muscle memory
        self.movement_history.append({
            'move': move,
            'duration': drag_duration,
            'distance': distance,
            'timestamp': time.time()
        })

        # Keep only recent history
        if len(self.movement_history) > 20:
            self.movement_history.pop(0)

        print("Move completed")

    def _calculate_drag_duration(self, move: str, board: chess.Board, game_state: GameState,
                                 start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> float:
        """Calculate appropriate drag duration based on move type and game state"""
        if not self.enable_human_delays:
            return 0.1

        distance = math.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
        base_duration = self.calculate_movement_duration(distance, game_state)

        # Special case durations
        if game_state.critical_time_pressure:
            return random.uniform(0.05, 0.1)
        elif game_state.severe_time_pressure:
            return random.uniform(0.1, 0.2)
        elif self._is_obvious_recapture(move, board):
            return random.uniform(0.1, 0.15)
        elif game_state.just_blundered:
            return base_duration * random.uniform(1.5, 2.0)  # Extra careful

        # Check for similar moves in history (muscle memory)
        similar_move_duration = self._get_muscle_memory_duration(move, distance)
        if similar_move_duration:
            # Use similar duration with small variation
            return similar_move_duration * random.uniform(0.9, 1.1)

        return base_duration

    def _calculate_post_move_delay(self, move: str, board: chess.Board, game_state: GameState) -> float:
        """Calculate delay after releasing the piece"""
        if not self.enable_human_delays:
            return 0.15

        if game_state.critical_time_pressure:
            return 0.05
        elif self._is_checkmate(move, board):
            return random.uniform(0.3, 0.6)  # Pause for effect
        else:
            base_delay = random.uniform(0.1, 0.3)
            # Fatigue increases post-move delay
            return base_delay * (1 + self.fatigue_factor * 0.3)

    def _handle_promotion(self, move: str, game_state: GameState):
        """Handle pawn promotion with appropriate mouse movements"""
        print(f"Promotion detected: {move[4]}")
        time.sleep(0.3 * (1 + self.fatigue_factor))

        promotion_square = move[2:4]
        base_x, base_y = self.move_to_screen_pos(promotion_square)

        # Calculate promotion piece position
        if move[4] == "n":
            offset = -1
        elif move[4] == "r":
            offset = -2
        elif move[4] == "b":
            offset = -3
        else:  # Queen
            offset = 0

        # Adjust y position for promotion menu
        promo_y = base_y + offset * 50  # Approximate spacing

        print(f"Clicking promotion piece at: {base_x}, {promo_y}")
        self._move_to_position(base_x, promo_y, game_state)
        pyautogui.click(button='left')

    def _get_muscle_memory_duration(self, move: str, distance: float) -> Optional[float]:
        """Check if we've made similar moves recently (muscle memory effect)"""
        if len(self.movement_history) < 3:
            return None

        # Look for similar moves in recent history
        for hist in self.movement_history[-10:]:
            # Similar starting square or similar distance
            if (hist['move'][:2] == move[:2] or
                    abs(hist['distance'] - distance) < 50):
                return hist['duration']

        return None

    def _is_obvious_move(self, move: str, board: chess.Board, game_state: GameState) -> bool:
        """Check if this is an obvious/instant move"""
        try:
            chess_move = chess.Move.from_uci(move)

            # Castling
            if board.is_castling(chess_move):
                return True

            # Forced moves
            if len(list(board.legal_moves)) == 1:
                return True

            # Check escapes with few options
            if board.is_check() and len(list(board.legal_moves)) <= 3:
                return True

            return False
        except:
            return False

    def _is_obvious_recapture(self, move: str, board: chess.Board) -> bool:
        """Check if this is an obvious recapture"""
        try:
            chess_move = chess.Move.from_uci(move)
            # Simple check - is this a capture on a square that was just moved to?
            if board.is_capture(chess_move) and len(board.move_stack) > 0:
                last_move = board.peek()
                return last_move.to_square == chess_move.to_square
            return False
        except:
            return False

    def _is_checkmate(self, move: str, board: chess.Board) -> bool:
        """Check if this move delivers checkmate"""
        try:
            chess_move = chess.Move.from_uci(move)
            temp_board = board.copy()
            temp_board.push(chess_move)
            return temp_board.is_checkmate()
        except:
            return False
