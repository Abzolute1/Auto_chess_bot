import multiprocessing
from stockfish import Stockfish
import pyautogui
import time
import sys
import os
import chess
import re
import random
from grabbers.chesscom_grabber import ChesscomGrabber
from grabbers.lichess_grabber import LichessGrabber
from utilities import char_to_num
import keyboard
import math


# REMOVED: from nnue_adapter import NNUEAdapter - moved to lazy import


class StockfishBot(multiprocessing.Process):
    def __init__(self, chrome_url, chrome_session_id, website, pipe, overlay_queue, stockfish_path, enable_manual_mode,
                 enable_mouseless_mode, enable_non_stop_puzzles, enable_non_stop_matches, mouse_latency, bongcloud,
                 slow_mover, skill_level, stockfish_depth, memory, cpu_threads, enable_human_delays=True,
                 min_thinking_time=0.5, max_thinking_time=12.0, human_accuracy=True, accuracy_threshold=75,
                 engine_type="stockfish", enable_blunders=True, blunders_per_game=2, blunder_severity=150):
        multiprocessing.Process.__init__(self)

        self.chrome_url = chrome_url
        self.chrome_session_id = chrome_session_id
        self.website = website
        self.pipe = pipe
        self.overlay_queue = overlay_queue
        self.stockfish_path = stockfish_path
        self.enable_manual_mode = enable_manual_mode
        self.enable_mouseless_mode = enable_mouseless_mode
        self.enable_non_stop_puzzles = enable_non_stop_puzzles
        self.enable_non_stop_matches = enable_non_stop_matches
        self.mouse_latency = mouse_latency
        self.bongcloud = bongcloud
        self.slow_mover = slow_mover
        self.skill_level = skill_level
        self.stockfish_depth = stockfish_depth
        self.grabber = None
        self.memory = memory
        self.cpu_threads = cpu_threads
        self.is_white = None
        self.engine_type = engine_type  # Add engine type parameter

        # Human-like thinking parameters - ENHANCED FOR FASTER PLAY
        self.enable_human_delays = enable_human_delays
        self.min_thinking_time = 0.1  # Reduced from 0.5 for faster play
        self.max_thinking_time = 30.0  # Keep 30 seconds for complex positions
        self.bullet_max_time = 5.0  # Reduced from 8.0
        self.blitz_max_time = 10.0  # Reduced from 15.0
        self.rapid_max_time = 20.0  # Reduced from 30.0

        # Human accuracy parameters - FIXED THRESHOLDS
        self.human_accuracy = human_accuracy  # Enable human-like move selection (not always best move)
        self.accuracy_threshold = accuracy_threshold  # Centipawns within which to randomize (75 = more mistakes)

        # NEW: Complete Error System (Inaccuracies, Mistakes, Blunders)
        self.enable_errors = enable_blunders  # Renamed to cover all errors
        self.errors_per_game = {
            'inaccuracy': 4,  # 25-75cp loss
            'mistake': 2,  # 75-150cp loss
            'blunder': blunders_per_game  # 150+ cp loss
        }
        self.error_severity = {
            'inaccuracy': (25, 75),
            'mistake': (75, 150),
            'blunder': (blunder_severity, blunder_severity * 2)
        }

        # Error tracking
        self.errors_this_game = {
            'inaccuracy': 0,
            'mistake': 0,
            'blunder': 0
        }
        self.last_error_move = 0
        self.error_cooldown = 5  # Minimum moves between any errors
        self.game_error_log = []
        self.force_error_this_move = None  # Type of error to force

        # Track game state for advanced move detection
        self.last_opponent_move = None
        self.last_our_move = None
        self.just_made_fork = False
        self.previous_evaluation = 0  # Track evaluation swings

        # NEW: Enhanced human-like state tracking
        self.consecutive_good_moves = 0  # Track if we're "in the zone"
        self.just_blundered = False  # Are we tilted?
        self.opponent_time_pressure = False  # Is opponent low on time?
        self.we_are_winning_big = False  # Are we crushing?
        self.surprised_by_move = False  # Did opponent play something unexpected?
        self.in_tactical_sequence = False  # Are we calculating a forcing line?
        self.last_move_was_brilliant = False  # Did we just play a great move?
        self.game_phase_changed = False  # Did we just transition to endgame?
        self.opponent_rating_estimate = None  # Estimate opponent strength
        self.our_confidence_level = 1.0  # Confidence multiplier
        self.recent_blunder_moves = []  # Track our recent mistakes
        self.opening_book_ended = False  # Are we out of book?
        self.complex_position_streak = 0  # How many complex positions in a row?

        # NEW: Time management tracking
        self.our_time_remaining = None  # Track our clock
        self.opponent_time_remaining = None  # Track opponent's clock
        self.initial_time = None  # Starting time for the game
        self.time_control_type = None  # bullet/blitz/rapid
        self.in_time_pressure = False  # Are we low on time?
        self.severe_time_pressure = False  # Are we critically low?
        self.critical_time_pressure = False  # Are we about to flag?
        self.last_thinking_times = []  # Track recent thinking times for patterns
        self.moves_in_time_pressure = 0  # Track how many moves we've made in time pressure

        # FASTER PLAY: Track opening book moves for instant play
        self.common_openings = {
            # Italian Game
            'e2e4,e7e5,g1f3,b8c6,f1c4': ['b7b5', 'f8c5', 'g8f6'],
            # Spanish/Ruy Lopez
            'e2e4,e7e5,g1f3,b8c6,f1b5': ['a7a6', 'g8f6', 'f7f5'],
            # Sicilian Defense
            'e2e4,c7c5': ['g1f3', 'd2d4', 'b1c3'],
            # French Defense
            'e2e4,e7e6': ['d2d4', 'g1f3'],
            # Caro-Kann
            'e2e4,c7c6': ['d2d4', 'g1f3', 'b1c3'],
            # Queen's Gambit
            'd2d4,d7d5,c2c4': ['e7e6', 'c7c6', 'd5c4'],
            # King's Indian
            'd2d4,g8f6,c2c4,g7g6': ['b1c3', 'g1f3'],
        }

    def get_time_remaining(self):
        """Get time remaining from the grabber"""
        try:
            # This method would need to be implemented in your grabber classes
            # For now, return a default value if not available
            if hasattr(self.grabber, 'get_player_time'):
                return self.grabber.get_player_time(self.is_white)
            return None
        except:
            return None

    def get_opponent_time_remaining(self):
        """Get opponent's time remaining"""
        try:
            if hasattr(self.grabber, 'get_player_time'):
                return self.grabber.get_player_time(not self.is_white)
            return None
        except:
            return None

    def detect_time_control(self):
        """Detect game time control based on initial time"""
        try:
            if self.initial_time is None:
                self.initial_time = self.get_time_remaining()

            if self.initial_time:
                # Convert to seconds if needed
                if self.initial_time < 180:  # Less than 3 minutes
                    self.time_control_type = "bullet"
                elif self.initial_time < 600:  # Less than 10 minutes
                    self.time_control_type = "blitz"
                else:
                    self.time_control_type = "rapid"
            else:
                # Default to blitz if we can't detect
                self.time_control_type = "blitz"

        except:
            self.time_control_type = "blitz"

    def update_time_pressure_state(self):
        """Update our time pressure state - ENHANCED for human-like panic"""
        try:
            time_remaining = self.get_time_remaining()
            if time_remaining is None:
                return

            # Different thresholds for different time controls
            if self.time_control_type == "bullet":
                self.critical_time_pressure = time_remaining < 5  # Less than 5 seconds - PANIC!
                self.severe_time_pressure = time_remaining < 10  # Less than 10 seconds
                self.in_time_pressure = time_remaining < 30  # Less than 30 seconds
            elif self.time_control_type == "blitz":
                self.critical_time_pressure = time_remaining < 10  # Less than 10 seconds - PANIC!
                self.severe_time_pressure = time_remaining < 20  # Less than 20 seconds
                self.in_time_pressure = time_remaining < 60  # Less than 1 minute
            else:  # rapid
                self.critical_time_pressure = time_remaining < 15  # Less than 15 seconds - PANIC!
                self.severe_time_pressure = time_remaining < 30  # Less than 30 seconds
                self.in_time_pressure = time_remaining < 120  # Less than 2 minutes

            # Track consecutive moves in time pressure
            if self.in_time_pressure:
                self.moves_in_time_pressure += 1
            else:
                self.moves_in_time_pressure = 0

            # Update opponent time pressure
            opp_time = self.get_opponent_time_remaining()
            if opp_time and self.time_control_type == "bullet":
                self.opponent_time_pressure = opp_time < 20
            elif opp_time and self.time_control_type == "blitz":
                self.opponent_time_pressure = opp_time < 40

        except:
            pass

    def should_force_error_now(self, board, move_count, stockfish):
        """
        ENHANCED: Determine if we should force any type of error (inaccuracy/mistake/blunder)
        """
        if not self.enable_errors:
            return None

        # Don't make errors in critical time pressure
        if self.critical_time_pressure:
            return None

        # Too soon after last error
        if move_count - self.last_error_move < self.error_cooldown:
            return None

        # Don't error in very early opening
        if move_count < 5:
            return None

        # Check each error type
        for error_type in ['blunder', 'mistake', 'inaccuracy']:
            if self.errors_this_game[error_type] >= self.errors_per_game[error_type]:
                continue

            # Calculate if we're behind schedule for this error type
            expected_game_length = 45  # Average game length
            expected_errors = (move_count / expected_game_length) * self.errors_per_game[error_type]
            error_deficit = expected_errors - self.errors_this_game[error_type]

            if error_deficit > 0.3:  # Behind schedule
                # Base probability based on error type and deficit
                if error_type == 'inaccuracy':
                    base_prob = min(0.25, error_deficit * 0.15)
                elif error_type == 'mistake':
                    base_prob = min(0.20, error_deficit * 0.12)
                else:  # blunder
                    base_prob = min(0.15, error_deficit * 0.10)

                # Modifiers
                error_chance = base_prob

                # More errors when winning (overconfidence)
                if self.we_are_winning_big:
                    error_chance *= 1.5

                # More errors after many good moves (complacency)
                if self.consecutive_good_moves > 5:
                    error_chance *= 1.3

                # More errors in time pressure (but not critical)
                if self.in_time_pressure and not self.severe_time_pressure:
                    error_chance *= 1.4

                # Less errors when losing badly
                try:
                    eval_data = stockfish.get_evaluation()
                    if eval_data and eval_data['type'] == 'cp':
                        current_eval = eval_data['value']
                        if not board.turn:
                            current_eval = -current_eval
                        if current_eval < -300:
                            error_chance *= 0.5
                except:
                    pass

                # Roll the dice
                if random.random() < error_chance:
                    print(f"🎲 Forcing {error_type}: {error_chance:.2%} chance hit!")
                    return error_type

        return None

    def calculate_thinking_time(self, board, move, move_count, stockfish=None):
        """
        ULTRA-ENHANCED human-like thinking with FASTER average play
        """
        if not self.enable_human_delays:
            return 0.1

        # Update time pressure state
        self.update_time_pressure_state()

        # CRITICAL TIME PRESSURE - Ultra fast moves to avoid flagging
        if self.critical_time_pressure:
            # Exponentially faster as time runs out
            time_left = self.get_time_remaining() or 5
            if time_left < 2:
                panic_time = 0.05  # Near instant
            elif time_left < 5:
                panic_time = random.uniform(0.05, 0.1)
            else:
                panic_time = random.uniform(0.1, 0.2)
            print(f"🚨🚨 CRITICAL TIME: {panic_time:.1f}s (only {time_left:.1f}s left!)")
            return panic_time

        # SEVERE TIME PRESSURE
        if self.severe_time_pressure:
            # Still panic but with slight variation
            panic_time = random.uniform(0.2, 0.5)
            # Even faster if we've been in time pressure for many moves
            if self.moves_in_time_pressure > 5:
                panic_time *= 0.5
            print(f"🚨 SEVERE TIME PRESSURE: {panic_time:.1f}s")
            return panic_time

        # ERROR THINKING PATTERNS - Keep as is
        if self.force_error_this_move:
            # Different thinking patterns for different errors
            if self.force_error_this_move == 'inaccuracy':
                # Quick oversight
                error_time = random.uniform(0.8, 2.5)
                print(f"😕 Inaccuracy thinking: {error_time:.1f}s")
            elif self.force_error_this_move == 'mistake':
                # Miscalculation after some thought
                error_time = random.uniform(2.0, 5.0)
                print(f"😣 Mistake thinking: {error_time:.1f}s")
            else:  # blunder
                # Often happens after "deep" calculation
                if random.random() < 0.3:
                    error_time = random.uniform(1.5, 3.0)  # Quick blunder
                else:
                    error_time = random.uniform(4.0, 10.0)  # Calculated blunder
                print(f"🤯 Blunder thinking: {error_time:.1f}s")
            return error_time

        # ============ INSTANT REACTIONS (0.05-0.15s) ============

        # Pre-moves and instant recaptures - TRULY INSTANT
        if move and self.last_opponent_move:
            if self.is_obvious_recapture(board, move):
                thinking_time = random.uniform(0.05, 0.15)
                print(f"⚡⚡ INSTANT recapture: {thinking_time:.1f}s")
                return thinking_time

        # ============ VERY FAST MOVES (0.1-0.3s) ============

        # Opening moves - MUCH FASTER throughout opening
        if self.is_opening_phase(move_count):
            # First 6 moves almost instant
            if move_count < 6:
                thinking_time = random.uniform(0.05, 0.15)
                print(f"⚡⚡ Early opening blitz: {thinking_time:.1f}s")
                return thinking_time
            # Moves 6-15 still very fast
            elif move_count < 15:
                thinking_time = random.uniform(0.1, 0.3)
                print(f"⚡ Opening theory: {thinking_time:.1f}s")
                return thinking_time
            # Late opening still quick
            else:
                thinking_time = random.uniform(0.2, 0.5)
                print(f"📖 Late opening: {thinking_time:.1f}s")
                return thinking_time

        # Natural development moves - INSTANT
        if self.is_piece_development_move(board, move, move_count):
            thinking_time = random.uniform(0.05, 0.2)
            print(f"🏃‍♂️ Natural development: {thinking_time:.1f}s")
            return thinking_time

        # Castling - VERY FAST
        if move and self.is_obvious_castling(board, move):
            thinking_time = random.uniform(0.05, 0.2)
            print(f"🏰 Instant castling: {thinking_time:.1f}s")
            return thinking_time

        # Only legal move - INSTANT
        if self.is_forced_move(board):
            thinking_time = random.uniform(0.05, 0.1)
            print(f"🎯 Only move: {thinking_time:.1f}s")
            return thinking_time

        # Check escapes - VERY FAST
        if self.is_check_escape_only(board):
            thinking_time = random.uniform(0.1, 0.3)
            print(f"👑 Check escape: {thinking_time:.1f}s")
            return thinking_time

        # Free piece captures - INSTANT
        if move and self.is_free_piece_capture(board, move):
            thinking_time = random.uniform(0.05, 0.2)
            print(f"🎁 Free piece grab: {thinking_time:.1f}s")
            return thinking_time

        # Obvious promotions - FAST
        if move and self.is_promotion_to_queen(move):
            thinking_time = random.uniform(0.1, 0.3)
            print(f"👸 Queen promotion: {thinking_time:.1f}s")
            return thinking_time

        # Simple captures of hanging pieces - FAST
        if move and self.is_piece_hanging_obviously(board, move):
            thinking_time = random.uniform(0.1, 0.3)
            print(f"🎯 Hanging piece: {thinking_time:.1f}s")
            return thinking_time

        # Fork followup captures - INSTANT
        if move and self.is_fork_followup(board, move):
            thinking_time = random.uniform(0.05, 0.15)
            print(f"🍴 Fork collection: {thinking_time:.1f}s")
            return thinking_time

        # Mate in 1 - SLIGHT PAUSE FOR DRAMA
        if move and self.is_mate_in_one(board, move):
            thinking_time = random.uniform(0.3, 0.8)
            print(f"♟️ Checkmate!: {thinking_time:.1f}s")
            return thinking_time

        # ============ COMPLEX POSITION ANALYSIS - FASTER BASE TIMES ============

        # Get max thinking time based on time control and remaining time
        effective_max_time = self.get_effective_max_thinking_time()

        if stockfish:
            try:
                candidate_count = self.count_candidate_moves(board, stockfish)
                position_complexity = self.analyze_position_complexity(board, move_count)
                eval_spread = self.get_evaluation_spread(stockfish)

                # FASTER BASE RANGES for normal moves
                if candidate_count <= 1:
                    base_range = (0.2, 0.5)  # Reduced from (0.4, 1.2)
                    complexity_label = "Only move"
                elif candidate_count <= 2:
                    base_range = (0.4, 1.5)  # Reduced from (0.8, 3.0)
                    complexity_label = f"Few options ({candidate_count})"
                elif candidate_count <= 4:
                    base_range = (0.8, 5.0)  # Reduced from (2.0, 10.0)
                    complexity_label = f"Several options ({candidate_count})"
                else:
                    # Many candidates - still allow deep thinks but less common
                    if eval_spread < 30:  # Very close evaluations
                        base_range = (3.0, effective_max_time)  # Reduced from (8.0, max)
                        complexity_label = f"VERY complex! ({candidate_count} similar moves)"
                    elif eval_spread < 60:
                        base_range = (2.0, effective_max_time * 0.5)  # Reduced
                        complexity_label = f"Complex ({candidate_count} options)"
                    else:
                        base_range = (1.0, 6.0)  # Reduced from (3.0, 12.0)
                        complexity_label = f"Many options ({candidate_count})"

                # Sample thinking time with realistic distribution
                thinking_time = self.sample_thinking_time_complex(base_range, eval_spread)

                # Apply modifiers
                thinking_time *= position_complexity
                thinking_time = self.add_psychological_delay(thinking_time)
                thinking_time = self.add_thinking_variance(thinking_time)

                print(f"🧠 {complexity_label}: {thinking_time:.1f}s")

            except Exception as e:
                print(f"Error in analysis: {e}")
                thinking_time = random.uniform(0.5, 2.0)  # Reduced fallback
        else:
            thinking_time = random.uniform(0.5, 2.0)  # Reduced fallback

        # Time pressure override - MORE AGGRESSIVE
        if self.in_time_pressure:
            max_allowed = 1.0 if not self.severe_time_pressure else 0.5
            thinking_time = min(thinking_time, max_allowed)
            print(f"⏰ Time pressure cap: {max_allowed}s")

        # Final bounds
        thinking_time = max(0.05, min(thinking_time, effective_max_time))

        # Track patterns
        self.last_thinking_times.append(thinking_time)
        if len(self.last_thinking_times) > 10:
            self.last_thinking_times.pop(0)

        return round(thinking_time, 1)

    def sample_thinking_time_complex(self, time_range, eval_spread):
        """Enhanced sampling for complex positions - FASTER BIAS"""
        min_time, max_time = time_range

        # For very complex positions, still allow deep thinks but less often
        if eval_spread < 30 and max_time > 15:
            # Only 20% chance of deep think (reduced from 40%)
            if random.random() < 0.2:
                return random.uniform(max_time * 0.6, max_time)

        # Normal distribution - bias toward faster times
        if random.random() < 0.7:
            # Beta distribution biased toward lower values
            beta_sample = random.betavariate(2, 5)
            return min_time + beta_sample * (max_time - min_time) * 0.3  # Reduced from 0.5
        else:
            # Uniform distribution weighted toward lower half
            if random.random() < 0.7:
                return random.uniform(min_time, min_time + (max_time - min_time) * 0.5)
            else:
                return random.uniform(min_time, max_time)

    def get_effective_max_thinking_time(self):
        """Get maximum thinking time based on time control and remaining time"""
        # Base max times by time control
        if self.time_control_type == "bullet":
            base_max = self.bullet_max_time
        elif self.time_control_type == "blitz":
            base_max = self.blitz_max_time
        else:
            base_max = self.rapid_max_time

        # Adjust based on remaining time
        time_remaining = self.get_time_remaining()
        if time_remaining:
            # Scale down as time decreases
            if time_remaining < 30:
                time_factor = 0.05  # Very conservative
            elif time_remaining < 60:
                time_factor = 0.1
            elif time_remaining < 120:
                time_factor = 0.15
            else:
                time_factor = 0.2  # Never use more than 20% of time

            time_based_max = time_remaining * time_factor

            # But ensure minimum thinking time unless critical
            if not self.severe_time_pressure:
                time_based_max = max(0.5, time_based_max)  # Reduced from 1.0

            return min(base_max, time_based_max)

        return base_max

    def select_human_like_move(self, stockfish, board, move_count):
        """
        Enhanced human-like move selection with complete error system
        """
        if not self.human_accuracy:
            return stockfish.get_best_move()

        try:
            # Update psychological state
            self.update_game_psychology(board, stockfish)

            # Check if we should force an error this move
            error_type = self.should_force_error_now(board, move_count, stockfish)
            if error_type:
                self.force_error_this_move = error_type
                error_move = self.select_error_move(stockfish, board, move_count, error_type)
                self.update_error_state(move_count, error_type)
                self.force_error_this_move = None
                return error_move

            # Get top moves
            top_moves = stockfish.get_top_moves(10)
            if not top_moves or len(top_moves) == 0:
                return stockfish.get_best_move()

            if len(top_moves) == 1:
                return top_moves[0]['Move']

            best_move = top_moves[0]
            best_eval = best_move.get('Centipawn', 0)
            best_move_uci = best_move['Move']

            # Critical positions - always play best
            if 'Mate' in best_move and best_move.get('Mate', 0) <= 2:
                print("🚨 Mate in 1-2 - playing best move")
                return best_move_uci

            # Time pressure override - play faster, not necessarily best
            if self.severe_time_pressure:
                # Just play a reasonable move quickly
                max_candidates = 3
                quick_candidates = top_moves[:max_candidates]
                weights = [0.6, 0.3, 0.1][:len(quick_candidates)]
                selected = random.choices(quick_candidates, weights=weights)[0]['Move']
                print(f"⏰ Time pressure move: {selected}")
                return selected

            # Build candidate list
            candidate_moves = []

            # Dynamic threshold based on rating/skill
            # Higher skill = tighter threshold
            if self.skill_level <= 5:
                base_threshold = 100
            elif self.skill_level <= 10:
                base_threshold = 75
            elif self.skill_level <= 15:
                base_threshold = 50
            else:
                base_threshold = 30

            # Psychology adjustments
            if self.just_blundered:
                base_threshold *= 0.7  # More careful
            elif self.we_are_winning_big:
                base_threshold *= 1.5  # More sloppy
            elif self.in_time_pressure:
                base_threshold *= 1.3  # More errors under pressure

            # Build candidates
            for i, move in enumerate(top_moves):
                move_eval = move.get('Centipawn', 0)
                eval_diff = abs(best_eval - move_eval)

                if i == 0:  # Always include best
                    candidate_moves.append(move)
                elif eval_diff <= base_threshold:
                    candidate_moves.append(move)
                    print(f"🎲 Candidate {len(candidate_moves)}: {move['Move']} (-{eval_diff}cp)")
                elif eval_diff > base_threshold * 3:
                    break

            # Selection with realistic weights
            if len(candidate_moves) == 1:
                selected_move = candidate_moves[0]['Move']
            else:
                # Weight distribution for more realistic play
                # Not always picking the best move
                if len(candidate_moves) == 2:
                    weights = [0.70, 0.30]  # 70% best, 30% second
                elif len(candidate_moves) == 3:
                    weights = [0.55, 0.30, 0.15]
                elif len(candidate_moves) == 4:
                    weights = [0.45, 0.25, 0.20, 0.10]
                else:
                    # Distribute weights with bias toward better moves
                    weights = []
                    remaining = 1.0
                    for i in range(len(candidate_moves)):
                        if i == 0:
                            w = 0.40
                        elif i < 3:
                            w = 0.20
                        else:
                            w = remaining / (len(candidate_moves) - i)
                        weights.append(w)
                        remaining -= w

                selected_move = random.choices(candidate_moves, weights=weights)[0]['Move']

            return selected_move

        except Exception as e:
            print(f"Error in move selection: {e}")
            return stockfish.get_best_move()

    def select_error_move(self, stockfish, board, move_count, error_type):
        """
        Select a move that constitutes the specified error type
        """
        try:
            print(f"🎯 Selecting {error_type.upper()} move...")

            top_moves = stockfish.get_top_moves(12)  # Get more moves for variety
            if not top_moves or len(top_moves) < 2:
                return top_moves[0]['Move'] if top_moves else stockfish.get_best_move()

            best_move = top_moves[0]
            best_eval = best_move.get('Centipawn', 0)

            # Get error severity range
            min_loss, max_loss = self.error_severity[error_type]

            # Find candidate moves in the error range
            error_candidates = []

            for i, move in enumerate(top_moves[1:], 1):
                move_eval = move.get('Centipawn', 0)
                eval_loss = best_eval - move_eval

                if min_loss <= eval_loss <= max_loss:
                    # Don't pick mate blunders unless it's a distant mate
                    if 'Mate' not in move or abs(move.get('Mate', 0)) > 10:
                        error_candidates.append({
                            'move': move['Move'],
                            'eval_loss': eval_loss,
                            'rank': i
                        })

            # If no candidates in range, adjust the range slightly
            if not error_candidates and len(top_moves) > 3:
                adjusted_min = min_loss * 0.7
                adjusted_max = max_loss * 1.3

                for i, move in enumerate(top_moves[1:], 1):
                    move_eval = move.get('Centipawn', 0)
                    eval_loss = best_eval - move_eval

                    if adjusted_min <= eval_loss <= adjusted_max:
                        error_candidates.append({
                            'move': move['Move'],
                            'eval_loss': eval_loss,
                            'rank': i
                        })

            # Select from candidates
            if error_candidates:
                # Prefer errors closer to the minimum loss (more realistic)
                error_candidates.sort(key=lambda x: abs(x['eval_loss'] - min_loss))

                # Weight selection
                if len(error_candidates) == 1:
                    selected = error_candidates[0]
                else:
                    # Exponential weights favoring smaller errors
                    weights = [0.5 ** i for i in range(len(error_candidates))]
                    total = sum(weights)
                    weights = [w / total for w in weights]
                    selected = random.choices(error_candidates, weights=weights)[0]

                print(f"💥 {error_type.upper()}: {selected['move']} (loses {selected['eval_loss']}cp)")

                # Log the error
                self.game_error_log.append({
                    'move_number': move_count,
                    'move': selected['move'],
                    'eval_loss': selected['eval_loss'],
                    'type': error_type
                })

                return selected['move']

            # Fallback - just play a suboptimal move
            fallback_rank = {
                'inaccuracy': min(2, len(top_moves) - 1),
                'mistake': min(3, len(top_moves) - 1),
                'blunder': min(4, len(top_moves) - 1)
            }

            selected_move = top_moves[fallback_rank[error_type]]['Move']
            print(f"💥 Fallback {error_type}: {selected_move}")

            return selected_move

        except Exception as e:
            print(f"Error selecting {error_type}: {e}")
            return stockfish.get_best_move()

    def update_error_state(self, move_count, error_type):
        """Update error tracking after making an error"""
        self.errors_this_game[error_type] += 1
        self.last_error_move = move_count

        if error_type == 'blunder':
            self.just_blundered = True
            self.consecutive_good_moves = 0
            self.our_confidence_level *= 0.5
        elif error_type == 'mistake':
            self.consecutive_good_moves = 0
            self.our_confidence_level *= 0.7
        else:  # inaccuracy
            self.consecutive_good_moves = max(0, self.consecutive_good_moves - 2)
            self.our_confidence_level *= 0.9

        total_errors = sum(self.errors_this_game.values())
        print(f"📊 Error #{self.errors_this_game[error_type]} ({error_type}) - Total errors: {total_errors}")

    def log_error_summary(self):
        """Log summary of all errors made this game"""
        if self.game_error_log:
            print("\n🎯 ERROR SUMMARY THIS GAME:")
            error_counts = { 'inaccuracy': 0, 'mistake': 0, 'blunder': 0 }
            total_loss = 0

            for i, error in enumerate(self.game_error_log, 1):
                print(f"  Error {i}: Move {error['move_number']} - {error['move']} "
                      f"({error['type']}, lost {error['eval_loss']}cp)")
                error_counts[error['type']] += 1
                total_loss += error['eval_loss']

            print(f"\n  Totals: {error_counts['inaccuracy']} inaccuracies, "
                  f"{error_counts['mistake']} mistakes, {error_counts['blunder']} blunders")
            print(f"  Average error severity: {total_loss / len(self.game_error_log):.0f}cp")
            print(f"  Total evaluation lost: {total_loss}cp")
        else:
            print("🎯 Perfect game - no errors!")

    def add_thinking_variance(self, base_time):
        """Add natural variance to avoid detectable patterns - ADJUSTED FOR FASTER PLAY"""
        # Avoid too many similar thinking times
        if len(self.last_thinking_times) >= 3:
            recent_avg = sum(self.last_thinking_times[-3:]) / 3
            if abs(base_time - recent_avg) < 0.3:  # Reduced from 0.5
                # Add more variance if times are too similar
                variance_factor = random.choice([0.5, 0.6, 1.2, 1.3])  # Reduced range
                base_time *= variance_factor
                print(f"🎲 Pattern breaker: x{variance_factor}")

        # Occasional random spikes or dips - less extreme
        if random.random() < 0.08:  # Reduced from 10% to 8%
            if random.random() < 0.6:  # Bias toward quick moves
                # Quick move
                base_time *= random.uniform(0.3, 0.5)
                print("⚡ Random quick move")
            else:
                # Deep think - less extreme
                base_time *= random.uniform(1.3, 1.8)  # Reduced from 2.0
                print("🤔 Random deep think")

        return base_time

    def get_evaluation_spread(self, stockfish):
        """Get the spread of evaluations among top moves"""
        try:
            top_moves = stockfish.get_top_moves(5)
            if not top_moves or len(top_moves) < 2:
                return 1000  # High spread if we can't determine

            evaluations = []
            for move in top_moves:
                if 'Centipawn' in move:
                    evaluations.append(move['Centipawn'])

            if len(evaluations) >= 2:
                return max(evaluations) - min(evaluations)
            return 1000
        except:
            return 1000

    def is_opening_phase(self, move_count):
        """Check if we're still in opening phase"""
        return move_count < 20  # Extended opening phase

    def is_early_opening(self, move_count):
        """First moves should be lightning fast"""
        return move_count < 6  # Reduced from 10 for truly early opening

    def is_obvious_castling(self, board, move):
        """Castling is usually a quick decision"""
        try:
            chess_move = chess.Move.from_uci(move)
            return board.is_castling(chess_move)
        except:
            return False

    def is_obvious_en_passant(self, board, move):
        """En passant is usually obvious when available"""
        try:
            chess_move = chess.Move.from_uci(move)
            return board.is_en_passant(chess_move)
        except:
            return False

    def is_piece_development_move(self, board, move, move_count):
        """Early piece development should be very fast"""
        if move_count > 15:  # Extended from 12
            return False

        try:
            chess_move = chess.Move.from_uci(move)
            piece = board.piece_at(chess_move.from_square)

            if not piece:
                return False

            # Knights and bishops moving from starting positions
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Check if moving from starting square
                if piece.color == chess.WHITE:
                    starting_squares = [chess.B1, chess.G1, chess.C1, chess.F1]  # Nb1, Ng1, Bc1, Bf1
                else:
                    starting_squares = [chess.B8, chess.G8, chess.C8, chess.F8]  # Nb8, Ng8, Bc8, Bf8

                if chess_move.from_square in starting_squares:
                    return True

            # Central pawn moves in opening
            if piece.piece_type == chess.PAWN:
                target_square = chess.square_name(chess_move.to_square)
                if target_square in ['e4', 'e5', 'd4', 'd5', 'c4', 'c5', 'f4', 'f5']:
                    return True

            # Rook to open file (after castling)
            if piece.piece_type == chess.ROOK and move_count > 10:
                # Check if moving to an open or semi-open file
                file_idx = chess.square_file(chess_move.to_square)
                file_is_open = True
                for rank in range(8):
                    square = chess.square(file_idx, rank)
                    if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
                        file_is_open = False
                        break
                if file_is_open:
                    return True

        except:
            pass

        return False

    def opponent_just_blundered(self, board, stockfish):
        """Check if opponent just made a terrible move - humans double-check gifts"""
        if not self.last_opponent_move:
            return False

        try:
            current_eval = stockfish.get_evaluation()
            if not current_eval or current_eval['type'] != 'cp':
                return False

            current_score = current_eval.get('value', 0)

            # Flip score for our perspective
            if not board.turn:  # If it's our turn, flip the evaluation
                current_score = -current_score

            # Big evaluation swing suggests opponent blundered
            eval_swing = current_score - self.previous_evaluation
            if eval_swing > 300:  # We gained 3+ points suddenly
                return True

        except:
            pass

        return False

    def is_mate_in_one(self, board, move):
        """Check if this move delivers checkmate"""
        try:
            chess_move = chess.Move.from_uci(move)
            temp_board = board.copy()
            temp_board.push(chess_move)

            return temp_board.is_checkmate()
        except:
            return False

    def is_check_escape_only(self, board):
        """Moving out of check with very limited options"""
        if not board.is_check():
            return False

        legal_moves = list(board.legal_moves)
        return len(legal_moves) <= 3  # Very few ways to escape check

    def is_promotion_to_queen(self, move):
        """Queen promotions are usually obvious"""
        return len(move) == 5 and move[4] == 'q'

    def is_piece_hanging_obviously(self, board, move):
        """Enhanced detection for obviously hanging pieces"""
        try:
            chess_move = chess.Move.from_uci(move)

            if not board.is_capture(chess_move):
                return False

            captured_piece = board.piece_at(chess_move.to_square)
            if not captured_piece:
                return False

            # Queen or rook hanging = always obvious
            if captured_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                return True

            # For minor pieces, check if they're truly undefended
            if captured_piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                # Count how many pieces defend this square
                defenders = len(board.attackers(captured_piece.color, chess_move.to_square))

                # If it's completely undefended, it's obvious
                if defenders <= 1:  # Only the piece itself "defends" the square
                    return True

        except:
            pass

        return False

    def is_obvious_recapture(self, board, move):
        """Enhanced recapture detection - TRULY INSTANT"""
        if not self.last_opponent_move:
            return False

        try:
            last_move = chess.Move.from_uci(self.last_opponent_move)
            current_move = chess.Move.from_uci(move)

            # If opponent captured and we're capturing back on same square
            if board.is_capture(last_move) and current_move.to_square == last_move.to_square:
                # ALWAYS instant recapture - realistic human behavior
                return True

            return False
        except:
            return False

    def is_forced_move(self, board):
        """Check if there's only one legal move"""
        legal_moves = list(board.legal_moves)
        return len(legal_moves) == 1

    def get_piece_value(self, piece_type):
        """Get standard piece values"""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        return values.get(piece_type, 0)

    def is_free_piece_capture(self, board, move):
        """Check if we're capturing a completely undefended piece"""
        try:
            chess_move = chess.Move.from_uci(move)

            if not board.is_capture(chess_move):
                return False

            captured_piece = board.piece_at(chess_move.to_square)
            if not captured_piece:
                return False

            # High-value pieces hanging = always obvious
            if captured_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                return True

            # Minor pieces completely undefended
            if captured_piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                defenders = board.attackers(captured_piece.color, chess_move.to_square)
                if len(defenders) <= 1:  # Only the piece itself
                    return True

            return False

        except:
            return False

    def get_material_gain(self, board, move):
        """Calculate net material gain from a move"""
        try:
            chess_move = chess.Move.from_uci(move)
            material_gain = 0

            # Material gained from capture
            if board.is_capture(chess_move):
                captured_piece = board.piece_at(chess_move.to_square)
                if captured_piece:
                    material_gain += self.get_piece_value(captured_piece.piece_type)

            # Material lost if our piece gets captured
            temp_board = board.copy()
            temp_board.push(chess_move)

            # Check if our piece is now hanging
            attackers = temp_board.attackers(not board.turn, chess_move.to_square)
            if len(attackers) > 0:
                our_piece = temp_board.piece_at(chess_move.to_square)
                if our_piece:
                    # Find the lowest value attacker to see what we'll lose
                    attacker_values = []
                    for attacker_square in attackers:
                        attacker_piece = temp_board.piece_at(attacker_square)
                        if attacker_piece:
                            attacker_values.append(self.get_piece_value(attacker_piece.piece_type))

                    if attacker_values:
                        # We lose our piece to the lowest value attacker
                        material_gain -= self.get_piece_value(our_piece.piece_type)

            return material_gain

        except:
            return 0

    def is_huge_material_gain(self, board, move, stockfish):
        """Check if this move wins significantly more material than alternatives"""
        try:
            our_material_gain = self.get_material_gain(board, move)

            # If we're winning a queen+ for free, it's obvious
            if our_material_gain >= 8:
                return True

            return False

        except:
            return False

    def is_huge_evaluation_gap(self, board, move, stockfish):
        """Check if best move is much better than alternatives"""
        try:
            # Get top moves from stockfish
            top_moves = stockfish.get_top_moves(3)
            if not top_moves or len(top_moves) < 2:
                return False

            # Check if our move is the best and significantly better
            best_move = top_moves[0]
            if best_move['Move'] == move:
                best_eval = best_move.get('Centipawn', 0)
                second_eval = top_moves[1].get('Centipawn', 0)

                # If best move is 500+ cp better, it's very obvious
                return (best_eval - second_eval) >= 500

            return False

        except:
            return False

    def is_fork_followup(self, board, move):
        """Check if this is capturing after a fork we just made"""
        if not self.just_made_fork or not self.last_opponent_move:
            return False

        try:
            chess_move = chess.Move.from_uci(move)

            # If we're capturing high value after a fork, it's obvious
            if board.is_capture(chess_move):
                captured_piece = board.piece_at(chess_move.to_square)
                if captured_piece and captured_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP,
                                                                    chess.KNIGHT]:
                    return True

            return False

        except:
            return False

    def detect_fork_made(self, board, move):
        """Detect if we just made a fork (for next move detection)"""
        try:
            chess_move = chess.Move.from_uci(move)
            temp_board = board.copy()
            temp_board.push(chess_move)

            our_piece = temp_board.piece_at(chess_move.to_square)
            if not our_piece:
                return False

            # Count how many opponent pieces we're attacking
            attacked_pieces = []
            for square in chess.SQUARES:
                piece = temp_board.piece_at(square)
                if piece and piece.color != our_piece.color:
                    if temp_board.is_attacked_by(our_piece.color, square):
                        attacked_pieces.append(piece)

            # Fork = attacking 2+ valuable pieces (including pawns in some cases)
            valuable_attacked = [p for p in attacked_pieces if p.piece_type != chess.PAWN]
            if len(valuable_attacked) >= 2:
                return True

            # Also count if we're attacking king + another piece
            king_attacked = any(p.piece_type == chess.KING for p in attacked_pieces)
            if king_attacked and len(attacked_pieces) >= 2:
                return True

            return False

        except:
            return False

    def count_candidate_moves(self, board, stockfish):
        """Count number of reasonable candidate moves"""
        try:
            # Use Stockfish's multipv to get multiple good moves
            top_moves = stockfish.get_top_moves(5)
            if not top_moves:
                return min(5, len(list(board.legal_moves)))

            if len(top_moves) == 1:
                return 1

            # Count moves within reasonable margin of best move
            best_eval = top_moves[0].get('Centipawn', 0)
            candidates = 1  # Best move is always a candidate

            for move in top_moves[1:]:
                move_eval = move.get('Centipawn', 0)
                # Consider moves within 100cp of best as serious candidates
                if abs(best_eval - move_eval) <= 100:
                    candidates += 1
                else:
                    break  # Stop at first move that's too much worse

            return candidates

        except:
            # Fallback to counting legal moves
            return min(6, len(list(board.legal_moves)))

    def analyze_position_complexity(self, board, move_count):
        """Analyze position complexity and return thinking time modifier - ADJUSTED FOR FASTER PLAY"""
        try:
            complexity_score = 1.0

            # Piece count affects complexity
            piece_count = len(board.piece_map())
            if piece_count > 20 and move_count > 10:
                complexity_score *= 1.05  # Reduced from 1.1
            elif piece_count < 10:  # Endgame
                complexity_score *= 1.2  # Reduced from 1.4

            # Tactical indicators
            if board.is_check():
                complexity_score *= 1.2  # Reduced from 1.3

            # Lots of captures available = more tactical
            legal_moves = list(board.legal_moves)
            capture_moves = [m for m in legal_moves if board.is_capture(m)]
            if len(capture_moves) > 3:
                complexity_score *= 1.1  # Reduced from 1.2

            # Opening simplification - MUCH FASTER
            if move_count < 20:
                complexity_score *= 0.2  # Very fast opening play (was 0.4)

            return max(0.1, min(1.5, complexity_score))  # Reduced max from 2.0

        except:
            return 1.0

    def update_evaluation_tracking(self, stockfish):
        """Update our evaluation tracking for blunder detection"""
        try:
            current_eval = stockfish.get_evaluation()
            if current_eval and current_eval['type'] == 'cp':
                self.previous_evaluation = current_eval.get('value', 0)
        except:
            pass

    # NEW: Enhanced human-like move detection methods
    def is_backward_move(self, board, move):
        """Check if a move goes backward - humans miss these more often"""
        try:
            chess_move = chess.Move.from_uci(move)
            from_rank = chess.square_rank(chess_move.from_square)
            to_rank = chess.square_rank(chess_move.to_square)

            if board.turn == chess.WHITE:
                return to_rank < from_rank
            else:
                return to_rank > from_rank
        except:
            return False

    def is_long_knight_move(self, board, move):
        """Check if it's a long knight move - humans sometimes miss these"""
        try:
            chess_move = chess.Move.from_uci(move)
            piece = board.piece_at(chess_move.from_square)

            if piece and piece.piece_type == chess.KNIGHT:
                # Calculate distance
                from_file = chess.square_file(chess_move.from_square)
                from_rank = chess.square_rank(chess_move.from_square)
                to_file = chess.square_file(chess_move.to_square)
                to_rank = chess.square_rank(chess_move.to_square)

                # Long knight moves across the board
                if abs(from_file - to_file) + abs(from_rank - to_rank) >= 5:
                    return True
        except:
            return False

    def is_quiet_defensive_move(self, board, move):
        """Check if it's a quiet defensive move - humans often prefer active moves"""
        try:
            chess_move = chess.Move.from_uci(move)

            # Not a capture
            if board.is_capture(chess_move):
                return False

            # Moving backward or to the side
            if self.is_backward_move(board, move):
                return True

            # Moving to defend a piece
            to_square = chess_move.to_square
            defenders_before = len(board.attackers(board.turn, to_square))

            temp_board = board.copy()
            temp_board.push(chess_move)
            defenders_after = len(temp_board.attackers(board.turn, to_square))

            return defenders_after > defenders_before
        except:
            return False

    def is_in_between_move(self, board, move):
        """Check if it's an in-between move - humans often miss these"""
        try:
            # In-between moves are typically discovered attacks or defensive moves
            # that interrupt an expected sequence
            chess_move = chess.Move.from_uci(move)

            # Check if it creates a discovered attack
            temp_board = board.copy()
            temp_board.push(chess_move)

            # Simple check: does this move create new attacks?
            attacks_before = 0
            attacks_after = 0

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != board.turn:
                    attacks_before += len(board.attackers(board.turn, square))
                    attacks_after += len(temp_board.attackers(board.turn, square))

            return attacks_after > attacks_before + 2  # Significant increase in attacks
        except:
            return False

    def update_game_psychology(self, board, stockfish):
        """Update psychological state based on game situation"""
        try:
            eval_data = stockfish.get_evaluation()
            if eval_data and eval_data['type'] == 'cp':
                current_eval = eval_data['value']

                # Are we winning big?
                self.we_are_winning_big = current_eval > 500

                # Did we just blunder? (Updated to work with intentional blunders)
                eval_loss = self.previous_evaluation - current_eval
                if eval_loss > 200 and not self.force_error_this_move:
                    self.just_blundered = True
                    self.consecutive_good_moves = 0
                    self.our_confidence_level *= 0.7  # Confidence drops after blunder
                    self.recent_blunder_moves.append(len(board.move_stack))
                else:
                    # Don't count intentional blunders as "just_blundered"
                    if not self.force_error_this_move:
                        self.just_blundered = False

                # Are we "in the zone"?
                if abs(current_eval - self.previous_evaluation) < 50:
                    self.consecutive_good_moves += 1
                    if self.consecutive_good_moves > 5:
                        self.our_confidence_level = min(1.5, self.our_confidence_level * 1.1)

            # Complex position tracking
            legal_moves = list(board.legal_moves)
            if len(legal_moves) > 30:
                self.complex_position_streak += 1
            else:
                self.complex_position_streak = 0

        except:
            pass

    def add_psychological_delay(self, base_time):
        """Add psychological factors to thinking time - ADJUSTED FOR FASTER PLAY"""
        psychological_time = base_time

        # If we just blundered, we're more careful (but not too slow)
        if self.just_blundered:
            psychological_time *= random.uniform(1.3, 1.7)  # Reduced from (1.5, 2.0)
            print("😰 Post-blunder caution")

        # If we're winning big, we might play faster (overconfidence)
        if self.we_are_winning_big:
            psychological_time *= random.uniform(0.5, 0.7)  # Even faster when winning
            print("😎 Winning confidence boost")

        # If we're in the zone, smooth rhythm
        if self.consecutive_good_moves > 5:
            psychological_time *= random.uniform(0.6, 0.8)  # Faster in the zone
            print("🎯 In the zone!")

        # Complex positions make us think longer (but not too much)
        if self.complex_position_streak > 2:
            psychological_time *= random.uniform(1.2, 1.5)  # Reduced from (1.3, 1.7)
            print("🤯 Complex position fatigue")

        # Apply confidence level
        psychological_time *= (1.5 - self.our_confidence_level * 0.5)  # Adjusted formula

        return psychological_time

    def simulate_mouse_hesitation(self, move):
        """Simulate human mouse movement patterns - FASTER VERSION"""
        if not self.enable_human_delays:
            return

        start_pos, end_pos = self.get_move_pos(move)

        # Less hesitation overall for faster play
        # Sometimes hover over the piece before grabbing
        if random.random() < 0.2:  # Reduced from 0.3
            hover_x = start_pos[0] + random.randint(-5, 5)
            hover_y = start_pos[1] + random.randint(-5, 5)
            pyautogui.moveTo(hover_x, hover_y, duration=0.05)
            time.sleep(random.uniform(0.03, 0.08))  # Reduced times

        # Rarely almost grab wrong piece (only when not in time pressure)
        if random.random() < 0.03 and not self.is_forced_move(chess.Board()) and not self.in_time_pressure:
            wrong_x = start_pos[0] + random.choice([-50, 50])
            wrong_y = start_pos[1] + random.choice([-50, 50])
            pyautogui.moveTo(wrong_x, wrong_y, duration=0.15)
            time.sleep(0.05)
            print("🤦 Almost grabbed wrong piece!")

    def should_miss_this_move(self, board, move, stockfish):
        """Determine if we should realistically miss this move"""
        # Never miss mate in 1-2
        if self.is_mate_in_one(board, move):
            return False

        miss_probability = 0.0

        # Backward moves - humans miss these more
        if self.is_backward_move(board, move):
            miss_probability += 0.15

        # Long knight moves
        if self.is_long_knight_move(board, move):
            miss_probability += 0.12

        # Quiet defensive moves when winning
        if self.we_are_winning_big and self.is_quiet_defensive_move(board, move):
            miss_probability += 0.20

        # In-between moves
        if self.is_in_between_move(board, move):
            miss_probability += 0.18

        # Reduce miss chance if we're being careful post-blunder
        if self.just_blundered:
            miss_probability *= 0.3

        # Increase miss chance if overconfident
        if self.consecutive_good_moves > 8:
            miss_probability *= 1.5

        return random.random() < min(0.4, miss_probability)

    # Converts a move to screen coordinates
    def move_to_screen_pos(self, move):
        print(f"=== Calculating coordinates for move: {move} ===")

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

        file_num = char_to_num(move[0])
        rank_num = int(move[1])
        print(f"Target square: {move[0:2]} = file {file_num}, rank {rank_num}")

        if self.is_white:
            x = board_x + square_size * (file_num - 1) + square_size / 2
            y = board_y + square_size * (8 - rank_num) + square_size / 2
            print(f"White perspective calculation:")
        else:
            x = board_x + square_size * (8 - file_num) + square_size / 2
            y = board_y + square_size * (rank_num - 1) + square_size / 2
            print(f"Black perspective calculation:")

        print(f"Final coordinates: x={x}, y={y}")
        print(f"=== End coordinate calculation ===")
        return x, y

    def get_move_pos(self, move):
        start_pos_x, start_pos_y = self.move_to_screen_pos(move[0:2])
        end_pos_x, end_pos_y = self.move_to_screen_pos(move[2:4])
        return (start_pos_x, start_pos_y), (end_pos_x, end_pos_y)

    def make_move(self, move):
        print(f"make_move called with: {move}")

        # NEW: Add mouse hesitation patterns
        self.simulate_mouse_hesitation(move)

        start_pos, end_pos = self.get_move_pos(move)
        print(f"Start position: {start_pos}, End position: {end_pos}")

        print(f"Moving mouse to start position: {start_pos}")

        # FASTER mouse movement for quicker play
        movement_duration = random.uniform(0.05, 0.15) if self.enable_human_delays else 0
        pyautogui.moveTo(start_pos[0], start_pos[1], duration=movement_duration)
        time.sleep(self.mouse_latency)

        print("Clicking and holding at start position")
        pyautogui.mouseDown()

        # FASTER hold time
        hold_time = random.uniform(0.05, 0.15) if self.enable_human_delays else 0.1
        time.sleep(hold_time)

        print(f"Dragging to end position: {end_pos}")

        # FASTER drag speed based on move type
        if self.enable_human_delays:
            if self.critical_time_pressure:
                drag_duration = random.uniform(0.05, 0.1)  # PANIC mode
            elif self.severe_time_pressure:
                drag_duration = random.uniform(0.1, 0.2)  # Very fast
            elif self.is_obvious_recapture(chess.Board(), move):
                drag_duration = random.uniform(0.1, 0.15)  # Instant recapture
            elif self.just_blundered:
                drag_duration = random.uniform(0.4, 0.6)  # Still careful after blunder
            else:
                drag_duration = random.uniform(0.2, 0.4)  # Normal but faster
        else:
            drag_duration = 0.1

        pyautogui.moveTo(end_pos[0], end_pos[1], duration=drag_duration)

        # Rare slight overshoot (less common for faster play)
        if self.enable_human_delays and random.random() < 0.05 and not self.critical_time_pressure:
            overshoot_x = end_pos[0] + random.randint(-10, 10)
            overshoot_y = end_pos[1] + random.randint(-10, 10)
            pyautogui.moveTo(overshoot_x, overshoot_y, duration=0.05)
            time.sleep(0.03)
            pyautogui.moveTo(end_pos[0], end_pos[1], duration=0.05)

        time.sleep(0.1)  # Reduced from 0.2

        print("Releasing mouse at end position")
        pyautogui.mouseUp()

        # FASTER post-move delay
        if self.enable_human_delays:
            if self.critical_time_pressure:
                post_delay = 0.05  # Almost no delay
            elif self.is_mate_in_one(chess.Board(), move):
                post_delay = random.uniform(0.3, 0.6)  # Pause after mate
            else:
                post_delay = random.uniform(0.1, 0.2)  # Faster normal delay
        else:
            post_delay = 0.15  # Reduced from 0.3
        time.sleep(post_delay)
        print("Move completed")

        # Handle promotion
        if len(move) == 5:
            print(f"Promotion detected: {move[4]}")
            time.sleep(0.3)  # Reduced from 0.5
            end_pos_x = None
            end_pos_y = None
            if move[4] == "n":
                end_pos_x, end_pos_y = self.move_to_screen_pos(move[2] + str(int(move[3]) - 1))
            elif move[4] == "r":
                end_pos_x, end_pos_y = self.move_to_screen_pos(move[2] + str(int(move[3]) - 2))
            elif move[4] == "b":
                end_pos_x, end_pos_y = self.move_to_screen_pos(move[2] + str(int(move[3]) - 3))

            print(f"Clicking promotion piece at: {end_pos_x}, {end_pos_y}")
            pyautogui.moveTo(x=end_pos_x, y=end_pos_y)
            pyautogui.click(button='left')

    def wait_for_gui_to_delete(self):
        while self.pipe.recv() != "DELETE":
            pass

    def go_to_next_puzzle(self):
        self.grabber.click_puzzle_next()
        self.pipe.send("RESTART")
        self.wait_for_gui_to_delete()

    def find_new_online_match(self):
        time.sleep(2)
        self.grabber.click_game_next()
        self.pipe.send("RESTART")
        self.wait_for_gui_to_delete()

    def run(self):
        try:
            print("Starting StockfishBot process...")

            # Initialize error system for this game
            self.errors_this_game = { 'inaccuracy': 0, 'mistake': 0, 'blunder': 0 }
            self.last_error_move = 0
            self.force_error_this_move = None
            self.game_error_log = []

            if self.enable_errors:
                print(f"🎯 Error system enabled:")
                print(f"   Target: {self.errors_per_game['inaccuracy']} inaccuracies, "
                      f"{self.errors_per_game['mistake']} mistakes, "
                      f"{self.errors_per_game['blunder']} blunders")

            # Initialize grabber
            print("Initializing grabber...")
            if self.website == "chesscom":
                self.grabber = ChesscomGrabber(self.chrome_url, self.chrome_session_id)
            else:
                self.grabber = LichessGrabber(self.chrome_url, self.chrome_session_id)
            print("Grabber initialized successfully")

            # Detect time control
            self.detect_time_control()
            print(f"⏰ Time control detected: {self.time_control_type}")

            # Initialize Stockfish
            print("Initializing Stockfish...")
            parameters = {
                "Threads": self.cpu_threads,
                "Hash": self.memory,
                "Ponder": "true",
                "Slow Mover": self.slow_mover,
                "Skill Level": self.skill_level,
                "MultiPV": 10  # NEW: Increased for better human-like selection
            }

            try:
                # Initialize the appropriate engine based on type
                if self.engine_type.lower() == "neural":
                    # Import neural adapter only when needed - LAZY IMPORT
                    try:
                        from nnue_adapter import NNUEAdapter
                    except ImportError as e:
                        print(f"Failed to import NNUEAdapter: {e}")
                        self.pipe.send(f"ERR_GENERAL_Cannot import neural adapter: {str(e)}")
                        return

                    # Neural network engines don't use skill level, but do use other parameters
                    nn_parameters = {
                        "Threads": self.cpu_threads,
                        "Hash": self.memory,
                        "MultiPV": 10,  # For better move selection
                        "Ponder": "true"
                    }

                    # Initialize with the NNUEAdapter
                    stockfish = NNUEAdapter(path=self.stockfish_path, depth=self.stockfish_depth,
                                            parameters=nn_parameters)
                    print(f"Neural network engine initialized successfully: {self.stockfish_path}")
                else:
                    # Standard Stockfish initialization
                    stockfish = Stockfish(path=self.stockfish_path, depth=self.stockfish_depth, parameters=parameters)
                    print("Stockfish initialized successfully")

                if self.human_accuracy:
                    print(f"🎭 Human accuracy enabled (threshold: {self.accuracy_threshold}cp)")
                    print(f"🎯 Target accuracy: ~85-90% (realistic for skill {self.skill_level})")
                    print(f"🧠 Psychological modeling: ON")
                    print(f"💥 Complete error system: ON")
                    print(f"⚡ FAST PLAY MODE: Average move time reduced")
                else:
                    print("🤖 Playing at maximum strength")
            except PermissionError as e:
                print(f"Engine permission error: {e}")
                self.pipe.send("ERR_PERM")
                return
            except OSError as e:
                print(f"Engine OS error: {e}")
                self.pipe.send("ERR_EXE")
                return
            except Exception as e:
                print(f"Engine initialization error: {e}")
                self.pipe.send(f"ERR_GENERAL_{str(e)}")
                return

            print("Updating board element...")
            self.grabber.update_board_elem()
            if self.grabber.get_board() is None:
                print("Board element not found")
                self.pipe.send("ERR_BOARD")
                return
            print("Board element found")

            print("Determining player color...")
            self.is_white = self.grabber.is_white()
            if self.is_white is None:
                print("Could not determine player color")
                self.pipe.send("ERR_COLOR")
                return
            print(f"Player is {'white' if self.is_white else 'black'}")

            print("Getting move list...")
            move_list = self.grabber.get_move_list()
            if move_list is None:
                print("Could not get move list")
                self.pipe.send("ERR_MOVES")
                return
            print(f"Move list: {move_list}")

            # Check if game is over
            score_pattern = r"([0-9]+)\-([0-9]+)"
            if len(move_list) > 0 and re.match(score_pattern, move_list[-1]):
                print("Game is already over")
                self.pipe.send("ERR_GAMEOVER")
                return

            print("Setting up chess board...")
            board = chess.Board()
            for move in move_list:
                board.push_san(move)
            move_list_uci = [move.uci() for move in board.move_stack]

            stockfish.set_position(move_list_uci)
            print("Chess board setup complete")

            # Initialize evaluation tracking
            self.update_evaluation_tracking(stockfish)

            print("Sending START signal to GUI...")
            time.sleep(0.5)
            self.pipe.send("START")

            if len(move_list) > 0:
                self.pipe.send("M_MOVE" + ",".join(move_list))

            print("Entering main game loop...")

            while True:
                # Our turn
                if (self.is_white and board.turn == chess.WHITE) or (not self.is_white and board.turn == chess.BLACK):
                    print("It's our turn, thinking of a move...")

                    move = None
                    move_count = len(board.move_stack)

                    # Bongcloud logic
                    if self.bongcloud and move_count <= 3:
                        if move_count == 0:
                            move = "e2e3"
                        elif move_count == 1:
                            move = "e7e6"
                        elif move_count == 2:
                            move = "e1e2"
                        elif move_count == 3:
                            move = "e8e7"

                        if not board.is_legal(chess.Move.from_uci(move)):
                            move = self.select_human_like_move(stockfish, board, move_count)
                    else:
                        move = self.select_human_like_move(stockfish, board, move_count)

                    print(f"Move selected: {move}")

                    # Calculate ultra-enhanced thinking time
                    thinking_time = self.calculate_thinking_time(board, move, move_count, stockfish)
                    if thinking_time > 0.1:
                        time.sleep(thinking_time)

                    # Manual mode handling
                    self_moved = False
                    if self.enable_manual_mode:
                        print("Manual mode enabled, waiting for keypress...")
                        move_start_pos, move_end_pos = self.get_move_pos(move)
                        self.overlay_queue.put([
                            ((int(move_start_pos[0]), int(move_start_pos[1])),
                             (int(move_end_pos[0]), int(move_end_pos[1]))),
                        ])
                        while True:
                            if keyboard.is_pressed("3"):
                                print("Key '3' pressed, making move...")
                                break

                            if len(move_list) != len(self.grabber.get_move_list()):
                                print("Player made a move manually")
                                self_moved = True
                                move_list = self.grabber.get_move_list()
                                move_san = move_list[-1]
                                move = board.parse_san(move_san).uci()
                                board.push_uci(move)
                                stockfish.make_moves_from_current_position([move])
                                break

                    if not self_moved:
                        print(f"✅ Making move: {move}")

                        # Detect if we're making a fork (for next move)
                        self.just_made_fork = self.detect_fork_made(board, move)
                        if self.just_made_fork:
                            print("🍴 Fork detected - next move might be quick!")

                        try:
                            chess_move = chess.Move.from_uci(move)
                            print(f"📋 Converting move {move} to SAN notation...")
                            print(f"📋 Board FEN: {board.fen()}")
                            print(f"📋 Is move legal? {chess_move in board.legal_moves}")

                            if chess_move in board.legal_moves:
                                move_san = board.san(chess_move)
                                board.push_uci(move)
                                stockfish.make_moves_from_current_position([move])
                                move_list.append(move_san)
                            else:
                                legal_moves_list = [m.uci() for m in list(board.legal_moves)]
                                print(f"❌ Move {move} is not legal!")
                                print(f"📋 Legal moves: {legal_moves_list[:5]}...")
                                continue
                        except Exception as e:
                            print(f"❌ Error processing move {move}: {str(e)}")
                            print(f"📋 Board state: {board.fen()}")
                            continue

                        # Store our move
                        self.last_our_move = move

                        # Update evaluation tracking
                        self.update_evaluation_tracking(stockfish)

                        if self.enable_mouseless_mode and not self.grabber.is_game_puzzles():
                            self.grabber.make_mouseless_move(move, move_count + 1)
                        else:
                            self.make_move(move)

                    self.overlay_queue.put([])
                    self.pipe.send("S_MOVE" + move_san)

                    # Check for checkmate
                    if board.is_checkmate():
                        print("Checkmate! Game over.")

                        # Log error summary
                        self.log_error_summary()

                        if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                            self.go_to_next_puzzle()
                        elif self.enable_non_stop_matches and not self.enable_non_stop_puzzles:
                            self.find_new_online_match()
                        return

                    time.sleep(0.1)

                # Wait for opponent move
                print("Waiting for opponent move...")
                previous_move_list = move_list.copy()
                while True:
                    if self.grabber.is_game_over():
                        print("Game over detected")

                        # Log error summary
                        self.log_error_summary()

                        if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                            self.go_to_next_puzzle()
                        elif self.enable_non_stop_matches and not self.enable_non_stop_puzzles:
                            self.find_new_online_match()
                        return
                    move_list = self.grabber.get_move_list()
                    if move_list is None:
                        print("Move list became None")
                        return
                    if len(move_list) > len(previous_move_list):
                        break

                # Process opponent move
                move = move_list[-1]
                print(f"Opponent move: {move}")
                self.pipe.send("S_MOVE" + move)

                # Store opponent move
                try:
                    opponent_move_uci = board.parse_san(move).uci()
                    self.last_opponent_move = opponent_move_uci
                except:
                    pass

                board.push_san(move)
                stockfish.make_moves_from_current_position([str(board.peek())])

                # Update evaluation tracking after opponent move
                self.update_evaluation_tracking(stockfish)

                if board.is_checkmate():
                    print("Opponent checkmate! Game over.")

                    # Log error summary
                    self.log_error_summary()

                    if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                        self.go_to_next_puzzle()
                    elif self.enable_non_stop_matches and not self.enable_non_stop_puzzles:
                        self.find_new_online_match()
                    return

        except Exception as e:
            print(f"StockfishBot error: {e}")
            print(f"Error type: {type(e)}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error details: {exc_type} {fname} {exc_tb.tb_lineno}")

            try:
                self.pipe.send(f"ERR_GENERAL_{str(e)}")
            except:
                print("Could not send error to GUI - pipe is broken")
            return
