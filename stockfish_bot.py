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
from mouse_controller import MouseController, GameState
from accuracy import HumanAccuracySystem


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
        self.bullet_max_time = 0.5  # Reduced from 8.0
        self.blitz_max_time = 5.0  # Reduced from 15.0
        self.rapid_max_time = 20.0  # Reduced from 30.0

        # Human accuracy parameters
        self.human_accuracy = human_accuracy

        # Initialize accuracy system
        self.accuracy_system = HumanAccuracySystem(
            skill_level=skill_level,
            enable_errors=enable_blunders,
            errors_per_game={
                'inaccuracy': 4,
                'mistake': 2,
                'blunder': blunders_per_game
            }
        )

        # Track game state for advanced move detection
        self.last_opponent_move = None
        self.last_our_move = None
        self.just_made_fork = False
        self.previous_evaluation = 0  # Track evaluation swings

        # Track opening book moves for instant play
        self.opening_book_ended = False  # Are we out of book?

        # NEW: Time management tracking
        self.our_time_remaining = None  # Track our clock
        self.opponent_time_remaining = None  # Track opponent's clock
        self.initial_time = None  # Starting time for the game
        self.time_control_type = None  # bullet/blitz/rapid
        self.last_thinking_times = []  # Track recent thinking times for patterns

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

        # Initialize mouse controller
        self.mouse_controller = None

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

            # Update accuracy system's time pressure
            self.accuracy_system.update_time_pressure(time_remaining, self.time_control_type)

            # Update opponent time pressure
            opp_time = self.get_opponent_time_remaining()
            if opp_time:
                self.accuracy_system.opponent_time_pressure = (
                    opp_time < 20 if self.time_control_type == "bullet" else opp_time < 40
                )

        except:
            pass

    def calculate_thinking_time(self, board, move, move_count, stockfish=None):
        """
        ULTRA-ENHANCED human-like thinking with FASTER average play
        """
        if not self.enable_human_delays:
            return 0.1

        # Update time pressure state
        self.update_time_pressure_state()

        # CRITICAL TIME PRESSURE - Ultra fast moves to avoid flagging
        if self.accuracy_system.critical_time_pressure:
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
        if self.accuracy_system.severe_time_pressure:
            # Still panic but with slight variation
            panic_time = random.uniform(0.2, 0.5)
            # Even faster if we've been in time pressure for many moves
            if self.accuracy_system.moves_in_time_pressure > 5:
                panic_time *= 0.5
            print(f"🚨 SEVERE TIME PRESSURE: {panic_time:.1f}s")
            return panic_time

        # ERROR THINKING PATTERNS - Different for each error type
        if hasattr(self, 'force_error_this_move') and self.force_error_this_move:
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
        if self.accuracy_system.in_time_pressure:
            max_allowed = 1.0 if not self.accuracy_system.severe_time_pressure else 0.5
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
            if not self.accuracy_system.severe_time_pressure:
                time_based_max = max(0.5, time_based_max)  # Reduced from 1.0

            return min(base_max, time_based_max)

        return base_max

    def select_human_like_move(self, stockfish, board, move_count):
        """
        Enhanced human-like move selection using accuracy system
        """
        if not self.human_accuracy:
            return stockfish.get_best_move()

        try:
            # Get current evaluation
            current_eval = stockfish.get_evaluation()
            position_eval = current_eval.get('value', 0) if current_eval and current_eval['type'] == 'cp' else 0

            # Update psychological state in accuracy system
            self.accuracy_system.update_psychological_state(
                board,
                position_eval,
                self.previous_evaluation,
                self.get_time_remaining(),
                self.get_opponent_time_remaining()
            )

            # Check if we should force an error this move
            error_type = self.accuracy_system.should_force_error_now(
                board, move_count, position_eval, self.get_time_remaining()
            )

            if error_type:
                # Get top moves for error selection
                top_moves = stockfish.get_top_moves(12)
                if top_moves and len(top_moves) >= 2:
                    best_eval = top_moves[0].get('Centipawn', 0)

                    # Select error move from accuracy system
                    error_move, eval_loss = self.accuracy_system.select_error_move(
                        top_moves, best_eval, error_type
                    )

                    if error_move:
                        # Record the error
                        self.accuracy_system.record_error(move_count, error_move, eval_loss, error_type)
                        # Set flag for thinking time adjustment
                        self.force_error_this_move = error_type
                        return error_move

            # Normal move selection
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
            if self.accuracy_system.severe_time_pressure:
                # Just play a reasonable move quickly
                max_candidates = 3
                quick_candidates = top_moves[:max_candidates]
                weights = [0.6, 0.3, 0.1][:len(quick_candidates)]
                selected = random.choices(quick_candidates, weights=weights)[0]['Move']
                print(f"⏰ Time pressure move: {selected}")
                return selected

            # Build candidate list using accuracy system's threshold
            candidate_moves = []
            threshold = self.accuracy_system.get_accuracy_threshold()

            for i, move in enumerate(top_moves):
                move_eval = move.get('Centipawn', 0)
                eval_diff = abs(best_eval - move_eval)

                if i == 0:  # Always include best
                    candidate_moves.append(move)
                elif self.accuracy_system.should_play_suboptimal(eval_diff, i):
                    candidate_moves.append(move)
                    print(f"🎲 Candidate {len(candidate_moves)}: {move['Move']} (-{eval_diff}cp)")
                elif eval_diff > threshold * 3:
                    break

            # Selection with realistic weights
            if len(candidate_moves) == 1:
                selected_move = candidate_moves[0]['Move']
            else:
                # Weight distribution for more realistic play
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

    def add_psychological_delay(self, base_time):
        """Add psychological factors to thinking time - ADJUSTED FOR FASTER PLAY"""
        psychological_time = base_time

        # If we just blundered, we're more careful (but not too slow)
        if self.accuracy_system.just_blundered:
            psychological_time *= random.uniform(1.3, 1.7)  # Reduced from (1.5, 2.0)
            print("😰 Post-blunder caution")

        # If we're winning big, we might play faster (overconfidence)
        if self.accuracy_system.we_are_winning_big:
            psychological_time *= random.uniform(0.5, 0.7)  # Even faster when winning
            print("😎 Winning confidence boost")

        # If we're in the zone, smooth rhythm
        if self.accuracy_system.consecutive_good_moves > 5:
            psychological_time *= random.uniform(0.6, 0.8)  # Faster in the zone
            print("🎯 In the zone!")

        # Complex positions make us think longer (but not too much)
        if self.accuracy_system.complex_position_streak > 2:
            psychological_time *= random.uniform(1.2, 1.5)  # Reduced from (1.3, 1.7)
            print("🤯 Complex position fatigue")

        # Apply confidence level
        psychological_time *= (1.5 - self.accuracy_system.confidence.current_confidence * 0.5)  # Adjusted formula

        return psychological_time

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

    def get_current_game_state(self, board):
        """Create a GameState object with current game information"""
        return GameState(
            critical_time_pressure=self.accuracy_system.critical_time_pressure,
            severe_time_pressure=self.accuracy_system.severe_time_pressure,
            in_time_pressure=self.accuracy_system.in_time_pressure,
            just_blundered=self.accuracy_system.just_blundered,
            we_are_winning_big=self.accuracy_system.we_are_winning_big,
            consecutive_good_moves=self.accuracy_system.consecutive_good_moves,
            our_confidence_level=self.accuracy_system.confidence.current_confidence,
            moves_in_time_pressure=self.accuracy_system.moves_in_time_pressure,
            game_phase="opening" if len(board.move_stack) < 20 else "middlegame" if len(
                board.move_stack) < 40 else "endgame",
            session_duration=time.time() - self.session_start_time if hasattr(self, 'session_start_time') else 0,
            total_moves_made=len(board.move_stack) // 2 if self.is_white else (len(board.move_stack) + 1) // 2,
            recent_move_times=self.last_thinking_times.copy()
        )

    def run(self):
        try:
            print("Starting StockfishBot process...")

            # Initialize session start time
            self.session_start_time = time.time()

            # Reset accuracy system for new game
            self.accuracy_system.reset_for_new_game()

            if self.accuracy_system.enable_errors:
                print(f"🎯 Error system enabled:")
                print(f"   Target: {self.accuracy_system.errors_per_game['inaccuracy']} inaccuracies, "
                      f"{self.accuracy_system.errors_per_game['mistake']} mistakes, "
                      f"{self.accuracy_system.errors_per_game['blunder']} blunders")

            # Initialize grabber
            print("Initializing grabber...")
            if self.website == "chesscom":
                self.grabber = ChesscomGrabber(self.chrome_url, self.chrome_session_id)
            else:
                self.grabber = LichessGrabber(self.chrome_url, self.chrome_session_id)
            print("Grabber initialized successfully")

            # Initialize mouse controller
            self.mouse_controller = MouseController(self.grabber, self.mouse_latency, self.enable_human_delays)

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
                    print(f"🎭 Human accuracy enabled (skill level: {self.skill_level})")
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

            # Set player color in mouse controller
            self.mouse_controller.set_player_color(self.is_white)

            print("Getting move list...")
            move_list = self.grabber.get_move_list()
            if move_list is None:
                print("Could not get move list")
                self.pipe.send("ERR_MOVES")
                return
            print(f"Move list: {move_list}")

            # NEW: Check for game status messages
            game_end_keywords = ['Gameaborted', 'Draw', 'Stalemate', 'Checkmate', 'Timeout',
                                 'Resignation', 'Abandoned', 'aborted', 'draw', 'resigned']
            if move_list and any(keyword.lower() in str(move_list[-1]).lower() for keyword in game_end_keywords):
                print(f"Game already ended with status: {move_list[-1]}")
                self.pipe.send("ERR_GAMEOVER")

                # Handle auto-next if enabled
                if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                    self.go_to_next_puzzle()
                elif self.enable_non_stop_matches:
                    self.find_new_online_match()
                return

            # Check if game is over (score pattern)
            score_pattern = r"([0-9]+)-([0-9]+)"
            if len(move_list) > 0 and re.match(score_pattern, move_list[-1]):
                print("Game is already over")
                self.pipe.send("ERR_GAMEOVER")

                # Handle auto-next if enabled
                if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                    self.go_to_next_puzzle()
                elif self.enable_non_stop_matches:
                    self.find_new_online_match()
                return

            print("Setting up chess board...")
            board = chess.Board()

            # Parse moves with error handling
            valid_moves = []
            for move in move_list:
                try:
                    # Skip if move looks like a status message
                    if any(keyword in str(move).lower() for keyword in ['abort', 'draw', 'resign', 'mate', 'time']):
                        print(f"Skipping status message: {move}")
                        continue

                    board.push_san(move)
                    valid_moves.append(move)
                except chess.InvalidMoveError as e:
                    print(f"Skipping invalid move/status: {move} (Error: {e})")
                    continue
                except Exception as e:
                    print(f"Unexpected error parsing move {move}: {e}")
                    continue

            move_list = valid_moves
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

                    # Clear error flag if it was set
                    if hasattr(self, 'force_error_this_move'):
                        self.force_error_this_move = None

                    # Manual mode handling
                    self_moved = False
                    if self.enable_manual_mode:
                        print("Manual mode enabled, waiting for keypress...")
                        start_pos, end_pos = self.mouse_controller.get_move_positions(move)
                        self.overlay_queue.put([
                            ((int(start_pos[0]), int(start_pos[1])),
                             (int(end_pos[0]), int(end_pos[1]))),
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
                            # Get current game state and make the move
                            game_state = self.get_current_game_state(board)
                            self.mouse_controller.make_move(move, board, game_state)

                    self.overlay_queue.put([])
                    self.pipe.send("S_MOVE" + move_san)

                    # Check for checkmate
                    if board.is_checkmate():
                        print("Checkmate! Game over.")

                        # Log error summary
                        self.accuracy_system.log_game_summary()

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
                        self.accuracy_system.log_game_summary()

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

                # Check if the move is actually a game status
                if any(keyword in str(move).lower() for keyword in ['abort', 'draw', 'resign', 'mate', 'time']):
                    print(f"Game ended with status: {move}")

                    # Log error summary
                    self.accuracy_system.log_game_summary()

                    if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                        self.go_to_next_puzzle()
                    elif self.enable_non_stop_matches:
                        self.find_new_online_match()
                    return

                self.pipe.send("S_MOVE" + move)

                # Store opponent move
                try:
                    opponent_move_uci = board.parse_san(move).uci()
                    self.last_opponent_move = opponent_move_uci
                except Exception as e:
                    print(f"Error parsing opponent move {move}: {e}")
                    # Continue anyway, might be game end
                    pass

                try:
                    board.push_san(move)
                    stockfish.make_moves_from_current_position([str(board.peek())])
                except Exception as e:
                    print(f"Error processing opponent move: {e}")
                    # Check if game ended
                    if self.grabber.is_game_over():
                        print("Game over after opponent move")

                        # Log error summary
                        self.accuracy_system.log_game_summary()

                        if self.enable_non_stop_puzzles and self.grabber.is_game_puzzles():
                            self.go_to_next_puzzle()
                        elif self.enable_non_stop_matches:
                            self.find_new_online_match()
                        return

                # Update evaluation tracking after opponent move
                self.update_evaluation_tracking(stockfish)

                if board.is_checkmate():
                    print("Opponent checkmate! Game over.")

                    # Log error summary
                    self.accuracy_system.log_game_summary()

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
