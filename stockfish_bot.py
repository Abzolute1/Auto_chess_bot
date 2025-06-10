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

        # Human-like thinking parameters
        self.enable_human_delays = enable_human_delays
        self.min_thinking_time = min_thinking_time
        self.max_thinking_time = max_thinking_time

        # Human accuracy parameters - FIXED THRESHOLDS
        self.human_accuracy = human_accuracy  # Enable human-like move selection (not always best move)
        self.accuracy_threshold = accuracy_threshold  # Centipawns within which to randomize (75 = more mistakes)

        # NEW: Blunder System Parameters
        self.enable_blunders = enable_blunders  # Enable deliberate blundering
        self.blunders_per_game = blunders_per_game  # Target blunders per game (1-3 realistic)
        self.blunder_severity = blunder_severity  # How bad the blunder should be (100-300cp)

        # Blunder tracking variables
        self.blunders_this_game = 0
        self.last_blunder_move = 0
        self.force_blunder_this_move = False
        self.blunder_cooldown = 10  # Minimum moves between blunders
        self.game_blunder_log = []  # Track blunders for analysis

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

    def should_force_blunder_now(self, board, move_count, stockfish):
        """
        Determine if we should force a blunder on this move
        """
        if not self.enable_blunders:
            return False

        # Already made enough blunders this game
        if self.blunders_this_game >= self.blunders_per_game:
            return False

        # Too soon after last blunder
        if move_count - self.last_blunder_move < self.blunder_cooldown:
            return False

        # Don't blunder in critical positions
        if self.is_critical_position(board, stockfish):
            return False

        # Game length consideration - space blunders throughout the game
        expected_game_length = 60  # Assume ~60 moves average
        moves_played = move_count

        # Calculate if we're falling behind on our blunder schedule
        expected_blunders_by_now = (moves_played / expected_game_length) * self.blunders_per_game
        blunder_deficit = expected_blunders_by_now - self.blunders_this_game

        # Force blunder if we're significantly behind schedule and in mid/late game
        if blunder_deficit >= 1.0 and move_count > 15:
            # Higher chance if we're really behind on blunders
            force_chance = min(0.4, blunder_deficit * 0.2)

            # Increase chance if we're winning (humans get overconfident)
            if self.we_are_winning_big:
                force_chance *= 1.5

            # Increase chance if we've made many good moves in a row
            if self.consecutive_good_moves > 8:
                force_chance *= 1.3

            return random.random() < force_chance

        return False

    def is_critical_position(self, board, stockfish):
        """
        Check if this is a critical position where we shouldn't blunder
        """
        try:
            # Don't blunder when already losing badly
            eval_data = stockfish.get_evaluation()
            if eval_data and eval_data['type'] == 'cp':
                current_eval = eval_data['value']
                # Flip for our perspective
                if not board.turn:
                    current_eval = -current_eval

                # Don't blunder if we're already losing by 3+ points
                if current_eval < -300:
                    return True

            # Don't blunder in very early opening (first 8 moves)
            move_count = len(board.move_stack)
            if move_count < 8:
                return True

            # Don't blunder when in check
            if board.is_check():
                return True

            # Don't blunder if only a few legal moves (forced positions)
            if len(list(board.legal_moves)) <= 2:
                return True

            # Don't blunder if mate is imminent (either way)
            top_moves = stockfish.get_top_moves(1)
            if top_moves and 'Mate' in top_moves[0]:
                mate_in = abs(top_moves[0].get('Mate', 0))
                if mate_in <= 3:
                    return True

            return False

        except Exception as e:
            print(f"Error checking critical position: {e}")
            return True  # Be safe, don't blunder if we can't evaluate

    def select_blunder_move(self, stockfish, board, move_count):
        """
        Select a move that constitutes a realistic blunder
        """
        try:
            print("🤯 SELECTING BLUNDER MOVE...")

            # Get top moves to understand what we're avoiding
            top_moves = stockfish.get_top_moves(8)
            if not top_moves or len(top_moves) < 2:
                print("❌ Not enough moves to blunder - playing best")
                return top_moves[0]['Move'] if top_moves else stockfish.get_best_move()

            best_move = top_moves[0]
            best_eval = best_move.get('Centipawn', 0)

            # Find moves that are significantly worse but not completely insane
            blunder_candidates = []

            for i, move in enumerate(top_moves[1:], 1):
                move_eval = move.get('Centipawn', 0)
                eval_loss = best_eval - move_eval

                # Look for moves that lose between blunder_severity and blunder_severity*2 centipawns
                min_loss = self.blunder_severity
                max_loss = self.blunder_severity * 2

                if min_loss <= eval_loss <= max_loss:
                    # Make sure it's not a mate blunder
                    if 'Mate' not in move or abs(move.get('Mate', 0)) > 10:
                        blunder_candidates.append({
                            'move': move['Move'],
                            'eval_loss': eval_loss,
                            'rank': i
                        })

            # If no good blunder candidates in top moves, look at more moves
            if not blunder_candidates:
                print("🔍 Looking for blunder in lower-ranked moves...")

                # Get all legal moves and evaluate a few more
                legal_moves = list(board.legal_moves)
                random.shuffle(legal_moves)

                for chess_move in legal_moves[:10]:  # Check 10 random legal moves
                    move_uci = chess_move.uci()

                    # Skip if this move is in our top moves already
                    if any(tm['Move'] == move_uci for tm in top_moves):
                        continue

                    # Quick evaluation of this move
                    temp_board = board.copy()
                    temp_board.push(chess_move)

                    # Simple heuristic blunder detection
                    if board.is_capture(chess_move):
                        captured_piece = board.piece_at(chess_move.to_square)
                        our_piece = board.piece_at(chess_move.from_square)

                        if captured_piece and our_piece:
                            # Check if it's a bad trade
                            material_diff = self.get_piece_value(captured_piece.piece_type) - self.get_piece_value(
                                our_piece.piece_type)

                            # Bad trade = potential blunder
                            if -3 <= material_diff <= -1:  # Losing 1-3 points of material
                                blunder_candidates.append({
                                    'move': move_uci,
                                    'eval_loss': abs(material_diff) * 100,  # Convert to centipawns
                                    'rank': 99,
                                    'type': 'bad_trade'
                                })
                                break

            # Select from blunder candidates
            if blunder_candidates:
                # Prefer milder blunders (closer to min loss)
                blunder_candidates.sort(key=lambda x: x['eval_loss'])

                # Weight selection toward smaller blunders (more realistic)
                if len(blunder_candidates) == 1:
                    selected = blunder_candidates[0]
                elif len(blunder_candidates) == 2:
                    weights = [0.7, 0.3]
                    selected = random.choices(blunder_candidates, weights=weights)[0]
                else:
                    # More weight on smaller blunders
                    weights = [0.5, 0.3, 0.2] + [0.1] * (len(blunder_candidates) - 3)
                    weights = weights[:len(blunder_candidates)]
                    selected = random.choices(blunder_candidates, weights=weights)[0]

                print(f"💥 BLUNDER SELECTED: {selected['move']} (loses {selected['eval_loss']}cp)")

                # Log this blunder
                self.game_blunder_log.append({
                    'move_number': move_count,
                    'move': selected['move'],
                    'eval_loss': selected['eval_loss'],
                    'type': selected.get('type', 'positional')
                })

                return selected['move']

            # Fallback: just pick the 3rd or 4th best move
            print("🎲 Fallback blunder: selecting lower-ranked move")
            fallback_index = min(3, len(top_moves) - 1)
            selected_move = top_moves[fallback_index]['Move']
            eval_loss = best_eval - top_moves[fallback_index].get('Centipawn', 0)

            print(f"💥 FALLBACK BLUNDER: {selected_move} (loses ~{eval_loss}cp)")

            self.game_blunder_log.append({
                'move_number': move_count,
                'move': selected_move,
                'eval_loss': eval_loss,
                'type': 'fallback'
            })

            return selected_move

        except Exception as e:
            print(f"Error selecting blunder move: {e}")
            # Safe fallback
            return stockfish.get_best_move()

    def update_blunder_state(self, move_count):
        """
        Update blunder-related state after making a blunder
        """
        self.blunders_this_game += 1
        self.last_blunder_move = move_count
        self.just_blundered = True
        self.consecutive_good_moves = 0
        self.our_confidence_level *= 0.5  # Confidence crash after blunder

        print(f"📊 BLUNDER #{self.blunders_this_game}/{self.blunders_per_game} this game")

    def log_blunder_summary(self):
        """
        Log summary of blunders made this game
        """
        if self.game_blunder_log:
            print("\n🎯 BLUNDER SUMMARY THIS GAME:")
            for i, blunder in enumerate(self.game_blunder_log, 1):
                print(f"  Blunder {i}: Move {blunder['move_number']} - {blunder['move']} "
                      f"(lost {blunder['eval_loss']}cp, type: {blunder['type']})")
            avg_loss = sum(b['eval_loss'] for b in self.game_blunder_log) / len(self.game_blunder_log)
            print(f"  Average blunder severity: {avg_loss:.0f}cp")
        else:
            print("🎯 No blunders made this game!")

    def is_opening_phase(self, move_count):
        """Check if we're still in opening phase"""
        return move_count < 20  # Extended opening phase

    def is_early_opening(self, move_count):
        """First few moves should be lightning fast"""
        return move_count < 10  # Extended early opening

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
        if move_count > 12:  # Only in opening
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
                if target_square in ['e4', 'e5', 'd4', 'd5', 'c4', 'c5']:
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
        """Enhanced recapture detection"""
        if not self.last_opponent_move:
            return False

        try:
            last_move = chess.Move.from_uci(self.last_opponent_move)
            current_move = chess.Move.from_uci(move)

            # If opponent captured and we're capturing back on same square
            if board.is_capture(last_move) and current_move.to_square == last_move.to_square:
                # Even more obvious if it's an equal or favorable trade
                captured_piece = board.piece_at(current_move.to_square)
                our_piece = board.piece_at(current_move.from_square)

                if captured_piece and our_piece:
                    # Equal or better trade = very obvious
                    if self.get_piece_value(captured_piece.piece_type) >= self.get_piece_value(our_piece.piece_type):
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
        """Analyze position complexity and return thinking time modifier"""
        try:
            complexity_score = 1.0

            # Piece count affects complexity
            piece_count = len(board.piece_map())
            if piece_count > 20 and move_count > 10:
                complexity_score *= 1.1
            elif piece_count < 10:  # Endgame
                complexity_score *= 1.4  # Endgames need precision

            # Tactical indicators
            if board.is_check():
                complexity_score *= 1.3

            # Lots of captures available = more tactical
            legal_moves = list(board.legal_moves)
            capture_moves = [m for m in legal_moves if board.is_capture(m)]
            if len(capture_moves) > 3:
                complexity_score *= 1.2

            # Opening simplification - NEW: Much faster opening play
            if move_count < 20:
                complexity_score *= 0.4  # Opening moves much faster (was 0.5)

            return max(0.2, min(2.0, complexity_score))

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
                if eval_loss > 200 and not self.force_blunder_this_move:
                    self.just_blundered = True
                    self.consecutive_good_moves = 0
                    self.our_confidence_level *= 0.7  # Confidence drops after blunder
                    self.recent_blunder_moves.append(len(board.move_stack))
                else:
                    # Don't count intentional blunders as "just_blundered"
                    if not self.force_blunder_this_move:
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
        """Add psychological factors to thinking time"""
        psychological_time = base_time

        # If we just blundered, we're more careful
        if self.just_blundered:
            psychological_time *= random.uniform(1.5, 2.0)
            print("😰 Post-blunder caution")

        # If we're winning big, we might play faster (overconfidence)
        if self.we_are_winning_big:
            psychological_time *= random.uniform(0.6, 0.8)
            print("😎 Winning confidence boost")

        # If we're in the zone, smooth rhythm
        if self.consecutive_good_moves > 5:
            psychological_time *= random.uniform(0.7, 0.9)
            print("🎯 In the zone!")

        # Complex positions make us think longer
        if self.complex_position_streak > 2:
            psychological_time *= random.uniform(1.3, 1.7)
            print("🤯 Complex position fatigue")

        # Apply confidence level
        psychological_time *= (2.0 - self.our_confidence_level)

        return psychological_time

    def simulate_mouse_hesitation(self, move):
        """Simulate human mouse movement patterns"""
        if not self.enable_human_delays:
            return

        start_pos, end_pos = self.get_move_pos(move)

        # Sometimes hover over the piece before grabbing
        if random.random() < 0.3:
            hover_x = start_pos[0] + random.randint(-5, 5)
            hover_y = start_pos[1] + random.randint(-5, 5)
            pyautogui.moveTo(hover_x, hover_y, duration=0.1)
            time.sleep(random.uniform(0.05, 0.15))

        # Occasionally almost grab wrong piece
        if random.random() < 0.05 and not self.is_forced_move(chess.Board()):
            wrong_x = start_pos[0] + random.choice([-50, 50])
            wrong_y = start_pos[1] + random.choice([-50, 50])
            pyautogui.moveTo(wrong_x, wrong_y, duration=0.2)
            time.sleep(0.1)
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

    def select_human_like_move(self, stockfish, board, move_count):
        """
        Enhanced human-like move selection with realistic blind spots AND blunder system
        """
        if not self.human_accuracy:
            return stockfish.get_best_move()

        try:
            # Update psychological state
            self.update_game_psychology(board, stockfish)

            # NEW: Check if we should force a blunder this move
            if self.should_force_blunder_now(board, move_count, stockfish):
                self.force_blunder_this_move = True
                blunder_move = self.select_blunder_move(stockfish, board, move_count)
                self.update_blunder_state(move_count)
                self.force_blunder_this_move = False
                return blunder_move

            # Get multiple top moves
            top_moves = stockfish.get_top_moves(10)  # Get even more for realistic selection
            if not top_moves or len(top_moves) == 0:
                return stockfish.get_best_move()

            if len(top_moves) == 1:
                return top_moves[0]['Move']

            best_move = top_moves[0]
            best_eval = best_move.get('Centipawn', 0)
            best_move_uci = best_move['Move']

            # CRITICAL POSITIONS - always play best
            # 1. Mate in 1-2 (humans rarely miss mate in 1)
            if 'Mate' in best_move and best_move.get('Mate', 0) <= 2:
                print("🚨 Mate in 1-2 - playing best move")
                return best_move_uci

            # 2. Free queen (but might miss free rook sometimes)
            if self.is_free_piece_capture(board, best_move_uci):
                captured_value = self.get_material_gain(board, best_move_uci)
                if captured_value >= 9:  # Queen
                    print("👑 Free queen - playing best move")
                    return best_move_uci
                elif captured_value >= 5 and not self.we_are_winning_big:  # Rook, but not if overconfident
                    print("🏰 Free rook - playing best move")
                    return best_move_uci

            # 3. Very early opening
            if move_count < 3:
                print("📚 Very early opening - playing best move")
                return best_move_uci

            # 4. Only legal move
            if len(list(board.legal_moves)) == 1:
                print("🎯 Only one legal move")
                return best_move_uci

            # NEW: Check if we should realistically miss the best move
            if self.should_miss_this_move(board, best_move_uci, stockfish):
                print(f"👁️ Realistically missing: {best_move_uci}")
                # Remove best move from candidates
                top_moves = top_moves[1:]
                if top_moves:
                    best_move = top_moves[0]
                    best_eval = best_move.get('Centipawn', 0)
                    best_move_uci = best_move['Move']

            # Build candidate list with psychological factors
            candidate_moves = [best_move]

            # Dynamic threshold based on game state
            base_threshold = min(50, self.accuracy_threshold)

            # Adjust threshold based on psychology
            if self.just_blundered:
                base_threshold *= 0.6  # More careful after blunder
            elif self.we_are_winning_big:
                base_threshold *= 1.4  # More sloppy when winning big
            elif self.consecutive_good_moves > 5:
                base_threshold *= 0.8  # Playing well, stay focused

            for i, move in enumerate(top_moves[1:], 1):
                move_eval = move.get('Centipawn', 0)
                eval_diff = abs(best_eval - move_eval)

                # Skip moves we'd realistically miss
                if self.should_miss_this_move(board, move['Move'], stockfish):
                    continue

                # Game phase thresholds
                current_threshold = base_threshold
                if move_count < 10:  # Opening
                    current_threshold = max(30, base_threshold * 0.7)
                elif move_count > 50:  # Endgame
                    current_threshold = max(20, base_threshold * 0.5)
                elif board.is_check():  # Under attack
                    current_threshold = max(15, base_threshold * 0.4)

                if eval_diff <= current_threshold:
                    candidate_moves.append(move)
                    print(f"🎲 Candidate {len(candidate_moves)}: {move['Move']} (-{eval_diff}cp)")
                elif eval_diff > current_threshold * 2:
                    break  # Too much worse

            # Selection with psychological bias
            if len(candidate_moves) == 1:
                selected_move = candidate_moves[0]['Move']
                print(f"⭐ Only viable move: {selected_move}")
            else:
                # Bias toward attacking moves when winning
                if self.we_are_winning_big:
                    # Prefer captures and checks
                    attack_bias = []
                    for move in candidate_moves:
                        bias = 1.0
                        try:
                            chess_move = chess.Move.from_uci(move['Move'])
                            if board.is_capture(chess_move):
                                bias = 2.0
                            temp_board = board.copy()
                            temp_board.push(chess_move)
                            if temp_board.is_check():
                                bias *= 1.5
                        except:
                            pass
                        attack_bias.append(bias)

                    # Normalize weights
                    total_bias = sum(attack_bias)
                    weights = [b / total_bias for b in attack_bias]
                else:
                    # Normal selection weights
                    if len(candidate_moves) == 2:
                        weights = [0.75, 0.25]
                    elif len(candidate_moves) == 3:
                        weights = [0.60, 0.25, 0.15]
                    else:
                        first_weight = 0.50
                        remaining = 0.50
                        other_weights = [remaining / (len(candidate_moves) - 1)] * (len(candidate_moves) - 1)
                        weights = [first_weight] + other_weights

                selected_move = random.choices(candidate_moves, weights=weights)[0]['Move']

                # Log selection
                selected_index = next(i for i, m in enumerate(candidate_moves) if m['Move'] == selected_move)
                if selected_index == 0:
                    print(f"⭐ Selected best available: {selected_move}")
                else:
                    eval_diff = abs(candidate_moves[0].get('Centipawn', 0) -
                                    candidate_moves[selected_index].get('Centipawn', 0))
                    print(f"🎯 Human choice: {selected_move} (-{eval_diff}cp)")

                    # Track if this was a significant inaccuracy
                    if eval_diff > 100:
                        print(f"😅 Slight inaccuracy!")
                    if eval_diff > 200:
                        print(f"😬 That wasn't best...")

            return selected_move

        except Exception as e:
            print(f"Error in human move selection: {e}")
            return stockfish.get_best_move()

    def calculate_thinking_time(self, board, move, move_count, stockfish=None):
        """
        Enhanced human-like thinking with psychological patterns
        """
        if not self.enable_human_delays:
            return 0.1

        # NEW: Blunder thinking patterns
        if self.force_blunder_this_move:
            # Blunders often happen when humans think they're calculating but miss something
            blunder_time = random.uniform(2.0, 8.0)  # Longer think for blunders (ironic!)
            print(f"🤯 Blunder thinking time: {blunder_time:.1f}s (calculated but missed something)")
            return blunder_time

        # ============ INSTANT REACTIONS (0.05-0.2s) ============

        # NEW: Pre-move situations (instant mouse click)
        if move and self.last_opponent_move:
            # Obvious recapture we were expecting
            if self.is_obvious_recapture(board, move):
                thinking_time = random.uniform(0.05, 0.15)
                print(f"⚡ Pre-move recapture: {thinking_time:.1f}s")
                return thinking_time

        # ============ LIGHTNING FAST MOVES (0.1-0.3s) ============

        # 1. EARLY OPENING - Super fast first moves
        if self.is_early_opening(move_count):
            thinking_time = random.uniform(0.05, 0.25)
            print(f"⚡ Early opening: {thinking_time:.1f}s")
            return thinking_time

        # 2. PIECE DEVELOPMENT - Fast in opening
        if self.is_piece_development_move(board, move, move_count):
            thinking_time = random.uniform(0.1, 0.35)
            print(f"🏃 Piece development: {thinking_time:.1f}s")
            return thinking_time

        # 3. CASTLING - Usually quick decision
        if move and self.is_obvious_castling(board, move):
            thinking_time = random.uniform(0.1, 0.5)
            print(f"🏰 Castling: {thinking_time:.1f}s")
            return thinking_time

        # 4. EN PASSANT - Obvious when available
        if move and self.is_obvious_en_passant(board, move):
            thinking_time = random.uniform(0.1, 0.4)
            print(f"👻 En passant: {thinking_time:.1f}s")
            return thinking_time

        # 5. MATE IN ONE - Quick but with slight double-check
        if move and self.is_mate_in_one(board, move):
            # Humans sometimes pause to savor the moment
            if self.we_are_winning_big:
                thinking_time = random.uniform(0.8, 1.5)  # Savoring the win
                print(f"☠️ Mate in one (savoring): {thinking_time:.1f}s")
            else:
                thinking_time = random.uniform(0.3, 0.7)  # Quick double-check
                print(f"☠️ Mate in one: {thinking_time:.1f}s")
            return thinking_time

        # 6. PROMOTION TO QUEEN - Usually obvious
        if move and self.is_promotion_to_queen(move):
            thinking_time = random.uniform(0.1, 0.6)
            print(f"👑 Queen promotion: {thinking_time:.1f}s")
            return thinking_time

        # ============ VERY FAST MOVES (0.1-0.5s) ============

        # 7. FORCED MOVES - Only one legal option
        if self.is_forced_move(board):
            thinking_time = random.uniform(0.1, 0.25)
            print(f"🎯 Forced move: {thinking_time:.1f}s")
            return thinking_time

        # 8. CHECK ESCAPES - Limited options
        if self.is_check_escape_only(board):
            # More panic if we're already losing
            if self.previous_evaluation < -300:
                thinking_time = random.uniform(0.4, 1.2)  # Panic mode
            else:
                thinking_time = random.uniform(0.2, 0.8)
            print(f"🛡️ Check escape: {thinking_time:.1f}s")
            return thinking_time

        # 9. HANGING PIECE CAPTURES - Super obvious
        if move and self.is_piece_hanging_obviously(board, move):
            # But double-check if opponent just blundered
            if self.opponent_just_blundered(board, stockfish):
                thinking_time = random.uniform(0.8, 1.5)  # "Is this real?"
                print(f"🎁 Hanging piece (suspicious): {thinking_time:.1f}s")
            else:
                thinking_time = random.uniform(0.1, 0.3)
                print(f"🎁 Hanging piece: {thinking_time:.1f}s")
            return thinking_time

        # 10. FORK FOLLOW-UPS - Instant after successful forks
        if move and self.is_fork_followup(board, move):
            thinking_time = random.uniform(0.1, 0.5)
            print(f"🍴 Fork follow-up: {thinking_time:.1f}s")
            return thinking_time

        # 11. OBVIOUS RECAPTURES - Already handled above for pre-moves
        if move and self.is_obvious_recapture(board, move):
            thinking_time = random.uniform(0.1, 0.5)
            print(f"⚡ Obvious recapture: {thinking_time:.1f}s")
            return thinking_time

        # ============ FAST MOVES (0.2-0.8s) ============

        # 12. FREE PIECE CAPTURES
        if move and self.is_free_piece_capture(board, move):
            thinking_time = random.uniform(0.2, 0.6)
            print(f"🆓 Free piece: {thinking_time:.1f}s")
            return thinking_time

        # 13. HUGE MATERIAL GAINS
        if move and stockfish and self.is_huge_material_gain(board, move, stockfish):
            thinking_time = random.uniform(0.3, 0.9)
            print(f"💰 Huge material gain: {thinking_time:.1f}s")
            return thinking_time

        # ============ BLUNDER REACTIONS ============

        # 14. OPPONENT BLUNDERED - Humans double-check gifts
        if stockfish and self.opponent_just_blundered(board, stockfish):
            # Reaction based on size of blunder
            eval_swing = abs(self.previous_evaluation - stockfish.get_evaluation().get('value', 0))
            if eval_swing > 500:  # Huge blunder
                thinking_time = random.uniform(2.0, 4.0)  # "Wait, what??"
                print(f"😲 Opponent huge blunder - double checking: {thinking_time:.1f}s")
            else:
                thinking_time = random.uniform(1.0, 2.5)
                print(f"🤔 Opponent blundered - checking: {thinking_time:.1f}s")
            return thinking_time

        # ============ NORMAL COMPLEXITY ANALYSIS ============

        base_time = random.uniform(self.min_thinking_time, self.max_thinking_time * 0.8)

        if stockfish:
            try:
                candidate_count = self.count_candidate_moves(board, stockfish)

                # Time based on number of good moves
                if candidate_count <= 1:
                    thinking_time = random.uniform(0.4, 1.2)
                    print(f"🎯 Only move: {thinking_time:.1f}s")
                elif candidate_count <= 2:
                    thinking_time = random.uniform(0.8, 2.5)
                    print(f"🤏 Few options ({candidate_count}): {thinking_time:.1f}s")
                elif candidate_count <= 4:
                    thinking_time = random.uniform(1.5, 4.0)
                    print(f"🤔 Several options ({candidate_count}): {thinking_time:.1f}s")
                else:
                    thinking_time = random.uniform(2.5, 6.0)
                    print(f"🧠 Many options ({candidate_count}): {thinking_time:.1f}s")

                # NEW: Add psychological factors
                thinking_time = self.add_psychological_delay(thinking_time)

            except Exception as e:
                print(f"Error in candidate analysis: {e}")
                thinking_time = base_time
        else:
            thinking_time = base_time

        # Position complexity modifier
        try:
            complexity_factor = self.analyze_position_complexity(board, move_count)
            thinking_time *= complexity_factor
        except:
            pass

        # Game phase adjustments
        if move_count > 40:  # Endgame
            thinking_time *= 1.2
            # Extra time in critical endgames
            if len(board.piece_map()) < 8:
                thinking_time *= 1.3
        elif self.is_opening_phase(move_count):
            thinking_time *= 0.6

        # NEW: Rhythm variations
        # Sometimes humans get into a rhythm and play consistently
        if self.consecutive_good_moves > 3:
            # Smooth, consistent timing
            rhythm_factor = random.uniform(0.9, 1.1)
        else:
            # More erratic timing
            rhythm_factor = random.uniform(0.7, 1.3)
        thinking_time *= rhythm_factor

        # Final bounds with more variation
        min_time = 0.1
        max_time = self.max_thinking_time * 1.2

        # But allow occasional long thinks
        if random.random() < 0.05:  # 5% chance of deep think
            max_time *= 2
            print("💭 Deep calculation...")

        thinking_time = max(min_time, min(thinking_time, max_time))

        return round(thinking_time, 1)

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

        # NEW: More natural mouse movement
        movement_duration = random.uniform(0.1, 0.3) if self.enable_human_delays else 0
        pyautogui.moveTo(start_pos[0], start_pos[1], duration=movement_duration)
        time.sleep(self.mouse_latency)

        print("Clicking and holding at start position")
        pyautogui.mouseDown()

        # NEW: Variable hold time (humans don't always grab instantly)
        hold_time = random.uniform(0.1, 0.3) if self.enable_human_delays else 0.2
        time.sleep(hold_time)

        print(f"Dragging to end position: {end_pos}")

        # NEW: Variable drag speed based on move type
        if self.enable_human_delays:
            if self.is_obvious_recapture(chess.Board(), move):
                drag_duration = random.uniform(0.2, 0.4)  # Fast recapture
            elif self.just_blundered:
                drag_duration = random.uniform(0.6, 0.9)  # Careful after blunder
            else:
                drag_duration = random.uniform(0.4, 0.7)  # Normal speed
        else:
            drag_duration = 0.5

        pyautogui.moveTo(end_pos[0], end_pos[1], duration=drag_duration)

        # NEW: Sometimes slight overshoot
        if self.enable_human_delays and random.random() < 0.1:
            overshoot_x = end_pos[0] + random.randint(-10, 10)
            overshoot_y = end_pos[1] + random.randint(-10, 10)
            pyautogui.moveTo(overshoot_x, overshoot_y, duration=0.1)
            time.sleep(0.05)
            pyautogui.moveTo(end_pos[0], end_pos[1], duration=0.1)

        time.sleep(0.2)

        print("Releasing mouse at end position")
        pyautogui.mouseUp()

        # NEW: Variable post-move delay
        if self.enable_human_delays:
            if self.is_mate_in_one(chess.Board(), move):
                post_delay = random.uniform(0.5, 1.0)  # Pause after delivering mate
            else:
                post_delay = random.uniform(0.2, 0.4)
        else:
            post_delay = 0.3
        time.sleep(post_delay)
        print("Move completed")

        # Handle promotion
        if len(move) == 5:
            print(f"Promotion detected: {move[4]}")
            time.sleep(0.5)
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

            # NEW: Initialize blunder system for this game
            self.blunders_this_game = 0
            self.last_blunder_move = 0
            self.force_blunder_this_move = False
            self.game_blunder_log = []

            if self.enable_blunders:
                print(
                    f"🎯 Blunder system enabled: {self.blunders_per_game} blunders/game, {self.blunder_severity}cp severity")

            # Initialize grabber
            print("Initializing grabber...")
            if self.website == "chesscom":
                self.grabber = ChesscomGrabber(self.chrome_url, self.chrome_session_id)
            else:
                self.grabber = LichessGrabber(self.chrome_url, self.chrome_session_id)
            print("Grabber initialized successfully")

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
                    print(f"🎯 Target accuracy: 80-90% (will make human-like mistakes!)")
                    print(f"🧠 Psychological modeling: ON")
                    if self.enable_blunders:
                        print(f"💥 Deliberate blundering: {self.blunders_per_game} per game")
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

                        # Log blunder summary
                        self.log_blunder_summary()

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

                        # Log blunder summary
                        self.log_blunder_summary()

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

                    # Log blunder summary
                    self.log_blunder_summary()

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
