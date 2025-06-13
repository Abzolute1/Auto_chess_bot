import random
import chess
import math
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque


class PatternType(Enum):
    """Types of tactical patterns in chess"""
    PIN = "pin"
    FORK = "fork"
    SKEWER = "skewer"
    DISCOVERED_ATTACK = "discovered_attack"
    DOUBLE_ATTACK = "double_attack"
    BACK_RANK_MATE = "back_rank_mate"
    HANGING_PIECE = "hanging_piece"
    TRAPPED_PIECE = "trapped_piece"
    OVERLOADED_PIECE = "overloaded_piece"
    DEFLECTION = "deflection"
    DECOY = "decoy"
    ZWISCHENZUG = "zwischenzug"


class TimeManagementStyle(Enum):
    """Different time management personalities"""
    STEADY = "steady"  # Consistent time usage
    FAST_THEN_THINK = "fast_then_think"  # Blitz out moves then think
    TIME_TROUBLE_ADDICT = "time_trouble_addict"  # Always gets low on time
    CAUTIOUS = "cautious"  # Saves time for endgame
    IMPULSIVE = "impulsive"  # Moves quickly when confident


class GameContext(Enum):
    """Game importance context"""
    CASUAL = "casual"
    MUST_WIN = "must_win"
    MUST_DRAW = "must_draw"
    TOURNAMENT = "tournament"
    LEARNING = "learning"


@dataclass
class PlayingStyle:
    """Defines a consistent playing personality"""
    aggression: float = 0.5  # 0-1, affects piece sacrifices, attacks
    positional_understanding: float = 0.5  # vs tactical preference
    risk_tolerance: float = 0.5  # willingness to enter complications
    endgame_preference: float = 0.5  # seeks/avoids endgames
    calculation_depth: float = 0.5  # how deeply they typically calculate
    intuition_reliance: float = 0.5  # intuition vs calculation
    time_management: TimeManagementStyle = TimeManagementStyle.STEADY
    favorite_pieces: List[chess.PieceType] = field(default_factory=list)
    preferred_structures: List[str] = field(default_factory=list)  # e.g., "isolated_pawn", "space_advantage"

    def __post_init__(self):
        # Initialize favorite pieces based on style
        if not self.favorite_pieces:
            if self.aggression > 0.7:
                self.favorite_pieces = [chess.QUEEN, chess.KNIGHT]
            elif self.positional_understanding > 0.7:
                self.favorite_pieces = [chess.BISHOP, chess.ROOK]
            else:
                self.favorite_pieces = [chess.KNIGHT, chess.BISHOP]


@dataclass
class EmotionalMemory:
    """Stores emotional associations with positions and patterns"""
    traumatic_patterns: Dict[str, float] = field(default_factory=dict)  # pattern -> trauma_level
    successful_patterns: Dict[str, float] = field(default_factory=dict)  # pattern -> confidence
    opening_confidence: Dict[str, float] = field(default_factory=dict)  # opening_key -> confidence
    piece_attachment: Dict[chess.PieceType, float] = field(default_factory=dict)  # piece -> attachment
    position_memories: List[Dict] = field(default_factory=list)  # Recent memorable positions
    revenge_targets: Set[str] = field(default_factory=set)  # Patterns to "get back at"

    def decay_memories(self, decay_rate: float = 0.95):
        """Gradually forget emotional associations"""
        for pattern in self.traumatic_patterns:
            self.traumatic_patterns[pattern] *= decay_rate
        for pattern in self.successful_patterns:
            self.successful_patterns[pattern] *= decay_rate
        for opening in self.opening_confidence:
            self.opening_confidence[opening] = 0.5 + (self.opening_confidence[opening] - 0.5) * decay_rate


class PatternRecognitionSystem:
    """Manages pattern blindness and recognition failures"""

    def __init__(self):
        self.pattern_blindness: Dict[PatternType, int] = defaultdict(int)  # Moves remaining blind
        self.recently_missed: deque = deque(maxlen=10)  # Recent missed patterns
        self.fatigue_level: float = 0.0
        self.concentration_level: float = 1.0
        self.pattern_complexity_threshold: float = 0.5  # Ability to see complex patterns

    def induce_blindness(self, pattern: PatternType, severity: int = 3):
        """Make player temporarily blind to a pattern type"""
        self.pattern_blindness[pattern] = max(self.pattern_blindness[pattern], severity)
        self.recently_missed.append(pattern)

    def update_blindness(self):
        """Reduce pattern blindness over time"""
        for pattern in list(self.pattern_blindness.keys()):
            if self.pattern_blindness[pattern] > 0:
                self.pattern_blindness[pattern] -= 1
                if self.pattern_blindness[pattern] == 0:
                    del self.pattern_blindness[pattern]

    def can_see_pattern(self, pattern: PatternType, complexity: float = 0.5) -> bool:
        """Check if player can currently recognize a pattern"""
        # Blind to this pattern type?
        if self.pattern_blindness.get(pattern, 0) > 0:
            return random.random() < 0.2  # 20% chance to see it anyway

        # Fatigue affects pattern recognition
        if self.fatigue_level > 0.7 and complexity > 0.6:
            return random.random() < 0.5

        # Complex patterns harder to see
        if complexity > self.pattern_complexity_threshold:
            success_chance = 1.0 - (complexity - self.pattern_complexity_threshold)
            return random.random() < success_chance * self.concentration_level

        return random.random() < 0.95 * self.concentration_level


class EndgameKnowledge:
    """Models endgame knowledge based on rating"""

    # Endgame types and required rating to play well
    ENDGAME_REQUIREMENTS = {
        "K+P vs K": 800,
        "K+Q vs K": 600,
        "K+R vs K": 1000,
        "K+2B vs K": 1600,
        "K+B+N vs K": 2000,
        "R+P vs R": 1400,
        "R+2P vs R+P": 1600,
        "Q+P vs Q": 1800,
        "opposite_bishops": 1400,
        "knight_endings": 1200,
        "pawn_endings": 1000,
        "rook_endings": 1600,
        "queen_endings": 1800,
    }

    def __init__(self, rating: int):
        self.rating = rating
        self.knowledge_matrix = self._build_knowledge_matrix()
        self.favorite_endgames = []
        self.weak_endgames = []
        self._categorize_endgames()

    def _build_knowledge_matrix(self) -> Dict[str, float]:
        """Build knowledge levels for each endgame type"""
        matrix = { }
        for endgame, required_rating in self.ENDGAME_REQUIREMENTS.items():
            if self.rating >= required_rating + 200:
                # Strong knowledge
                matrix[endgame] = random.uniform(0.85, 0.95)
            elif self.rating >= required_rating:
                # Adequate knowledge
                matrix[endgame] = random.uniform(0.65, 0.85)
            elif self.rating >= required_rating - 200:
                # Shaky knowledge
                matrix[endgame] = random.uniform(0.35, 0.65)
            else:
                # Poor knowledge
                matrix[endgame] = random.uniform(0.1, 0.35)

        # Add some randomness for personality
        variance_endgame = random.choice(list(matrix.keys()))
        matrix[variance_endgame] = min(1.0, max(0.1, matrix[variance_endgame] + random.uniform(-0.3, 0.3)))

        return matrix

    def _categorize_endgames(self):
        """Identify strong and weak endgames"""
        for endgame, knowledge in self.knowledge_matrix.items():
            if knowledge > 0.8:
                self.favorite_endgames.append(endgame)
            elif knowledge < 0.4:
                self.weak_endgames.append(endgame)

    def get_endgame_confidence(self, position_features: Dict[str, bool]) -> float:
        """Get confidence level for current endgame type"""
        confidence = 0.7  # Base confidence

        for feature, present in position_features.items():
            if present and feature in self.knowledge_matrix:
                confidence *= self.knowledge_matrix[feature]

        return confidence


class ChessHabits:
    """Models consistent behavioral patterns"""

    def __init__(self, rating: int, style: PlayingStyle):
        self.rating = rating
        self.style = style
        self.habits = self._generate_habits()
        self.superstitions = self._generate_superstitions()
        self.thinking_triggers = self._generate_thinking_triggers()
        self.move_order_preference = self._generate_move_preferences()

    def _generate_habits(self) -> Dict[str, float]:
        """Generate personal habits based on rating and style"""
        habits = {
            "always_castle_early": random.random() < 0.7,
            "fianchetto_lover": random.random() < 0.3,
            "trade_queens_when_ahead": random.random() < 0.6,
            "avoid_trades_when_behind": random.random() < 0.7,
            "push_h_pawn_attacker": self.style.aggression > 0.7 and random.random() < 0.5,
            "time_trouble_recovery": self.style.time_management == TimeManagementStyle.STEADY,
            "premove_recaptures": self.rating > 1500 and random.random() < 0.8,
            "think_on_opponent_time": self.rating > 1200 and random.random() < 0.7,
        }
        return habits

    def _generate_superstitions(self) -> List[str]:
        """Generate chess superstitions/preferences"""
        all_superstitions = [
            "avoid_f7_weakness",
            "knights_before_bishops",
            "develop_kingside_first",
            "hate_doubled_pawns",
            "love_bishop_pair",
            "avoid_early_queen",
            "center_pawns_first",
            "rooks_love_seventh",
        ]

        # Select 2-4 superstitions
        num_superstitions = random.randint(2, 4)
        return random.sample(all_superstitions, num_superstitions)

    def _generate_thinking_triggers(self) -> Dict[str, float]:
        """Situations that trigger longer thinking"""
        triggers = {
            "opponent_queen_move": 1.5,
            "pawn_break": 1.3,
            "piece_exchange": 1.2,
            "check": 1.4,
            "opponent_sacrifice": 2.0,
            "entering_endgame": 1.6,
            "time_advantage": 0.8,  # Think less when opponent is low
            "unfamiliar_position": 1.8,
        }

        # Adjust based on style
        if self.style.calculation_depth > 0.7:
            triggers = { k: v * 1.2 for k, v in triggers.items() }
        elif self.style.intuition_reliance > 0.7:
            triggers = { k: v * 0.8 for k, v in triggers.items() }

        return triggers

    def _generate_move_preferences(self) -> List[str]:
        """Order in which to consider move types"""
        base_order = ["checks", "captures", "threats", "development", "pawn_moves", "quiet_moves"]

        # Shuffle based on style
        if self.style.aggression > 0.7:
            # Aggressive players check forcing moves first
            return base_order
        elif self.style.positional_understanding > 0.7:
            # Positional players consider quiet moves earlier
            return ["development", "quiet_moves", "pawn_moves", "checks", "captures", "threats"]
        else:
            # Random personality
            random.shuffle(base_order)
            return base_order


class RatingSpecificWeaknesses:
    """Models rating-appropriate blind spots and weaknesses"""

    RATING_WEAKNESSES = {
        (0, 1000): {
            "hanging_pieces": 0.3,
            "one_move_tactics": 0.4,
            "back_rank_mates": 0.5,
            "forks": 0.35,
            "pins": 0.45,
            "undefended_pieces": 0.4,
            "piece_coordination": 0.6,
        },
        (1000, 1400): {
            "hanging_pieces": 0.15,
            "backward_moves": 0.4,
            "quiet_tactics": 0.5,
            "long_diagonal_bishops": 0.35,
            "knight_forks": 0.25,
            "discovered_attacks": 0.4,
            "piece_coordination": 0.3,
            "weak_squares": 0.5,
        },
        (1400, 1800): {
            "positional_sacrifices": 0.6,
            "backward_moves": 0.25,
            "quiet_tactics": 0.3,
            "prophylaxis": 0.7,
            "pawn_breaks": 0.4,
            "piece_coordination": 0.2,
            "weak_squares": 0.3,
            "opposite_side_castling": 0.5,
        },
        (1800, 2200): {
            "deep_tactics": 0.4,
            "positional_sacrifices": 0.3,
            "prophylaxis": 0.4,
            "dynamic_compensation": 0.5,
            "fortress_positions": 0.6,
            "zugzwang": 0.5,
            "pawn_breaks": 0.2,
            "piece_coordination": 0.1,
        },
        (2200, 3000): {
            "computer_moves": 0.6,
            "deep_tactics": 0.2,
            "dynamic_compensation": 0.3,
            "fortress_positions": 0.3,
            "zugzwang": 0.2,
            "opposite_evaluations": 0.4,
            "long_term_sacrifices": 0.3,
        }
    }

    def __init__(self, rating: int):
        self.rating = rating
        self.weaknesses = self._get_weaknesses()
        self.improving_areas = []
        self.stubborn_weaknesses = []
        self._categorize_weaknesses()

    def _get_weaknesses(self) -> Dict[str, float]:
        """Get weakness levels for current rating"""
        for (min_rating, max_rating), weaknesses in self.RATING_WEAKNESSES.items():
            if min_rating <= self.rating < max_rating:
                # Add some individual variation
                personal_weaknesses = { }
                for weakness, base_level in weaknesses.items():
                    # ±20% personal variation
                    personal_level = base_level * random.uniform(0.8, 1.2)
                    personal_weaknesses[weakness] = min(1.0, max(0.0, personal_level))
                return personal_weaknesses

        # Default for very high ratings
        return { k: 0.1 for k in ["computer_moves", "deep_tactics", "long_term_sacrifices"] }

    def _categorize_weaknesses(self):
        """Identify improving and stubborn weak areas"""
        for weakness, level in self.weaknesses.items():
            if random.random() < 0.3:  # 30% chance to be improving
                self.improving_areas.append(weakness)
            elif level > 0.5 and random.random() < 0.2:  # 20% chance to be stubborn
                self.stubborn_weaknesses.append(weakness)

    def miss_due_to_weakness(self, move_features: Set[str]) -> bool:
        """Check if a move should be missed due to weaknesses"""
        for feature in move_features:
            if feature in self.weaknesses:
                miss_probability = self.weaknesses[feature]

                # Stubborn weaknesses are harder to overcome
                if feature in self.stubborn_weaknesses:
                    miss_probability *= 1.2

                # Improving areas have reduced miss rate
                if feature in self.improving_areas:
                    miss_probability *= 0.7

                if random.random() < miss_probability:
                    return True

        return False


class ConfidenceSystem:
    """Advanced confidence modeling affecting all decisions"""

    def __init__(self, base_confidence: float = 0.7):
        self.base_confidence = base_confidence
        self.current_confidence = base_confidence
        self.confidence_history = deque(maxlen=20)
        self.position_familiarity = 0.5
        self.opponent_strength_perception = 0.5
        self.momentum = 0.0
        self.comfort_zone = True

    def update_confidence(self, move_quality: float, position_complexity: float,
                          time_remaining: float, opponent_think_time: float):
        """Update confidence based on recent events"""
        # Move quality impact
        if move_quality > 0.8:
            confidence_boost = 0.05
        elif move_quality < 0.3:
            confidence_boost = -0.1
        else:
            confidence_boost = 0.02

        # Position complexity impact
        if position_complexity > 0.7:
            if self.comfort_zone:
                confidence_boost -= 0.03
        else:
            confidence_boost += 0.01

        # Time comparison impact
        if time_remaining > opponent_think_time * 1.5:
            confidence_boost += 0.02
        elif time_remaining < opponent_think_time * 0.5:
            confidence_boost -= 0.03

        # Update with momentum
        self.momentum = self.momentum * 0.8 + confidence_boost
        self.current_confidence += self.momentum

        # Bounds
        self.current_confidence = max(0.3, min(1.0, self.current_confidence))
        self.confidence_history.append(self.current_confidence)

    def get_confidence_multiplier(self) -> float:
        """Get multiplier for various decisions based on confidence"""
        if self.current_confidence > 0.8:
            return 1.2  # Overconfident
        elif self.current_confidence < 0.5:
            return 0.8  # Underconfident
        else:
            return 1.0

    def is_tilted(self) -> bool:
        """Check if player is tilted (rapid confidence drop)"""
        if len(self.confidence_history) < 5:
            return False

        recent_avg = sum(list(self.confidence_history)[-5:]) / 5
        older_avg = sum(list(self.confidence_history)[-10:-5]) / 5 if len(
            self.confidence_history) >= 10 else self.base_confidence

        return recent_avg < older_avg * 0.7


class HumanAccuracySystem:
    """
    Enterprise-level human chess simulation system.
    Combines all subsystems to create emergent human-like behavior.
    """

    def __init__(self, skill_level: int = 10, enable_errors: bool = True,
                 errors_per_game: Optional[Dict[str, int]] = None,
                 personality_profile: Optional[Dict] = None):

        # Convert skill level to approximate rating
        self.skill_level = skill_level
        self.rating = self._skill_to_rating(skill_level)
        self.enable_errors = enable_errors

        # Load or generate personality
        if personality_profile:
            self.personality = self._load_personality(personality_profile)
        else:
            self.personality = self._generate_personality()

        # Initialize all subsystems
        self.playing_style = self.personality['style']
        self.pattern_recognition = PatternRecognitionSystem()
        self.endgame_knowledge = EndgameKnowledge(self.rating)
        self.emotional_memory = EmotionalMemory()
        self.habits = ChessHabits(self.rating, self.playing_style)
        self.rating_weaknesses = RatingSpecificWeaknesses(self.rating)
        self.confidence = ConfidenceSystem(base_confidence=0.5 + self.rating / 4000)

        # Game context
        self.game_context = GameContext.CASUAL
        self.must_win_mode = False
        self.opponent_model = {
            'estimated_rating': self.rating,
            'style': 'unknown',
            'patterns_shown': [],
        }

        # Error system (enhanced from original)
        default_errors = self._rating_based_error_targets()
        self.base_errors_per_game = errors_per_game or default_errors
        self.errors_per_game = self._randomize_error_targets()

        # Enhanced error severity with rating adjustment
        base_severity = {
            'inaccuracy': (20, 80),
            'mistake': (80, 200),
            'blunder': (200, 800)
        }
        self.error_severity = self._adjust_severity_for_rating(base_severity)

        # Tracking
        self.errors_this_game = { 'inaccuracy': 0, 'mistake': 0, 'blunder': 0 }
        self.last_error_move = 0
        self.error_cooldown = self._rating_based_cooldown()
        self.game_error_log = []

        # Psychological state (enhanced)
        self.just_blundered = False
        self.consecutive_good_moves = 0
        self.we_are_winning_big = False
        self.opponent_time_pressure = False
        self.tilt_level = 0.0
        self.flow_state = False  # In the zone
        self.pressure_points = 0  # Accumulated pressure

        # Time management
        self.in_time_pressure = False
        self.severe_time_pressure = False
        self.critical_time_pressure = False
        self.moves_in_time_pressure = 0
        self.time_usage_pattern = []

        # Position tracking
        self.complex_position_streak = 0
        self.in_sharp_position = False
        self.position_tension = 0.0
        self.familiar_structure = False

        # Session tracking
        self.games_played_session = 0
        self.session_fatigue = 0.0
        self.learning_adjustments = { }

    def _skill_to_rating(self, skill_level: int) -> int:
        """Convert skill level (0-20) to approximate rating"""
        # Exponential scaling for more realistic distribution
        return int(600 + skill_level * 100 + (skill_level ** 2) * 10)

    def _generate_personality(self) -> Dict:
        """Generate a random but coherent personality"""
        # Generate base traits
        aggression = random.betavariate(2, 2)  # Beta distribution for realistic spread

        # Correlated traits
        if aggression > 0.7:
            risk_tolerance = aggression * random.uniform(0.8, 1.2)
            positional_understanding = random.uniform(0.3, 0.6)
            time_style = random.choice([TimeManagementStyle.FAST_THEN_THINK,
                                        TimeManagementStyle.IMPULSIVE])
        elif aggression < 0.3:
            risk_tolerance = aggression * random.uniform(0.8, 1.0)
            positional_understanding = random.uniform(0.6, 0.9)
            time_style = random.choice([TimeManagementStyle.STEADY,
                                        TimeManagementStyle.CAUTIOUS])
        else:
            risk_tolerance = random.betavariate(2, 2)
            positional_understanding = random.betavariate(2, 2)
            time_style = random.choice(list(TimeManagementStyle))

        style = PlayingStyle(
            aggression=aggression,
            positional_understanding=positional_understanding,
            risk_tolerance=risk_tolerance,
            endgame_preference=random.betavariate(2, 2),
            calculation_depth=random.betavariate(2, 2),
            intuition_reliance=random.betavariate(2, 2),
            time_management=time_style
        )

        return {
            'style': style,
            'name': self._generate_player_name(style),
            'quirks': self._generate_quirks(style),
        }

    def _generate_player_name(self, style: PlayingStyle) -> str:
        """Generate a descriptive name based on style"""
        adjectives = []

        if style.aggression > 0.7:
            adjectives.append("Aggressive")
        elif style.aggression < 0.3:
            adjectives.append("Solid")

        if style.positional_understanding > 0.7:
            adjectives.append("Positional")
        elif style.positional_understanding < 0.3:
            adjectives.append("Tactical")

        if style.risk_tolerance > 0.7:
            adjectives.append("Fearless")
        elif style.risk_tolerance < 0.3:
            adjectives.append("Cautious")

        noun = "Player"
        if style.time_management == TimeManagementStyle.TIME_TROUBLE_ADDICT:
            noun = "Time Scrambler"
        elif style.calculation_depth > 0.8:
            noun = "Calculator"
        elif style.intuition_reliance > 0.8:
            noun = "Intuitive"

        return f"{random.choice(adjectives)} {noun}"

    def _generate_quirks(self, style: PlayingStyle) -> List[str]:
        """Generate personality quirks"""
        all_quirks = [
            "plays_fast_in_won_positions",
            "thinks_long_on_recaptures",
            "hates_isolated_pawns",
            "loves_knight_outposts",
            "always_plays_Nf3_first",
            "never_underpromotes",
            "quick_in_theoretical_positions",
            "slow_starter",
            "endgame_specialist",
            "opening_expert",
            "time_pressure_fighter",
            "drawish_tendency",
        ]

        # Select 2-4 quirks that match personality
        num_quirks = random.randint(2, 4)
        selected = []

        for quirk in random.sample(all_quirks, len(all_quirks)):
            if len(selected) >= num_quirks:
                break

            # Check if quirk matches personality
            if quirk == "plays_fast_in_won_positions" and style.risk_tolerance > 0.6:
                selected.append(quirk)
            elif quirk == "endgame_specialist" and style.endgame_preference > 0.7:
                selected.append(quirk)
            elif quirk == "time_pressure_fighter" and style.time_management == TimeManagementStyle.TIME_TROUBLE_ADDICT:
                selected.append(quirk)
            elif random.random() < 0.3:  # 30% chance for any quirk
                selected.append(quirk)

        return selected

    def _load_personality(self, profile: Dict) -> Dict:
        """Load personality from profile"""
        # Convert dict to PlayingStyle
        style_data = profile.get('style', { })
        style = PlayingStyle(**style_data)

        return {
            'style': style,
            'name': profile.get('name', 'Custom Player'),
            'quirks': profile.get('quirks', []),
        }

    def _rating_based_error_targets(self) -> Dict[str, int]:
        """Get error targets based on rating"""
        if self.rating < 1000:
            return { 'inaccuracy': 6, 'mistake': 4, 'blunder': 3 }
        elif self.rating < 1400:
            return { 'inaccuracy': 5, 'mistake': 3, 'blunder': 2 }
        elif self.rating < 1800:
            return { 'inaccuracy': 4, 'mistake': 2, 'blunder': 1 }
        elif self.rating < 2200:
            return { 'inaccuracy': 3, 'mistake': 1, 'blunder': 0 }
        else:
            return { 'inaccuracy': 2, 'mistake': 1, 'blunder': 0 }

    def _adjust_severity_for_rating(self, base_severity: Dict) -> Dict:
        """Adjust error severity based on rating"""
        adjusted = { }

        # Higher rated players make subtler errors
        rating_factor = min(1.0, max(0.5, self.rating / 2000))

        for error_type, (min_loss, max_loss) in base_severity.items():
            if error_type == 'inaccuracy':
                # Inaccuracies stay similar across ratings
                adjusted[error_type] = (min_loss, max_loss)
            else:
                # Mistakes and blunders are less severe at higher ratings
                adjusted[error_type] = (
                    int(min_loss * rating_factor),
                    int(max_loss * rating_factor)
                )

        return adjusted

    def _rating_based_cooldown(self) -> int:
        """Get error cooldown based on rating"""
        if self.rating < 1200:
            return 3  # Can make errors more frequently
        elif self.rating < 1800:
            return 5
        else:
            return 7  # Higher rated players learn from mistakes

    def _randomize_error_targets(self) -> Dict[str, int]:
        """Add variance to error targets with personality influence"""
        randomized = { }

        for error_type, base_count in self.base_errors_per_game.items():
            # Personality affects error likelihood
            if error_type == 'blunder' and self.playing_style.risk_tolerance > 0.7:
                # Risk takers blunder more
                base_count = int(base_count * 1.3)
            elif error_type == 'inaccuracy' and self.playing_style.calculation_depth < 0.3:
                # Poor calculators make more inaccuracies
                base_count = int(base_count * 1.2)

            # Add variance
            if random.random() < 0.2:
                reduction = random.choices([0, 1, 2], weights=[0.5, 0.4, 0.1])[0]
                randomized[error_type] = max(0, base_count - reduction)
            else:
                randomized[error_type] = base_count

        # Personality-based perfect game chance
        perfect_game_chance = 0.02
        if self.playing_style.risk_tolerance < 0.3:
            perfect_game_chance *= 2
        if self.playing_style.calculation_depth > 0.8:
            perfect_game_chance *= 1.5

        if random.random() < perfect_game_chance:
            print(f"🌟 {self.personality['name']} is aiming for a perfect game!")
            return { 'inaccuracy': 0, 'mistake': 0, 'blunder': 0 }

        return randomized

    def set_game_context(self, context: GameContext, must_win: bool = False):
        """Set the game context which affects all decisions"""
        self.game_context = context
        self.must_win_mode = must_win

        # Adjust error targets based on context
        if context == GameContext.MUST_WIN:
            # Take more risks, make more errors
            self.errors_per_game = {
                k: int(v * 1.3) for k, v in self.errors_per_game.items()
            }
        elif context == GameContext.MUST_DRAW:
            # Play more carefully
            self.errors_per_game = {
                k: int(v * 0.7) for k, v in self.errors_per_game.items()
            }
        elif context == GameContext.TOURNAMENT:
            # Slightly more careful than casual
            self.errors_per_game = {
                k: int(v * 0.9) for k, v in self.errors_per_game.items()
            }

    def should_force_error_now(self, board: chess.Board, move_count: int,
                               position_eval: float, time_remaining: Optional[float] = None) -> Optional[str]:
        """
        Enhanced error forcing with all subsystems considered
        """
        if not self.enable_errors:
            return None

        # Never error when flagging
        if self.critical_time_pressure:
            return None

        # Cooldown check
        if move_count - self.last_error_move < self.error_cooldown:
            return None

        # Don't error in very early opening (unless low rated)
        if move_count < 5 and self.rating > 1200:
            return None

        # Check pattern blindness - might force errors
        if any(self.pattern_recognition.pattern_blindness.values()):
            # Increase error chance when pattern blind
            pattern_blind_multiplier = 1.5
        else:
            pattern_blind_multiplier = 1.0

        # Emotional state check
        emotional_multiplier = 1.0
        if self.emotional_memory.traumatic_patterns:
            # Recent trauma increases errors
            trauma_level = max(self.emotional_memory.traumatic_patterns.values())
            emotional_multiplier = 1 + (trauma_level * 0.3)

        # Confidence affects errors dramatically
        confidence_multiplier = 2.0 - self.confidence.current_confidence

        # Fatigue increases errors
        fatigue_multiplier = 1 + self.session_fatigue * 0.5

        # Game context
        context_multiplier = 1.0
        if self.game_context == GameContext.MUST_WIN:
            context_multiplier = 1.3
        elif self.game_context == GameContext.TOURNAMENT:
            context_multiplier = 0.9

        # Calculate position pressure
        position_pressure = self._calculate_enhanced_position_pressure(board, position_eval)

        # Check each error type with all modifiers
        for error_type in ['blunder', 'mistake', 'inaccuracy']:
            if self.errors_this_game[error_type] >= self.errors_per_game[error_type]:
                continue

            # Base probability
            base_prob = self._get_base_error_probability(error_type, move_count)

            # Apply ALL modifiers
            error_chance = base_prob
            error_chance *= pattern_blind_multiplier
            error_chance *= emotional_multiplier
            error_chance *= confidence_multiplier
            error_chance *= fatigue_multiplier
            error_chance *= context_multiplier

            # Apply standard context modifiers
            error_chance = self._apply_context_modifiers(
                error_chance, error_type, position_pressure, time_remaining
            )

            # Personality modifier
            if error_type == 'blunder' and self.playing_style.risk_tolerance > 0.7:
                error_chance *= 1.2
            elif error_type == 'inaccuracy' and self.playing_style.intuition_reliance > 0.7:
                error_chance *= 1.1

            # Habitual behavior might prevent errors
            if "think_on_opponent_time" in self.habits.habits and self.habits.habits["think_on_opponent_time"]:
                error_chance *= 0.9

            # Roll the dice
            if random.random() < min(0.5, error_chance):  # Cap at 50%
                print(f"🎲 {self.personality['name']} forcing {error_type}: {error_chance:.2%} chance hit!")

                # Update pattern recognition if applicable
                if error_type == 'blunder':
                    # Blunders often involve missing patterns
                    missed_pattern = random.choice(list(PatternType))
                    self.pattern_recognition.induce_blindness(missed_pattern, severity=5)

                return error_type

        return None

    def _calculate_enhanced_position_pressure(self, board: chess.Board, position_eval: float) -> float:
        """Enhanced position pressure calculation"""
        pressure = 0.0

        # Complexity based on legal moves
        legal_moves = list(board.legal_moves)
        num_moves = len(legal_moves)

        if num_moves > 40:
            pressure += 0.25
        elif num_moves > 30:
            pressure += 0.15
        elif num_moves < 10:
            pressure += 0.1  # Few moves can also be pressure

        # Tactical richness
        captures = [m for m in legal_moves if board.is_capture(m)]
        checks = [m for m in legal_moves if board.gives_check(m)]

        tactical_richness = (len(captures) + len(checks) * 2) / num_moves
        pressure += tactical_richness * 0.3

        # King safety pressure
        if board.is_check():
            pressure += 0.2

        # Time-based position pressure
        move_count = len(board.move_stack)
        if move_count < 15:
            # Opening pressure (for non-book positions)
            if not self.familiar_structure:
                pressure += 0.15
        elif move_count > 40:
            # Long game fatigue
            pressure += 0.1

        # Material imbalance with close eval = sharp position
        material_diff = self._count_material_difference(board)
        if abs(material_diff) > 3 and abs(position_eval) < 200:
            pressure += 0.2
            self.in_sharp_position = True
        else:
            self.in_sharp_position = False

        # Personality affects pressure perception
        if self.playing_style.risk_tolerance < 0.3:
            pressure *= 1.2  # Cautious players feel more pressure
        elif self.playing_style.risk_tolerance > 0.7:
            pressure *= 0.8  # Risk takers thrive in chaos

        return min(1.0, pressure)

    def select_error_move(self, top_moves: List[Dict], best_eval: float,
                          error_type: str) -> Tuple[Optional[str], float]:
        """
        Enhanced error selection considering personality and patterns
        """
        if not top_moves or len(top_moves) < 2:
            return None, 0

        # Get loss range for error type
        min_loss, max_loss = self.error_severity[error_type]

        # Personality affects error selection
        if self.playing_style.calculation_depth < 0.3 and error_type == 'blunder':
            # Poor calculators make worse blunders
            max_loss = int(max_loss * 1.2)
        elif self.playing_style.positional_understanding > 0.8 and error_type == 'inaccuracy':
            # Strong positional players make subtler inaccuracies
            max_loss = int(max_loss * 0.8)

        # Target loss with personality-based distribution
        if self.playing_style.intuition_reliance > 0.7:
            # Intuitive players make more random errors
            target_loss = random.uniform(min_loss, max_loss)
        else:
            # Calculating players make errors closer to threshold
            beta_sample = random.betavariate(2, 5)
            target_loss = min_loss + beta_sample * (max_loss - min_loss)

        # Find candidate moves
        error_candidates = []

        for i, move in enumerate(top_moves[1:], 1):
            move_eval = move.get('Centipawn', 0)
            eval_loss = best_eval - move_eval

            # Skip if move is actually better
            if eval_loss < 0:
                continue

            # Check if move fits error profile
            if min_loss <= eval_loss <= max_loss:
                # Additional checks based on habits
                move_obj = chess.Move.from_uci(move['Move'])

                # Check if this error makes psychological sense
                psychological_fit = self._error_makes_psychological_sense(
                    move_obj, error_type, eval_loss
                )

                if psychological_fit > 0.5:
                    error_candidates.append({
                        'move': move['Move'],
                        'eval_loss': eval_loss,
                        'psychological_fit': psychological_fit,
                        'distance_from_target': abs(eval_loss - target_loss)
                    })

        if error_candidates:
            # Sort by psychological fit and distance from target
            error_candidates.sort(
                key=lambda x: (1 - x['psychological_fit']) * x['distance_from_target']
            )

            # Select with weighted probability
            if len(error_candidates) == 1:
                selected = error_candidates[0]
            else:
                # Weight by psychological fit
                weights = [c['psychological_fit'] ** 2 for c in error_candidates[:5]]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                selected = random.choices(error_candidates[:5], weights=weights)[0]

            print(f"💥 {self.personality['name']} plays {error_type.upper()}: "
                  f"{selected['move']} (loses {selected['eval_loss']:.0f}cp, "
                  f"psychological fit: {selected['psychological_fit']:.2f})")

            # Store traumatic memory if blunder
            if error_type == 'blunder':
                self.emotional_memory.traumatic_patterns[f"position_{len(self.game_error_log)}"] = 0.8

            return selected['move'], selected['eval_loss']

        return None, 0

    def _error_makes_psychological_sense(self, move: chess.Move, error_type: str,
                                         eval_loss: float) -> float:
        """Check if an error is psychologically realistic"""
        psychological_fit = 0.5  # Base fitness

        # Personality-based error patterns
        if error_type == 'blunder':
            # Risk takers make aggressive blunders
            if self.playing_style.risk_tolerance > 0.7:
                if self._is_aggressive_move(move):
                    psychological_fit += 0.3

            # Cautious players blunder by being too passive
            elif self.playing_style.risk_tolerance < 0.3:
                if self._is_passive_move(move):
                    psychological_fit += 0.3

        elif error_type == 'mistake':
            # Intuitive players miss tactical shots
            if self.playing_style.intuition_reliance > 0.7:
                psychological_fit += 0.2

            # Time trouble makes all mistakes more likely
            if self.in_time_pressure:
                psychological_fit += 0.3

        elif error_type == 'inaccuracy':
            # Positional players rarely make positional inaccuracies
            if self.playing_style.positional_understanding > 0.7:
                psychological_fit -= 0.2

        # Recent pattern blindness
        if self.pattern_recognition.recently_missed:
            psychological_fit += 0.2

        # Confidence affects error likelihood
        if self.confidence.current_confidence > 0.8:
            # Overconfident players make careless errors
            psychological_fit += 0.1
        elif self.confidence.current_confidence < 0.5:
            # Underconfident players make passive errors
            if self._is_passive_move(move):
                psychological_fit += 0.2

        # Fatigue makes all errors more likely
        psychological_fit += self.session_fatigue * 0.3

        return max(0.1, min(1.0, psychological_fit))

    def _is_aggressive_move(self, move: chess.Move) -> bool:
        """Check if a move is aggressive in nature"""
        # Simplified check - would be more sophisticated in production
        return (
                str(move).endswith('x') or  # Capture
                str(move).endswith('+') or  # Check
                move.promotion is not None  # Promotion
        )

    def _is_passive_move(self, move: chess.Move) -> bool:
        """Check if a move is passive in nature"""
        # Simplified check - would be more sophisticated in production
        piece_moved = str(move)[:2]
        target_square = str(move)[2:4]

        # Moving backwards or to the side is often passive
        if piece_moved[1] > target_square[1]:  # Moving backwards (simplified)
            return True

        return False

    def get_move_selection_weights(self, candidate_moves: List[Dict],
                                   board: chess.Board) -> List[float]:
        """
        Get sophisticated weights for move selection based on all factors
        """
        if not candidate_moves:
            return []

        weights = []

        for i, move_data in enumerate(candidate_moves):
            weight = 1.0
            move = chess.Move.from_uci(move_data['Move'])

            # Base weight by position
            if i == 0:
                weight = 5.0  # Best move base weight
            elif i == 1:
                weight = 3.0
            elif i == 2:
                weight = 2.0
            else:
                weight = 1.0

            # Personality adjustments
            if self._move_fits_style(move, board):
                weight *= 1.5

            # Habitual preferences
            weight *= self._get_habit_multiplier(move, board)

            # Emotional memory
            weight *= self._get_emotional_multiplier(move, board)

            # Confidence adjustment
            weight *= self.confidence.get_confidence_multiplier()

            # Pattern recognition
            if not self._can_see_move_pattern(move, board):
                weight *= 0.1  # Massive reduction if pattern blind

            # Time pressure
            if self.severe_time_pressure:
                # Prefer simpler moves
                if self._is_simple_move(move, board):
                    weight *= 2.0

            # Game context
            if self.game_context == GameContext.MUST_WIN:
                if self._is_aggressive_move(move):
                    weight *= 1.3
            elif self.game_context == GameContext.MUST_DRAW:
                if self._is_drawing_move(move, board):
                    weight *= 1.5

            weights.append(max(0.01, weight))  # Ensure positive weight

        # Normalize weights
        total_weight = sum(weights)
        return [w / total_weight for w in weights]

    def _move_fits_style(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move fits playing style"""
        fits = True

        # Aggressive players like attacks
        if self.playing_style.aggression > 0.7:
            if board.is_capture(move) or board.gives_check(move):
                return True

        # Positional players like quiet moves
        if self.playing_style.positional_understanding > 0.7:
            if not board.is_capture(move) and not board.gives_check(move):
                return True

        # Risk averse players avoid complications
        if self.playing_style.risk_tolerance < 0.3:
            if self._move_simplifies(move, board):
                return True

        return False

    def _get_habit_multiplier(self, move: chess.Move, board: chess.Board) -> float:
        """Get multiplier based on habitual preferences"""
        multiplier = 1.0

        # Check against habits
        if "knights_before_bishops" in self.habits.superstitions:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KNIGHT:
                multiplier *= 1.2
            elif piece and piece.piece_type == chess.BISHOP:
                multiplier *= 0.9

        if "avoid_early_queen" in self.habits.superstitions:
            if len(board.move_stack) < 10:
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.QUEEN:
                    multiplier *= 0.7

        return multiplier

    def _get_emotional_multiplier(self, move: chess.Move, board: chess.Board) -> float:
        """Get multiplier based on emotional memory"""
        multiplier = 1.0

        # Check if move involves favorite pieces
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in self.playing_style.favorite_pieces:
            multiplier *= 1.1

        # Check traumatic patterns
        move_pattern = self._identify_move_pattern(move, board)
        if move_pattern and str(move_pattern) in self.emotional_memory.traumatic_patterns:
            trauma_level = self.emotional_memory.traumatic_patterns[str(move_pattern)]
            multiplier *= (1 - trauma_level * 0.5)  # Avoid traumatic patterns

        return multiplier

    def _can_see_move_pattern(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if player can see the pattern in this move"""
        pattern = self._identify_move_pattern(move, board)

        if pattern:
            complexity = self._assess_pattern_complexity(pattern, move, board)
            return self.pattern_recognition.can_see_pattern(pattern, complexity)

        return True  # No pattern, so visible

    def _identify_move_pattern(self, move: chess.Move, board: chess.Board) -> Optional[PatternType]:
        """Identify what tactical pattern a move represents"""
        # Simplified pattern detection
        temp_board = board.copy()
        temp_board.push(move)

        # Fork detection
        piece = board.piece_at(move.from_square)
        if piece:
            attacks = 0
            for square in chess.SQUARES:
                if temp_board.is_attacked_by(board.turn, square):
                    target = temp_board.piece_at(square)
                    if target and target.color != board.turn:
                        attacks += 1
            if attacks >= 2:
                return PatternType.FORK

        # Pin detection (simplified)
        # Would need more sophisticated detection in production

        return None

    def _assess_pattern_complexity(self, pattern: PatternType, move: chess.Move,
                                   board: chess.Board) -> float:
        """Assess how complex a pattern is to see"""
        base_complexity = {
            PatternType.HANGING_PIECE: 0.2,
            PatternType.FORK: 0.4,
            PatternType.PIN: 0.5,
            PatternType.SKEWER: 0.6,
            PatternType.DISCOVERED_ATTACK: 0.7,
            PatternType.ZWISCHENZUG: 0.9,
        }

        complexity = base_complexity.get(pattern, 0.5)

        # Adjust for move distance
        from_square = move.from_square
        to_square = move.to_square
        distance = abs(from_square % 8 - to_square % 8) + abs(from_square // 8 - to_square // 8)

        if distance > 4:
            complexity += 0.1  # Long moves harder to see

        # Adjust for piece type
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            complexity += 0.1  # Knight patterns harder

        return min(1.0, complexity)

    def _is_simple_move(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move is simple (for time pressure)"""
        # Recaptures are simple
        if board.is_capture(move) and len(board.move_stack) > 0:
            last_move = board.peek()
            if last_move.to_square == move.to_square:
                return True

        # One square moves are simple
        from_square = move.from_square
        to_square = move.to_square
        distance = abs(from_square % 8 - to_square % 8) + abs(from_square // 8 - to_square // 8)

        return distance <= 2

    def _is_drawing_move(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move tends toward a draw"""
        # Trade pieces when material is equal
        if board.is_capture(move):
            material_diff = self._count_material_difference(board)
            if abs(material_diff) < 2:
                return True

        return False

    def _move_simplifies(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move simplifies the position"""
        if board.is_capture(move):
            # Equal trades simplify
            capturing_piece = board.piece_at(move.from_square)
            captured_piece = board.piece_at(move.to_square)

            if capturing_piece and captured_piece:
                cap_value = self._get_piece_value(capturing_piece.piece_type)
                capt_value = self._get_piece_value(captured_piece.piece_type)

                return abs(cap_value - capt_value) <= 1

        return False

    def update_psychological_state(self, board: chess.Board, current_eval: float,
                                   previous_eval: float, time_remaining: Optional[float] = None,
                                   opponent_time: Optional[float] = None, move_played: Optional[str] = None):
        """
        Comprehensive psychological state update
        """
        # Track evaluation changes
        eval_change = current_eval - previous_eval if previous_eval else 0

        # Update winning state
        self.we_are_winning_big = current_eval > 500

        # Blunder detection and tilt
        if eval_change < -200:  # We lost 200+ cp
            self.just_blundered = True
            self.consecutive_good_moves = 0
            self.tilt_level = min(1.0, self.tilt_level + 0.4)

            # Update pattern recognition - we missed something
            if move_played:
                pattern = self._identify_move_pattern(chess.Move.from_uci(move_played), board)
                if pattern:
                    self.pattern_recognition.induce_blindness(pattern, severity=4)

            # Emotional impact
            self.emotional_memory.traumatic_patterns[f"blunder_{len(self.game_error_log)}"] = 0.9

            # Confidence hit
            self.confidence.current_confidence *= 0.6

        else:
            # Good move tracking
            if abs(eval_change) < 50:
                self.consecutive_good_moves += 1
                self.confidence.current_confidence = min(1.0, self.confidence.current_confidence * 1.05)

                # Enter flow state
                if self.consecutive_good_moves > 5:
                    self.flow_state = True

            # Decay tilt
            self.tilt_level = max(0, self.tilt_level - 0.05)
            self.just_blundered = False

        # Update confidence system
        move_quality = 0.5 + eval_change / 200  # Convert to 0-1 scale
        position_complexity = self._calculate_enhanced_position_pressure(board, current_eval)

        self.confidence.update_confidence(
            move_quality,
            position_complexity,
            time_remaining or 600,
            opponent_time or 600
        )

        # Pattern recognition fatigue
        self.pattern_recognition.fatigue_level = min(1.0,
                                                     self.pattern_recognition.fatigue_level + 0.02)
        self.pattern_recognition.concentration_level = max(0.3,
                                                           1.0 - self.pattern_recognition.fatigue_level * 0.5)

        # Update pattern blindness
        self.pattern_recognition.update_blindness()

        # Time pressure state
        if time_remaining is not None:
            old_pressure = self.in_time_pressure
            time_control = self._estimate_time_control(time_remaining)
            self.update_time_pressure(time_remaining, time_control)

            # Entering time pressure
            if self.in_time_pressure and not old_pressure:
                self.tilt_level = min(1.0, self.tilt_level + 0.1)
                self.pressure_points += 2

        # Opponent modeling
        if opponent_time:
            self.opponent_time_pressure = opponent_time < 60

        # Session fatigue
        self.session_fatigue = min(1.0, self.games_played_session * 0.1 +
                                   len(board.move_stack) * 0.001)

        # Emotional memory decay
        self.emotional_memory.decay_memories()

        # Complex position tracking
        legal_moves = list(board.legal_moves)
        if len(legal_moves) > 30:
            self.complex_position_streak += 1
            self.pressure_points += 1
        else:
            self.complex_position_streak = max(0, self.complex_position_streak - 1)

    def get_thinking_time_modifier(self) -> float:
        """
        Get thinking time modifier based on all psychological factors
        """
        modifier = 1.0

        # Personality base
        if self.playing_style.time_management == TimeManagementStyle.FAST_THEN_THINK:
            modifier *= 0.8
        elif self.playing_style.time_management == TimeManagementStyle.TIME_TROUBLE_ADDICT:
            modifier *= 0.7
        elif self.playing_style.time_management == TimeManagementStyle.CAUTIOUS:
            modifier *= 1.2

        # Confidence affects speed
        if self.confidence.current_confidence > 0.8:
            modifier *= 0.85  # Play faster when confident
        elif self.confidence.current_confidence < 0.5:
            modifier *= 1.15  # Think more when uncertain

        # Flow state
        if self.flow_state:
            modifier *= 0.7  # Everything flows quickly

        # Tilt affects time usage
        if self.tilt_level > 0.5:
            modifier *= random.uniform(0.5, 1.5)  # Erratic time usage

        # Habitual triggers
        thinking_triggers = self.habits.thinking_triggers
        # Would check current position features here

        # Pattern blindness compensation
        if any(self.pattern_recognition.pattern_blindness.values()):
            modifier *= 1.1  # Think a bit more when pattern blind

        # Session fatigue
        modifier *= (1 + self.session_fatigue * 0.3)

        return modifier

    def should_play_instantly(self, move: chess.Move, board: chess.Board) -> bool:
        """
        Check if a move should be played almost instantly
        """
        # Habitual instant moves
        if self.habits.habits.get("premove_recaptures", False):
            if board.is_capture(move) and len(board.move_stack) > 0:
                last_move = board.peek()
                if last_move.to_square == move.to_square:
                    return True

        # Personality-based instant moves
        if self.playing_style.intuition_reliance > 0.8:
            # Intuitive players move quickly on "obvious" moves
            if self._is_obvious_move(move, board):
                return random.random() < 0.7

        # Flow state enables quick moves
        if self.flow_state and self.consecutive_good_moves > 3:
            return random.random() < 0.4

        return False

    def _is_obvious_move(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move is obvious"""
        # Only legal move
        if len(list(board.legal_moves)) == 1:
            return True

        # Escaping check with few options
        if board.is_check() and len(list(board.legal_moves)) <= 3:
            return True

        # Recapture
        if board.is_capture(move) and len(board.move_stack) > 0:
            last_move = board.peek()
            if last_move.to_square == move.to_square:
                return True

        return False

    def _get_piece_value(self, piece_type: chess.PieceType) -> int:
        """Get piece value"""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        return values.get(piece_type, 0)

    def _count_material_difference(self, board: chess.Board) -> int:
        """Count material difference"""
        white_material = 0
        black_material = 0

        for square, piece in board.piece_map().items():
            value = self._get_piece_value(piece.piece_type)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value

        return white_material - black_material

    def _get_base_error_probability(self, error_type: str, move_count: int) -> float:
        """Get base error probability with personality influence"""
        # Expected game length varies by style
        if self.playing_style.endgame_preference > 0.7:
            expected_game_length = 50
        elif self.playing_style.aggression > 0.7:
            expected_game_length = 35
        else:
            expected_game_length = 42

        # Progress through game
        progress = move_count / expected_game_length
        expected_errors = progress * self.errors_per_game[error_type]
        actual_errors = self.errors_this_game[error_type]
        error_deficit = expected_errors - actual_errors

        # Base probabilities with personality
        skill_factor = (20 - self.skill_level) / 20

        if error_type == 'inaccuracy':
            base = 0.15 * (1 + skill_factor)
            # Intuitive players make more inaccuracies
            if self.playing_style.intuition_reliance > 0.7:
                base *= 1.2
        elif error_type == 'mistake':
            base = 0.10 * (1 + skill_factor)
            # Poor calculators make more mistakes
            if self.playing_style.calculation_depth < 0.3:
                base *= 1.3
        else:  # blunder
            base = 0.05 * (1 + skill_factor)
            # Risk takers blunder more
            if self.playing_style.risk_tolerance > 0.7:
                base *= 1.4

        # Increase if behind schedule
        if error_deficit > 0.3:
            urgency_multiplier = min(2.0, 1 + error_deficit)
            base *= urgency_multiplier

        return min(0.4, base)

    def _apply_context_modifiers(self, base_prob: float, error_type: str,
                                 position_pressure: float, time_remaining: Optional[float]) -> float:
        """Apply all contextual modifiers to error probability"""
        modified_prob = base_prob

        # Time pressure
        if self.severe_time_pressure:
            time_multiplier = 2.5 if error_type == 'blunder' else 1.8
            modified_prob *= time_multiplier
        elif self.in_time_pressure:
            modified_prob *= 1.4

        # Position pressure
        if position_pressure > 0.5:
            complexity_mult = 1 + position_pressure
            if error_type == 'blunder':
                complexity_mult *= 1.3
            modified_prob *= complexity_mult

        # Tilt cascade
        if self.tilt_level > 0.3:
            tilt_mult = 1 + (self.tilt_level * 0.7)
            if error_type == 'blunder' and self.just_blundered:
                tilt_mult *= 1.8  # Blunder chains when tilted
            modified_prob *= tilt_mult

        # Overconfidence
        if self.we_are_winning_big:
            modified_prob *= 1.5
            if self.confidence.current_confidence > 0.85:
                modified_prob *= 1.2  # Compound overconfidence

        # Complacency
        if self.consecutive_good_moves > 8:
            modified_prob *= 1.4

        # Fatigue
        if self.moves_in_time_pressure > 10:
            modified_prob *= 1.3
        if self.session_fatigue > 0.5:
            modified_prob *= (1 + self.session_fatigue * 0.5)

        # But careful after blunder (unless tilted heavily)
        if self.just_blundered and self.tilt_level < 0.7:
            modified_prob *= 0.5

        # Game context
        if self.game_context == GameContext.MUST_WIN:
            modified_prob *= 1.2
        elif self.game_context == GameContext.TOURNAMENT:
            modified_prob *= 0.9

        return min(0.6, modified_prob)  # Cap at 60%

    def update_time_pressure(self, time_remaining: float, time_control: str = 'blitz'):
        """Update time pressure states with personality influence"""
        # Base thresholds
        if time_control == 'bullet':
            critical = 5
            severe = 10
            normal = 30
        elif time_control == 'blitz':
            critical = 10
            severe = 20
            normal = 60
        else:  # rapid
            critical = 15
            severe = 30
            normal = 120

        # Personality adjustments
        if self.playing_style.time_management == TimeManagementStyle.TIME_TROUBLE_ADDICT:
            # These players are comfortable in time pressure
            critical *= 0.7
            severe *= 0.7
            normal *= 0.7
        elif self.playing_style.time_management == TimeManagementStyle.CAUTIOUS:
            # These players feel pressure earlier
            critical *= 1.3
            severe *= 1.3
            normal *= 1.3

        # Set states
        self.critical_time_pressure = time_remaining < critical
        self.severe_time_pressure = time_remaining < severe
        self.in_time_pressure = time_remaining < normal

        if self.in_time_pressure:
            self.moves_in_time_pressure += 1
        else:
            self.moves_in_time_pressure = 0

    def _estimate_time_control(self, time_remaining: float) -> str:
        """Estimate time control from remaining time"""
        if time_remaining < 180:
            return 'bullet'
        elif time_remaining < 600:
            return 'blitz'
        else:
            return 'rapid'

    def record_error(self, move_count: int, move: str, eval_loss: float, error_type: str):
        """Record error with full psychological impact"""
        # Base recording
        self.errors_this_game[error_type] += 1
        self.last_error_move = move_count

        self.game_error_log.append({
            'move_number': move_count,
            'move': move,
            'eval_loss': eval_loss,
            'type': error_type,
            'confidence_before': self.confidence.current_confidence,
            'tilt_level': self.tilt_level,
        })

        # Pattern learning
        if error_type == 'blunder':
            # Learn from blunders
            pattern_key = f"blunder_pattern_{len(self.game_error_log)}"
            self.emotional_memory.traumatic_patterns[pattern_key] = 0.9

            # Major confidence hit
            self.just_blundered = True
            self.tilt_level = min(1.0, self.tilt_level + 0.5)
            self.confidence.current_confidence *= 0.5
            self.consecutive_good_moves = 0

            # Induce pattern blindness
            self.pattern_recognition.induce_blindness(
                random.choice(list(PatternType)), severity=5
            )

        elif error_type == 'mistake':
            self.tilt_level = min(1.0, self.tilt_level + 0.2)
            self.confidence.current_confidence *= 0.75
            self.consecutive_good_moves = max(0, self.consecutive_good_moves - 3)

        else:  # inaccuracy
            self.tilt_level = min(1.0, self.tilt_level + 0.05)
            self.confidence.current_confidence *= 0.9
            self.consecutive_good_moves = max(0, self.consecutive_good_moves - 1)

        # Break flow state
        self.flow_state = False

    def log_game_summary(self):
        """Enhanced game summary with personality insights"""
        print("\n" + "=" * 60)
        print(f"🎯 {self.personality['name']} - GAME REPORT")
        print(f"📊 Rating: {self.rating} | Skill Level: {self.skill_level}")
        print("=" * 60)

        # Personality summary
        print(f"\n🧠 Playing Style:")
        print(f"   Aggression: {self.playing_style.aggression:.2f}")
        print(f"   Positional: {self.playing_style.positional_understanding:.2f}")
        print(f"   Risk Tolerance: {self.playing_style.risk_tolerance:.2f}")
        print(f"   Time Management: {self.playing_style.time_management.value}")

        # Error summary
        total_errors = sum(self.errors_this_game.values())
        if total_errors == 0:
            print("\n✨ PERFECT GAME - No errors!")
            print(f"   Confidence maintained: {self.confidence.current_confidence:.2f}")
        else:
            print(f"\n📊 Total errors: {total_errors}")
            print(f"   - Inaccuracies: {self.errors_this_game['inaccuracy']}")
            print(f"   - Mistakes: {self.errors_this_game['mistake']}")
            print(f"   - Blunders: {self.errors_this_game['blunder']}")

            if self.game_error_log:
                total_loss = sum(e['eval_loss'] for e in self.game_error_log)
                avg_loss = total_loss / len(self.game_error_log)
                print(f"\n📈 Average error severity: {avg_loss:.0f} centipawns")
                print(f"📉 Total evaluation lost: {total_loss:.0f} centipawns")

                # Psychological journey
                print("\n🧘 Psychological Journey:")
                for error in self.game_error_log:
                    icon = "😕" if error['type'] == 'inaccuracy' else "😰" if error['type'] == 'mistake' else "💀"
                    print(f"   Move {error['move_number']}: {icon} {error['move']} "
                          f"(-{error['eval_loss']:.0f}cp) "
                          f"[Confidence: {error['confidence_before']:.2f}, "
                          f"Tilt: {error['tilt_level']:.2f}]")

        # Pattern blindness report
        if self.pattern_recognition.recently_missed:
            print(f"\n👁️ Pattern Recognition Issues:")
            pattern_counts = { }
            for pattern in self.pattern_recognition.recently_missed:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            for pattern, count in pattern_counts.items():
                print(f"   {pattern.value}: missed {count} times")

        # Emotional state
        print(f"\n💭 Final Mental State:")
        print(f"   Confidence: {self.confidence.current_confidence:.2f}")
        print(f"   Tilt Level: {self.tilt_level:.2f}")
        print(f"   Session Fatigue: {self.session_fatigue:.2f}")

        if self.emotional_memory.traumatic_patterns:
            print(f"   Traumatic Memories: {len(self.emotional_memory.traumatic_patterns)}")

        # Performance vs targets
        print(f"\n🎯 Performance vs Personality Expectations:")
        for error_type in ['inaccuracy', 'mistake', 'blunder']:
            actual = self.errors_this_game[error_type]
            target = self.errors_per_game[error_type]
            if actual == target:
                print(f"   {error_type}: ✓ {actual}/{target} (as expected)")
            elif actual < target:
                print(f"   {error_type}: ⬇️ {actual}/{target} (better than expected!)")
            else:
                print(f"   {error_type}: ⬆️ {actual}/{target} (more than expected)")

        # Habits exhibited
        if hasattr(self, 'habits_exhibited'):
            print(f"\n🎭 Habits Exhibited:")
            for habit in self.habits_exhibited:
                print(f"   - {habit}")

        print("=" * 60 + "\n")

    def reset_for_new_game(self):
        """Reset for new game with learning and personality persistence"""
        # Increment session counter
        self.games_played_session += 1

        # Partial emotional memory decay (not full reset)
        self.emotional_memory.decay_memories(decay_rate=0.7)

        # Pattern recognition partial reset
        self.pattern_recognition.fatigue_level *= 0.5
        self.pattern_recognition.concentration_level = 1.0 - self.pattern_recognition.fatigue_level * 0.3
        self.pattern_recognition.pattern_blindness.clear()

        # Confidence partial reset toward base
        self.confidence.current_confidence = (
                self.confidence.current_confidence * 0.3 +
                self.confidence.base_confidence * 0.7
        )

        # New error targets with session influence
        self.errors_per_game = self._randomize_error_targets()

        # Reset game-specific tracking
        self.errors_this_game = { 'inaccuracy': 0, 'mistake': 0, 'blunder': 0 }
        self.last_error_move = 0
        self.game_error_log = []

        # Reset psychological state (but not completely)
        self.just_blundered = False
        self.consecutive_good_moves = 0
        self.tilt_level *= 0.3  # Some tilt carries over
        self.flow_state = False
        self.pressure_points = 0

        # Reset time pressure
        self.in_time_pressure = False
        self.severe_time_pressure = False
        self.critical_time_pressure = False
        self.moves_in_time_pressure = 0

        # Reset position tracking
        self.complex_position_streak = 0
        self.in_sharp_position = False
        self.position_tension = 0.0
        self.familiar_structure = False

        # Learning adjustments
        if self.games_played_session > 1:
            # Apply learning from previous games
            self._apply_session_learning()

    def _apply_session_learning(self):
        """Apply learning effects from session experience"""
        # Reduce weakness levels slightly
        for weakness in self.rating_weaknesses.improving_areas:
            if weakness in self.rating_weaknesses.weaknesses:
                self.rating_weaknesses.weaknesses[weakness] *= 0.95

        # Improve pattern recognition thresholds
        self.pattern_recognition.pattern_complexity_threshold *= 0.98

        # Confidence becomes more stable
        self.confidence.base_confidence = (
                self.confidence.base_confidence * 0.9 +
                self.confidence.current_confidence * 0.1
        )

    def export_personality(self) -> Dict:
        """Export personality for reuse"""
        return {
            'name': self.personality['name'],
            'style': {
                'aggression': self.playing_style.aggression,
                'positional_understanding': self.playing_style.positional_understanding,
                'risk_tolerance': self.playing_style.risk_tolerance,
                'endgame_preference': self.playing_style.endgame_preference,
                'calculation_depth': self.playing_style.calculation_depth,
                'intuition_reliance': self.playing_style.intuition_reliance,
                'time_management': self.playing_style.time_management.value,
            },
            'quirks': self.personality['quirks'],
            'rating': self.rating,
            'games_played': self.games_played_session,
        }
