"""
Bowling Game Manager
====================
Manages game state, player turns, and implements standard 10-pin bowling scoring.

Author: AI Intro Final Project
Date: 2024-12
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time


class GameState(Enum):
    """Game state machine states"""
    SETTINGS = "settings"
    REGISTRATION = "registration"
    WAITING_FOR_THROW = "waiting"
    TRACKING_PLAYER = "tracking_player"
    TRACKING_BALL = "tracking_ball"
    TRACKING_BOTTLE = "tracking_bottle"
    COUNTING = "counting"
    GAME_OVER = "game_over"


@dataclass
class Frame:
    """Represents a single bowling frame"""
    throw1: Optional[int] = None  # Pins knocked down in first throw
    throw2: Optional[int] = None  # Pins knocked down in second throw
    throw3: Optional[int] = None  # Only used in 10th frame
    
    def is_strike(self) -> bool:
        return self.throw1 == 10
    
    def is_spare(self) -> bool:
        if self.throw1 is None or self.throw2 is None:
            return False
        return not self.is_strike() and (self.throw1 + self.throw2 == 10)
    
    def is_complete(self, is_tenth_frame: bool = False) -> bool:
        if is_tenth_frame:
            # 10th frame: need 3 throws if strike or spare
            if self.throw1 == 10:  # Strike
                return self.throw2 is not None and self.throw3 is not None
            elif self.throw1 is not None and self.throw2 is not None:
                if self.throw1 + self.throw2 >= 10:  # Spare
                    return self.throw3 is not None
                return True
            return False
        else:
            # Normal frame: complete after strike or 2 throws
            if self.throw1 == 10:
                return True
            return self.throw1 is not None and self.throw2 is not None
    
    def pins_knocked(self) -> int:
        """Total pins knocked in this frame (not score)"""
        total = 0
        if self.throw1 is not None:
            total += self.throw1
        if self.throw2 is not None:
            total += self.throw2
        if self.throw3 is not None:
            total += self.throw3
        return total
    
    def get_display(self, is_tenth_frame: bool = False) -> Tuple[str, str, Optional[str]]:
        """Get display strings for the frame"""
        def format_throw(pins: Optional[int], prev_pins: int = 0, is_first: bool = True) -> str:
            if pins is None:
                return ""
            if pins == 10 and is_first:
                return "X"
            if pins + prev_pins == 10 and not is_first:
                return "/"
            if pins == 0:
                return "-"
            return str(pins)
        
        t1 = format_throw(self.throw1, 0, True)
        
        if is_tenth_frame:
            # Special handling for 10th frame
            if self.throw1 == 10:
                t2 = format_throw(self.throw2, 0, True)  # After strike, reset count
                if self.throw2 == 10:
                    t3 = format_throw(self.throw3, 0, True)
                else:
                    t3 = format_throw(self.throw3, self.throw2 or 0, False)
            else:
                t2 = format_throw(self.throw2, self.throw1 or 0, False)
                if self.throw1 is not None and self.throw2 is not None and self.throw1 + self.throw2 == 10:
                    t3 = format_throw(self.throw3, 0, True)  # After spare, reset count
                else:
                    t3 = format_throw(self.throw3, 0, True) if self.throw3 is not None else None
            return (t1, t2, t3)
        else:
            t2 = format_throw(self.throw2, self.throw1 or 0, False)
            return (t1, t2, None)


@dataclass
class Player:
    """Represents a bowling player"""
    name: str
    frames: List[Frame] = field(default_factory=lambda: [Frame() for _ in range(10)])
    scores: List[Optional[int]] = field(default_factory=lambda: [None] * 10)  # Cumulative scores
    
    def get_current_frame_index(self) -> int:
        """Get the index of the current frame being played"""
        num_frames = len(self.frames)
        for i, frame in enumerate(self.frames):
            is_last = (i == num_frames - 1)
            # Treat the last frame as special (allow 3 throws if strike/spare)
            if not frame.is_complete(is_last):
                return i
        return num_frames - 1  # All frames complete
    
    def get_current_throw(self) -> int:
        """Get which throw we're on (1, 2, or 3 for final frame)"""
        frame_idx = self.get_current_frame_index()
        frame = self.frames[frame_idx]
        
        if frame.throw1 is None:
            return 1
        elif frame.throw2 is None:
            return 2
        else:
            return 3
    
    def calculate_scores(self) -> None:
        """Calculate cumulative scores for all frames"""
        cumulative = 0
        num_frames = len(self.frames)
        
        for i in range(num_frames):
            frame = self.frames[i]
            
            # Can't score if throw1 not done
            if frame.throw1 is None:
                self.scores[i] = None
                continue
            
            is_last = (i == num_frames - 1)
            
            if is_last:  # Final frame
                if frame.is_complete(is_tenth_frame=True):
                    frame_score = frame.pins_knocked()
                    cumulative += frame_score
                    self.scores[i] = cumulative
                else:
                    self.scores[i] = None
            else:
                if frame.is_strike():
                    # Need next 2 throws
                    bonus = self._get_next_throws(i, 2)
                    if bonus is not None:
                        cumulative += 10 + bonus
                        self.scores[i] = cumulative
                    else:
                        self.scores[i] = None
                elif frame.is_spare():
                    # Need next 1 throw
                    bonus = self._get_next_throws(i, 1)
                    if bonus is not None:
                        cumulative += 10 + bonus
                        self.scores[i] = cumulative
                    else:
                        self.scores[i] = None
                else:
                    # Normal frame
                    if frame.throw2 is not None:
                        cumulative += frame.throw1 + frame.throw2
                        self.scores[i] = cumulative
                    else:
                        self.scores[i] = None
    
    def _get_next_throws(self, frame_idx: int, count: int) -> Optional[int]:
        """Get the sum of the next 'count' throws after a frame"""
        throws = []
        num_frames = len(self.frames)
        
        for i in range(frame_idx + 1, num_frames):
            frame = self.frames[i]
            if frame.throw1 is not None:
                throws.append(frame.throw1)
            if len(throws) >= count:
                break
            if frame.throw2 is not None:
                throws.append(frame.throw2)
            if len(throws) >= count:
                break
            # Only include throw3 if this is the final frame
            if i == num_frames - 1 and frame.throw3 is not None:
                throws.append(frame.throw3)
            if len(throws) >= count:
                break
        
        if len(throws) >= count:
            return sum(throws[:count])
        return None
    
    def get_total_score(self) -> int:
        """Get the current total score"""
        num_frames = len(self.scores)
        for i in range(num_frames - 1, -1, -1):
            if self.scores[i] is not None:
                return self.scores[i]
        return 0
    
    def is_game_complete(self) -> bool:
        """Check if this player has completed all frames"""
        num_frames = len(self.frames)
        last_frame_idx = num_frames - 1
        is_tenth = (num_frames == 10)
        return self.frames[last_frame_idx].is_complete(is_tenth_frame=is_tenth)


class GameManager:
    """
    Manages the bowling game state and flow.
    
    Handles:
    - Player registration
    - Turn management
    - Score tracking
    - Game state transitions
    """
    
    def __init__(self, num_frames: int = 10, num_bottles: int = 10):
        self.num_frames = min(20, max(1, num_frames))  # 1-20 frames
        self.num_bottles = max(1, num_bottles)  # At least 1 bottle
        self.state = GameState.SETTINGS
        self.players: List[Player] = []
        self.current_player_index = 0
        self.registered_player_names: List[str] = []
        
        # Tracking state
        self.tracking_target: str = "person"  # Current tracking target
        self.last_ball_seen_time: Optional[float] = None
        self.counting_start_time: Optional[float] = None
        self.standing_bottles_before_throw: int = self.num_bottles
        
        # Settings
        self.com_port: str = "/dev/cu.usbmodem1201"
        
        # Score correction
        self.last_throw_info: Optional[Dict] = None  # For score correction
    
    def set_settings(self, com_port: str, num_frames: int, num_bottles: int = 10) -> None:
        """Set game settings"""
        self.com_port = com_port
        self.num_frames = min(20, max(1, num_frames))
        self.num_bottles = max(1, num_bottles)
        self.standing_bottles_before_throw = self.num_bottles
    
    def register_player(self, name: str) -> bool:
        """Register a new player"""
        if len(self.players) >= 2:
            return False
        if name in self.registered_player_names:
            return False
        
        # Create player with the configured number of frames
        player = Player(name=name)
        # Adjust frames if not playing full 10 (or if >10)
        # Always re-init frames to match current settings
        player.frames = [Frame() for _ in range(self.num_frames)]
        player.scores = [None] * self.num_frames
        
        self.players.append(player)
        self.registered_player_names.append(name)
        return True
    
    def start_game(self) -> bool:
        """Start the game after registration"""
        if len(self.players) != 2:
            return False
        
        self.state = GameState.WAITING_FOR_THROW
        self.current_player_index = 0
        self.standing_bottles_before_throw = self.num_bottles
        return True
    
    def get_current_player(self) -> Optional[Player]:
        """Get the current player"""
        if not self.players:
            return None
        return self.players[self.current_player_index]
    
    def get_current_player_name(self) -> str:
        """Get current player's name"""
        player = self.get_current_player()
        return player.name if player else "Unknown"
    
    def get_tracking_target(self) -> str:
        """Get what should be tracked based on current state"""
        if self.state == GameState.TRACKING_PLAYER:
            return "person"
        elif self.state == GameState.TRACKING_BALL:
            return "tennis_ball"
        elif self.state in (GameState.TRACKING_BOTTLE, GameState.COUNTING):
            return "bottle"
        return "person"
    
    def start_throw(self) -> None:
        """Called when a throw sequence begins"""
        self.state = GameState.TRACKING_PLAYER
        self.tracking_target = "person"
        # Do NOT reset standing_bottles_before_throw here. 
        # It is set correctly by start_game, reset_game, and record_throw (for next player/throw).
    
    def on_ball_detected(self) -> None:
        """Called when the ball is detected (leaving player's hand)"""
        if self.state == GameState.TRACKING_PLAYER:
            self.state = GameState.TRACKING_BALL
            self.tracking_target = "tennis_ball"
            self.last_ball_seen_time = time.time()
    
    def on_ball_lost(self) -> None:
        """Called when ball is no longer visible"""
        if self.state == GameState.TRACKING_BALL:
            self.last_ball_seen_time = None
    
    def on_bottles_visible(self) -> None:
        """Called when bottles become visible"""
        if self.state == GameState.TRACKING_BALL:
            self.state = GameState.TRACKING_BOTTLE
            self.tracking_target = "bottle"
            self.counting_start_time = time.time()
    
    def check_counting_complete(self) -> bool:
        """Check if 3-second counting wait is complete"""
        if self.state == GameState.TRACKING_BOTTLE and self.counting_start_time:
            if time.time() - self.counting_start_time >= 3.0:
                self.state = GameState.COUNTING
                return True
        return False
    
    def record_throw(self, standing_bottles: int) -> Dict:
        """
        Record the result of a throw.
        
        Args:
            standing_bottles: Number of bottles still standing
            
        Returns:
            Dict with throw result info
        """
        player = self.get_current_player()
        if not player:
            return {"error": "No current player"}
        
        pins_knocked = self.standing_bottles_before_throw - standing_bottles
        pins_knocked = max(0, min(self.num_bottles, pins_knocked))  # Clamp to valid range
        
        frame_idx = player.get_current_frame_index()
        frame = player.frames[frame_idx]
        throw_num = player.get_current_throw()
        is_last_frame = (frame_idx == self.num_frames - 1)
        
        # Record the throw
        if throw_num == 1:
            frame.throw1 = pins_knocked
        elif throw_num == 2:
            frame.throw2 = pins_knocked
        else:
            frame.throw3 = pins_knocked
        
        # Save for potential correction
        self.last_throw_info = {
            "player_index": self.current_player_index,
            "frame_index": frame_idx,
            "throw_num": throw_num,
            "pins_knocked": pins_knocked
        }
        
        # Recalculate scores
        player.calculate_scores()
        
        # Determine next action
        result = {
            "player": player.name,
            "frame": frame_idx + 1,
            "throw": throw_num,
            "pins_knocked": pins_knocked,
            "is_strike": throw_num == 1 and pins_knocked == self.num_bottles,
            "is_spare": throw_num == 2 and frame.is_spare(),
            "frame_complete": frame.is_complete(is_last_frame),
            "total_score": player.get_total_score(),
        }
        
        # Handle next state
        if frame.is_complete(is_last_frame):
            # Frame complete, check if game over or switch player
            if self._check_game_over():
                self.state = GameState.GAME_OVER
                result["game_over"] = True
            else:
                self._next_player()
                self.standing_bottles_before_throw = self.num_bottles
                self.state = GameState.WAITING_FOR_THROW
        else:
            # Same player, next throw
            if is_last_frame:
                # In final frame, reset pins after strike
                if throw_num == 1 and pins_knocked == self.num_bottles:
                    self.standing_bottles_before_throw = self.num_bottles
                elif throw_num == 2 and (frame.throw1 or 0) + (frame.throw2 or 0) >= self.num_bottles:
                    self.standing_bottles_before_throw = self.num_bottles
                else:
                    self.standing_bottles_before_throw = standing_bottles
            else:
                self.standing_bottles_before_throw = standing_bottles
            
            self.state = GameState.WAITING_FOR_THROW
        
        return result
    
    def correct_last_throw(self, new_pins_knocked: int) -> Optional[Dict]:
        """
        Correct the last recorded throw.
        
        Args:
            new_pins_knocked: Corrected number of pins knocked
            
        Returns:
            Updated throw info or None if correction failed
        """
        if not self.last_throw_info:
            return None
        
        info = self.last_throw_info
        player = self.players[info["player_index"]]
        frame = player.frames[info["frame_index"]]
        
        # Update the throw
        if info["throw_num"] == 1:
            frame.throw1 = new_pins_knocked
        elif info["throw_num"] == 2:
            frame.throw2 = new_pins_knocked
        else:
            frame.throw3 = new_pins_knocked
        
        # Recalculate all scores
        player.calculate_scores()
        
        # Update last throw info
        self.last_throw_info["pins_knocked"] = new_pins_knocked
        
        return {
            "player": player.name,
            "frame": info["frame_index"] + 1,
            "throw": info["throw_num"],
            "new_pins_knocked": new_pins_knocked,
            "total_score": player.get_total_score()
        }
    
    def _next_player(self) -> None:
        """Switch to the next player"""
        self.current_player_index = (self.current_player_index + 1) % 2
    
    def _check_game_over(self) -> bool:
        """Check if the game is over"""
        # Game is over when both players complete all their frames
        for player in self.players:
            last_frame_idx = self.num_frames - 1
            is_last = (last_frame_idx == len(player.frames) - 1)
            if not player.frames[last_frame_idx].is_complete(is_last):
                return False
        return True
    
    def get_winner(self) -> Optional[str]:
        """Get the winner's name"""
        if self.state != GameState.GAME_OVER:
            return None
        
        scores = [(p.name, p.get_total_score()) for p in self.players]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores[0][1] == scores[1][1]:
            return "Tie"
        return scores[0][0]
    
    def get_game_state_dict(self) -> Dict:
        """Get the full game state as a dictionary"""
        players_data = []
        for player in self.players:
            frames_data = []
            for i, frame in enumerate(player.frames):
                is_last_frame = (i == self.num_frames - 1)
                display = frame.get_display(is_last_frame)
                frames_data.append({
                    "throw1": display[0],
                    "throw2": display[1],
                    "throw3": display[2] if is_last_frame else None,
                    "score": player.scores[i],
                    "is_strike": frame.is_strike(),
                    "is_spare": frame.is_spare()
                })
            
            players_data.append({
                "name": player.name,
                "frames": frames_data,
                "total_score": player.get_total_score(),
                "current_frame": player.get_current_frame_index() + 1,
                "current_throw": player.get_current_throw()
            })
        
        current_player = self.get_current_player()
        
        return {
            "state": self.state.value,
            "players": players_data,
            "current_player_index": self.current_player_index,
            "current_player_name": current_player.name if current_player else None,
            "current_frame": current_player.get_current_frame_index() + 1 if current_player else 0,
            "current_throw": current_player.get_current_throw() if current_player else 0,
            "num_frames": self.num_frames,
            "tracking_target": self.tracking_target,
            "winner": self.get_winner() if self.state == GameState.GAME_OVER else None
        }
    
    def reset_game(self) -> None:
        """Reset the game for a new round"""
        self.players = []
        self.registered_player_names = []
        self.current_player_index = 0
        self.state = GameState.REGISTRATION
        self.last_throw_info = None
        self.standing_bottles_before_throw = self.num_bottles


# Test code
if __name__ == "__main__":
    # Test bowling scoring
    gm = GameManager(num_frames=10)
    gm.register_player("Alice")
    gm.register_player("Bob")
    gm.start_game()
    
    # Simulate a game
    print("Testing bowling scoring...")
    
    # Alice: Strike
    gm.record_throw(0)  # 10 pins knocked
    print(f"Alice Frame 1: Strike - Score: {gm.players[0].get_total_score()}")
    
    # Bob: 7, spare
    gm.record_throw(3)  # 7 pins knocked
    gm.record_throw(0)  # 3 more pins (spare)
    print(f"Bob Frame 1: Spare - Score: {gm.players[1].get_total_score()}")
    
    print("\nGame state:", gm.get_game_state_dict())
