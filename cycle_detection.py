from typing import Union
import numpy as np
from constants import RoomState

class CycleDetection:
    """
    Cycle detection class for detecting cycles.
    """
    def __init__(self, room_state: Union[np.ndarray, None]=None):
        # Dict: keys = list of box positions, values = list of player positions of that board state
        self.graph = {}
        if room_state is not None:
            self.push_state(room_state)

    def push_state(self, room_state: np.ndarray):
        """
        Update visited states.
        """
        box_position_map = np.logical_or(room_state == RoomState.BOX.value, room_state == RoomState.BOX_ON_GOAL.value)
        box_positions = self._convert_position_to_tuple(np.where(box_position_map))

        player_position_map = np.logical_or(room_state == RoomState.PLAYER.value, room_state == RoomState.PLAYER_ON_GOAL.value)
        player_position = np.where(player_position_map)
        player_position = (int(player_position[0]), int(player_position[1]))
        if box_positions not in self.graph:
            # New box position, create a new entry in visited states
            self.graph[box_positions] = [player_position]
        else:
            # Otherwise, just add to an existing entry
            self.graph[box_positions].append(player_position)

    def is_cycle(self, room_state: np.ndarray, x: int, y: int) -> bool:
        """
        Return True if player position (y, x) will cause a cycle, else False.
        """
        box_position_map = np.logical_or(room_state == RoomState.BOX.value, room_state == RoomState.BOX_ON_GOAL.value)
        box_positions = self._convert_position_to_tuple(np.where(box_position_map))
        if box_positions in self.graph:
            if (y, x) in self.graph[box_positions]:
                return True
        return False

    def _convert_position_to_tuple(self, position: tuple[np.ndarray, np.ndarray]) -> tuple[np.int64, np.int64]:
        """
        Convert a position to a tuple.
        """
        y, x = position[0], position[1]
        return tuple([pos for pos in zip(y, x)])
