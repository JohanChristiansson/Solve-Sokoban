from enum import Enum

class RoomState(Enum):
    WALL = 0
    EMPTY = 1
    GOAL = 2
    BOX_ON_GOAL = 3
    BOX = 4
    PLAYER = 5
    PLAYER_ON_GOAL = 6
