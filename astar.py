import copy
# from collections import deque
from queue import PriorityQueue
from constants import RoomState
import numpy as np # type: ignore

class Node:
    """
    A node in the searching space. Represents a state.
    """
    def __init__(self, priority, state, move=None, parent=None):
        self.priority = priority
        self.state = state
        self.move = move
        self.parent = parent

    def __lt__(self, other):
        return self.priority < other.priority

class AStar:
    """
    A* search using priority queue with heuristics.
    """
    def __init__(self, start_state, heuristics_with_weights, env):
        self.start_state = start_state
        self.visited = set()
        self.heuristics_with_weights = heuristics_with_weights
        self.env = env

    def search(self) -> list["Node"]:
        """
        Perform an a* search.
        """
        queue = PriorityQueue() # type: ignore
        queue.put(Node(self.calculate_heuristic(self.start_state), self.start_state))
        while queue:
            node = queue.get()
            self.visited.add(node.state.tostring())
            if self.is_done(node.state):
                print("Done. Visited states: ", len(self.visited))
                path = []
                while node is not None:
                    if node.move:
                        path.append(node.move.translate_action())
                    node = node.parent
                path.reverse()
                return path
            for move in self.valid_moves(node.state):
                new_state = self.simulate_move(node.state, move)
                if new_state.tostring() in self.visited:
                    continue
                queue.put(Node(self.calculate_heuristic(new_state), new_state, move, node))
        return []

    def calculate_heuristic(self, state: np.ndarray) -> float:
        heuristic_value = 0
        for heuristic, weight in self.heuristics_with_weights:
            current_value = heuristic.run(state, self.env)
            heuristic_value += current_value * weight
        return heuristic_value

    def simulate_move(self, state: np.ndarray, move: "SensibleMove") -> np.ndarray:
        """
        Simulate the move in the given state
        """
        new_state = copy.deepcopy(state)
        player_position = np.where(np.logical_or(new_state == RoomState.PLAYER.value, new_state == RoomState.PLAYER_ON_GOAL.value))
        if move.move_box:
            # Update old box position state values
            box_position = (player_position[0] + move.y, player_position[1] + move.x)
            if new_state[box_position] == RoomState.BOX_ON_GOAL.value:
                new_state[box_position] = RoomState.GOAL.value
            else:
                new_state[box_position] = RoomState.EMPTY.value
            # Update new box position state values
            new_box_position = (box_position[0] + move.y, box_position[1] + move.x)
            if new_state[new_box_position] == RoomState.GOAL.value:
                new_state[new_box_position] = RoomState.BOX_ON_GOAL.value
            else:
                new_state[new_box_position] = RoomState.BOX.value
        # Update old player position state values
        if new_state[player_position] == RoomState.PLAYER_ON_GOAL.value:
            new_state[player_position] = RoomState.GOAL.value
        else:
            new_state[player_position] = RoomState.EMPTY.value
        # Update new player position state values
        new_player_position = (player_position[0] + move.y, player_position[1] + move.x)
        if new_state[new_player_position] == RoomState.GOAL.value:
            new_state[new_player_position] = RoomState.PLAYER_ON_GOAL.value
        else:
            new_state[new_player_position] = RoomState.PLAYER.value
        return new_state

    def is_done(self, state):
        """
        Check if the game is done
        """
        return not np.any(state == RoomState.BOX.value)

    def valid_moves(self, room_state: np.ndarray) -> list["SensibleMove"]:
        """
        Returns a list of valid moves that the player can make
        """
        def inbound(y : int, x : int) -> bool:
            """
            Check if the position is inbound
            """
            return 0 <= y < len(room_state) and 0 <= x < len(room_state[0])

        player_position = np.where(np.logical_or(room_state == RoomState.PLAYER.value, room_state == RoomState.PLAYER_ON_GOAL.value))
        sensible_actions = []
        for y, x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_y, new_x = player_position[0] + y, player_position[1] + x
            # Make sure new move is inbound
            if not inbound(new_y, new_x):
                continue
            # Check if move is to empty space or pushing of box
            if room_state[new_y, new_x] in [RoomState.EMPTY.value, RoomState.GOAL.value]:
                sensible_actions.append(SensibleMove(x, y, False))
            elif room_state[new_y, new_x] in [RoomState.BOX.value, RoomState.BOX_ON_GOAL.value]:
                # Make sure we can push the box
                new_box_y, new_box_x = new_y + y, new_x + x

                # Make sure push of box is inbound
                if not inbound(new_box_y, new_box_x):
                    continue

                if room_state[new_box_y, new_box_x] in [RoomState.GOAL.value, RoomState.EMPTY.value]:
                    sensible_actions.append(SensibleMove(x, y, True))
        return sensible_actions

class SensibleMove:
    def __init__(self, x, y, move_box):
        self.x = x
        self.y = y
        self.move_box = move_box

    def __repr__(self):
        return f"SensibleMove(x={self.x}, y={self.y}, move_box={self.move_box})"

    def translate_action(self) -> int:
        """
        Translate the move to the action that the environment understands
        """
        if self.x == 0 and self.y == -1:
            return 1
        elif self.x == 0 and self.y == 1:
            return 2
        elif self.x == -1 and self.y == 0:
            return 3
        elif self.x == 1 and self.y == 0:
            return 4
        raise ValueError("Invalid move")
