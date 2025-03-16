import copy
import random
import torch.nn # type: ignore
import numpy as np # type: ignore
from typing import Union, List, Tuple
from utils import hashable
from gym_sokoban.envs.sokoban_env import SokobanEnv
from heuristics.heuristic_base import Heuristic
from cycle_detection import CycleDetection

from node import Node
from constants import RoomState

class ActionTracker:
    def __init__(self, parent: Union["ActionTracker", None] = None):
        self.parent = parent
        self.performed_actions: list["SensibleMove"] = []
    
    def add_action(self, sensible_action: "SensibleMove"):
        self.performed_actions.append(sensible_action)

class MCTS:
    def __init__(self,
                 env,
                 heuristics_with_weights: List[Tuple[Heuristic, float]],
                 neural_network: torch.nn.Module,
                 train_model: bool,
                 n_playout: int,
                 max_depth: int,
                 exploration: float,
                 depth_penalty: float,
                 eval_mode: bool,
                 johgust_score: bool):

        self.env: SokobanEnv = env
        self.heuristics_with_weights = heuristics_with_weights
        self.neural_network = neural_network
        self.train_model = train_model
        self.n_playout = n_playout
        self.max_depth = max_depth
        self.exploration = np.sqrt(exploration)
        self.depth_penalty = depth_penalty
        self.eval_mode = eval_mode
        self.current_tracker = ActionTracker()
        self.johgust_score = johgust_score

    def take_best_actions(self):
        """
        Take the best action according to the MCTS algorithm
        """
        best_actions = self.mcts()

        if best_actions is None:
            # Unsolvable board, go back one step
            parent_tracker = self.current_tracker.parent
            if parent_tracker is None:
                self.current_tracker = ActionTracker()
                yield None, -1, True, {"mcts_giveup": "MCTS Gave up, board unsolvable. Reset board"}
            else:
                previous_move = parent_tracker.performed_actions[-1]
                reversed_move = previous_move.get_reversed_move().translate_action()
                observation, reward, _, _, info = self.env.step(reversed_move, observation_mode=str(self.env.render_mode))
                self.current_tracker = parent_tracker
                yield observation, reward, self.is_done(self.env.room_state), info
        else:
            for action in best_actions:
                tracker = ActionTracker(self.current_tracker)
                self.current_tracker.add_action(action)
                self.current_tracker = tracker
                
                move = action.translate_action()
                observation, reward, _, _, info = self.env.step(move, observation_mode=str(self.env.render_mode))
                self.env.cycle_detection.push_state(self.env.room_state)
                yield observation, reward, self.is_done(self.env.room_state), info


    def select_best_child(self, children: list[Node]) -> Node:
        """
        Selects the child that achives the highest score on fewest steps
        """
        # Select children with the highest score
        if self.johgust_score:
            max_score = max(child.score for child in children)
            highest_scoring_children = [child for child in children if child.score == max_score]

            # Select children with the lowest number of steps among the best children
            min_steps = min(child.steps for child in highest_scoring_children)
            best_children =  [child for child in highest_scoring_children if child.steps == min_steps]
        else:
            max_visits = max(child.visits for child in children)
            best_children = [child for child in children if child.visits == max_visits]

        # Select a random child if multiple are just as good to avoid prioritizing certain actions
        best_child = random.choice(best_children)
        
        return best_child

    def mcts(self):
        """
        Run the Monte Carlo Tree Search algorithm
        """
        root = Node("root", self.env.room_state)
        # if self.env.render_mode == "human":
        for _ in range(self.n_playout):
            child, playout_score = self.playout(root)
            if playout_score == np.inf:
                return child.actions
            if child.parent is None:
                return None
            self.backpropagate(child, playout_score)
        state = self.env.room_state
        player_position = np.where(np.logical_or(state == RoomState.PLAYER.value, state == RoomState.PLAYER_ON_GOAL.value))

        sensible_children: list[Node] = [] # list of children not resulting in cycle
        for child in root.children:
            new_player_position_x = player_position[1] + child.actions[0].x
            new_player_position_y = player_position[0] + child.actions[0].y
            # Cycle detection, don't add child (move) that causes cycle
            if self.env.cycle_detection.is_cycle(self.env.room_state, new_player_position_x, new_player_position_y):
                continue
            sensible_children.append(child)
        # No children without cycles
        if not sensible_children:
            return None

        best_child = self.select_best_child(sensible_children)
        return best_child.actions

    def playout(self, node: Node) -> tuple[Node, float]:
        """
        Run a playout from a node
        """
        child, _, path = self.select(node)
        playout_score = self.simulate(child, path)
        return child, playout_score

    def backpropagate(self, node: Node, score: float):
        """
        Backpropagate the best score to the parent nodes
        """

        current: Union[Node, None] = node
        steps = 0
        while current is not None:
            current.visits += 1
            current.total_score += score

            # This assumes that current.score can never be 1.0 from the heuristic values.
            # Which it can't as we use negative scores and the best value is 0 (or np.inf)
            if current.score == 1.0 or score > current.score: 
                current.score = score
                current.steps = steps
            elif score == current.score and steps < current.steps:
                current.steps = steps

            steps += 1
            current = current.parent

    def select(self, root: Node) -> tuple[Node, float, list["SensibleMove"]]:
        """
        Select the best child node to explore
        """
        tree = root
        path: list["SensibleMove"] = []
        while not tree.done:
            sensible_actions = self.sensible_actions(tree.state)
            if not sensible_actions:
                # The player is stuck
                return tree, -np.inf, path + tree.actions if tree.actions else []
            if len(tree.children) < len(sensible_actions):
                # Expand the tree since we have not visited all the children
                return self.expand(tree, sensible_actions, path)
            # Select the child with the highest UCT score
            tree = self.uct_select(tree)
            if tree.actions:
                path += tree.actions
        if self.is_done(tree.state):
            return tree, np.inf, path + tree.actions if tree.actions else []
        return tree, 0, path

    def uct_select(self, node: "Node") -> "Node":
        """
        Select the child node with the highest UCT score
        """
        def uct(parent, child):
            if self.johgust_score:
                return child.score - child.steps* self.depth_penalty + np.sqrt(self.exploration * np.log(parent.visits) / child.visits)
            else:
                return child.total_score / child.visits + self.exploration * np.log(parent.visits) / child.visits
        best_child = max(node.children, key=lambda child: uct(node, child))
        return best_child

    def expand(self, node: "Node", sensible_actions: list["SensibleMove"], path: list["SensibleMove"]) -> tuple[Node, float, list["SensibleMove"]]:
        """
        Expand the tree by adding a child nodes for each valid move
        """
        for move in sensible_actions:
            new_state = copy.deepcopy(node.state)
            self.simulate_move(new_state, move)
            Node(node.name + "(" + str(move.x) + "," + str(move.y) + ")", new_state, parent=node, actions=[move])
        choice = random.choice(node.children)
        return choice, choice.score, path + choice.actions if choice.actions else []

    def convert_position_to_tuple(self, position: tuple[np.ndarray, np.ndarray]):
        y, x = position[0], position[1]
        return tuple([pos for pos in zip(y, x)])

    def calculate_heuristic(self, state: np.ndarray) -> float:
        heuristic_value = 0
        for heuristic, weight in self.heuristics_with_weights:
            current_value = heuristic.run(state, self.env)
            heuristic_value += current_value * weight * self.env.penalty_for_step
        return heuristic_value

    def simulate(self, node: "Node", path: list["SensibleMove"]) -> float:
        """
        Simulate the game until it is done or the max depth is reached
        """
        highest_found_heuristic = self.calculate_heuristic(node.state)
        depth_of_best_heuristic = 0

        room_states = [node.state]
        training_heuristics = [self.calculate_heuristic(node.state)]
        training_depths = [0]

        training_room_states = [node.state]

        pure_heuristics = {hashable(node.state): self.calculate_heuristic(node.state)}

        # Copy the current state
        room_state = copy.deepcopy(node.state)
        steps_taken: List["SensibleMove"] = []

        cycle_detection = CycleDetection(room_state)

        for current_depth in range(1, self.max_depth + 1): # Since we start at depth 1

            if self.is_done(room_state):
                node.actions = path + steps_taken
                return np.inf
            sensible_actions = self.sensible_actions(room_state, cycle_detection)
            if not sensible_actions:
                # The player is stuck, end simulation
                break
            move = random.choice(sensible_actions)
            self.simulate_move(room_state, move)
            steps_taken.append(move)

            cycle_detection.push_state(room_state)
            current_heuristic = self.calculate_heuristic(room_state)

            if highest_found_heuristic < current_heuristic:
                highest_found_heuristic = current_heuristic
                depth_of_best_heuristic = current_depth

            pure_heuristics[hashable(room_state)] = current_heuristic

            
            if self.train_model:
                for depth, parent in enumerate(room_states):
                    if current_heuristic < pure_heuristics[hashable(parent)]:
                        training_room_states.append(parent)
                        training_heuristics.append(current_heuristic)
                        training_depths.append(current_depth - depth)

                room_states.append(room_state)
                training_room_states.append(room_state)
                training_heuristics.append(current_heuristic)
                training_depths.append(0)

        if self.train_model:
            self.neural_network.train_model(training_room_states, training_depths, training_heuristics)
        return highest_found_heuristic

    def is_done(self, state):
        """
        Check if the game is done
        """
        return not np.any(state == RoomState.BOX.value)

    def simulate_move(self, state: np.ndarray, move: "SensibleMove"):
        """
        Simulate the move in the given state
        """
        player_position = np.where(np.logical_or(state == RoomState.PLAYER.value, state == RoomState.PLAYER_ON_GOAL.value))
        if move.move_box:
            # Update old box position state values
            box_position = (player_position[0] + move.y, player_position[1] + move.x)
            if state[box_position] == RoomState.BOX_ON_GOAL.value:
                state[box_position] = RoomState.GOAL.value
            else:
                state[box_position] = RoomState.EMPTY.value
            # Update new box position state values
            new_box_position = (box_position[0] + move.y, box_position[1] + move.x)
            if state[new_box_position] == RoomState.GOAL.value:
                state[new_box_position] = RoomState.BOX_ON_GOAL.value
            else:
                state[new_box_position] = RoomState.BOX.value
        # Update old player position state values
        if state[player_position] == RoomState.PLAYER_ON_GOAL.value:
            state[player_position] = RoomState.GOAL.value
        else:
            state[player_position] = RoomState.EMPTY.value
        # Update new player position state values
        new_player_position = (player_position[0] + move.y, player_position[1] + move.x)
        if state[new_player_position] == RoomState.GOAL.value:
            state[new_player_position] = RoomState.PLAYER_ON_GOAL.value
        else:
            state[new_player_position] = RoomState.PLAYER.value

    def sensible_actions(
            self,
            room_state: np.ndarray,
            cycle_detection: Union[CycleDetection, None] = None
        ) -> list["SensibleMove"]:
        """
        Returns a list of valid moves that the player can make
        """
        def inbound(y : int, x : int) -> bool:
            """
            Check if the position is inbound
            """
            return 0 <= y < len(room_state) and 0 <= x < len(room_state[0])

        def is_in_corner(box_position_y: int, box_position_x: int) -> bool:
            """
            Checks if a box is in a corner. A box is in a corner if there is a wall in front
            of the box and a wall to the side of the box
            """
            is_wall_above = room_state[box_position_y - 1, box_position_x] == RoomState.WALL.value
            is_wall_below = room_state[box_position_y + 1, box_position_x] == RoomState.WALL.value
            is_wall_to_left = room_state[box_position_y, box_position_x - 1] == RoomState.WALL.value
            is_wall_to_right = room_state[box_position_y, box_position_x + 1] == RoomState.WALL.value
            # Case 1: Upper right corner
            if is_wall_to_right and is_wall_above:
                return True
            # Case 2: Lower right corner
            if is_wall_to_right and is_wall_below:
                return True
            # Case 3: Lower left corner
            if is_wall_to_left and is_wall_below:
                return True
            # Case 4: Upper left corner
            if is_wall_to_left and is_wall_above:
                return True
            return False

        def is_box_deadlock(box_position_y: int, box_position_x: int, board: np.ndarray=room_state) -> bool:
            """
            Checks if a box is in a deadlock with another box. Two boxes are in a deadlock if they are
            horizontally or vertically aligned and there is a wall in front of both boxes
            """
            possible_neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dangerous_neighbours = []

            is_wall_above = board[box_position_y - 1, box_position_x] == RoomState.WALL.value
            is_wall_below = board[box_position_y + 1, box_position_x] == RoomState.WALL.value
            is_wall_to_right = board[box_position_y, box_position_x + 1] == RoomState.WALL.value
            is_wall_to_left = board[box_position_y, box_position_x - 1] == RoomState.WALL.value

            box_values = [RoomState.BOX.value, RoomState.BOX_ON_GOAL.value]

            for y_offset, x_offset in possible_neighbours:
                if not inbound(box_position_y + y_offset, box_position_x + x_offset):
                    continue
                neighbour_value = board[box_position_y + y_offset, box_position_x + x_offset]
                if neighbour_value not in box_values:
                    continue

                if y_offset == 0 and (is_wall_above or is_wall_below):
                    dangerous_neighbours.append((box_position_y + y_offset, box_position_x + x_offset))
                elif x_offset == 0 and (is_wall_to_right or is_wall_to_left):
                    dangerous_neighbours.append((box_position_y + y_offset, box_position_x + x_offset))

            for neighbour_y, neighbour_x in dangerous_neighbours:
                if neighbour_y == box_position_y:
                    is_wall_above_neighbour = board[neighbour_y - 1, neighbour_x] == RoomState.WALL.value
                    is_wall_below_neighbour = board[neighbour_y + 1, neighbour_x] == RoomState.WALL.value
                    if is_wall_above_neighbour or is_wall_below_neighbour:
                        return True
                elif neighbour_x == box_position_x:
                    is_wall_to_right_of_neighbour = board[neighbour_y, neighbour_x + 1] == RoomState.WALL.value
                    is_wall_to_left_of_neighbour = board[neighbour_y, neighbour_x - 1] == RoomState.WALL.value
                    if is_wall_to_right_of_neighbour or is_wall_to_left_of_neighbour:
                        return True
            return False

        def in_tracker_position(move: "SensibleMove") -> bool:
            """
            Checks if a move is in the current tracker.
            """
            for tracker_move in self.current_tracker.performed_actions:
                if tracker_move.x == move.x and tracker_move.y == move.y:
                    return True
            return False


        player_position = np.where(np.logical_or(room_state == RoomState.PLAYER.value, room_state == RoomState.PLAYER_ON_GOAL.value))
        sensible_actions = []
        for y, x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            current_move = SensibleMove(x, y)
            new_y, new_x = int(player_position[0]) + y, int(player_position[1]) + x
            # Tracker
            if in_tracker_position(current_move):
                continue
            # Make sure new move is inbound
            if not inbound(new_y, new_x):
                continue
            # Cycle detection for current board state
            if cycle_detection is not None and cycle_detection.is_cycle(room_state, new_x, new_y):
                continue
            # Check if move is to empty space or pushing of box
            if room_state[new_y, new_x] in [RoomState.EMPTY.value, RoomState.GOAL.value]:
                current_move.move_box = False
                sensible_actions.append(current_move)
            elif room_state[new_y, new_x] in [RoomState.BOX.value, RoomState.BOX_ON_GOAL.value]:
                current_move.move_box = True
                # Make sure we can push the box
                new_box_y, new_box_x = new_y + y, new_x + x

                # Make sure push of box is inbound
                if not inbound(new_box_y, new_box_x):
                    continue

                # Check if box is pushed to goal or empty space
                if room_state[new_box_y, new_box_x] == RoomState.GOAL.value:
                    sensible_actions.append(current_move)
                elif room_state[new_box_y, new_box_x] == RoomState.EMPTY.value:
                    # Prevent box from being pushed into a (empty space) corner,
                    # which results in unwinnable state
                    if is_in_corner(new_box_y, new_box_x):
                        continue
                    # Create a copy of the board where the box is moved
                    new_board = copy.deepcopy(room_state)
                    self.simulate_move(new_board, current_move)
                    if is_box_deadlock(new_box_y, new_box_x, new_board):
                        continue
                    sensible_actions.append(current_move)
        return sensible_actions

class SensibleMove:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.move_box = None

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
    
    def get_reversed_move(self):
        reversed_x = -self.x
        reversed_y = -self.y
        return SensibleMove(reversed_x, reversed_y)
