from constants import RoomState
from heuristics.heuristic_base import Heuristic
import numpy as np
import torch

class ManhattanBoxGoal(Heuristic):
    def run(self, room_state, env):
        """
        Calculates the manhattan distance between the boxes and the closest goal square
        """
        room_state = np.asarray(room_state)

        box_positions = np.where((room_state == RoomState.BOX.value))
        goal_positions = np.where(env.room_fixed == RoomState.GOAL.value)

        # Positions are in the format (x, y)
        goal_x, goal_y = goal_positions
        box_x, box_y = box_positions

        # Find the closest goal for each box
        distances = np.abs(goal_x[:, None] - box_x) + np.abs(goal_y[:, None] - box_y)

        # Find the minimum distance for each box
        min_distances = np.min(distances, axis=0)

        return float(np.sum(min_distances))

class ManhattanBoxPlayer(Heuristic):
    def run(self, room_state, env):
        """
        Calculates the mean manhattan distance between the player position and the boxes
        """
        room_state = np.asarray(room_state)

        box_positions = np.where(room_state == RoomState.BOX.value)
        player_position = np.where(np.logical_or(room_state == RoomState.PLAYER.value, room_state == RoomState.PLAYER_ON_GOAL.value))

        x_dist = abs(box_positions[0] - player_position[0])
        y_dist = abs(box_positions[1] - player_position[1])

        distances = x_dist + y_dist

        return float(np.mean(distances))
    
class BoxesOnGoal(Heuristic):
    def run(self, room_state, env):
        """
        Calculates the amount of boxes that are not on a goal square
        """
        room_state = np.asarray(room_state)
        return np.count_nonzero(room_state == RoomState.BOX.value)

class AlphaZeroHeuristic(Heuristic):
    def __init__(self, file_path="models/trained_model"):
        self.model = torch.load(file_path, weights_only=False)
        self.model.eval()

    def run(self, room_state, env):
        return self.model.forward(room_state, [0])[1].squeeze().item()

class NeighboringBoxes(Heuristic):
    def run(self, room_state, env):
        """
        Calculates the adjacency (direct neighboring) of boxes to each other, as well as to walls and boxes on goal.
        This heuristic essentially penalizes boxes that are not able to move freely.
        """
        room_state = np.asarray(room_state)

        box_positions = np.where(room_state == RoomState.BOX.value)
        wall_and_goal_box_positions = np.where(np.logical_or(room_state == RoomState.WALL.value, room_state == RoomState.BOX_ON_GOAL.value))

        # Positions are in the format (y, x)
        box_y, box_x = box_positions
        wall_and_goal_box_y, wall_and_goal_box_x = wall_and_goal_box_positions

        # Get distances
        dist_between_boxes = np.abs(box_x[:, None] - box_x) + np.abs(box_y[:, None] - box_y)
        dist_between_boxes_and_walls = np.abs(box_x[:, None] - wall_and_goal_box_x) + np.abs(box_y[:, None] - wall_and_goal_box_y)

        # Get number of neighboring
        num_neighboring_boxes = np.sum(dist_between_boxes == 1) // 2
        num_neighboring_boxes_and_walls = np.sum(dist_between_boxes_and_walls == 1)

        return (num_neighboring_boxes + num_neighboring_boxes_and_walls)

