from typing import Tuple
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Box
from .room_utils import generate_fixed_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
from cycle_detection import CycleDetection
import numpy as np
import pygame

class SokobanEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 original_map,
                 render_mode='human',
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 player_position=(0, 0),
                 step_penalty: float=0.1,
                 reset=True):

        # General Configuration
        self.original_map = original_map
        self.render_mode = render_mode
        self.dim_room = dim_room
        self.penalty_for_step = -step_penalty
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0
        self.player_position = (0, 0)

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.clock = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

        if reset:
            # Initialize Room
            _ = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='human'):
        assert action in ACTION_LOOKUP

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
        return observation, self.reward_last, done, False, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target

        game_won = self._check_if_all_boxes_on_target()
        if game_won:
            self.reward_last += self.reward_finished

        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def _get_obs(self):
        return np.array(self.room_state)

    def _get_info(self):
        return {
            "distance": 0
        }

    def render(self, mode='human', scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            if self.viewer is None:
                pygame.display.init()
                pygame.display.set_caption('Sokoban')
                self.viewer = pygame.display.set_mode((img.shape[1], img.shape[0]))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.viewer.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
            pygame.event.pump()
            self.clock.tick(30)
            pygame.display.flip()
            return True

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render()  # just raise an exception


    def reset(self, seed=None, options={}, second_player=False, render_mode='rgb_array'):
        super().reset(seed=seed, options=options)
        try:
            self.room_fixed, self.room_state, self.box_mapping, self.player_position = generate_fixed_room(self.original_map)
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(seed=seed, options=options, second_player=second_player, render_mode=render_mode)

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.cycle_detection = CycleDetection(self.room_state)

        self.render(self.render_mode)

        starting_observation = self._get_obs()
        info = self._get_info()
        return starting_observation, info

    def get_image(self, mode, scale=1):

        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            pygame.display.quit()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
