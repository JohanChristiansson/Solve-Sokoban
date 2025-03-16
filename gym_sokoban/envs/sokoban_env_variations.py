from .sokoban_env import SokobanEnv

class SokobanEnv1(SokobanEnv):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array'],
    }

    def __init__(self):
        super(SokobanEnv1, self).__init__(
            original_map=None, render_mode="rgb_array", num_boxes=3, max_steps=200
        )

