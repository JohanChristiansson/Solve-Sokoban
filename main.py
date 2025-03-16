from mcts import MCTS
from utils import parse
import gymnasium as gym
import gym_sokoban
import argparse
import json
import torch
from models.alphazero import AlphaZero
from mcts import MCTS
from time import time
from pathlib import Path
from types import SimpleNamespace
from heuristics.heuristic_base import Heuristic
from heuristics.heuristic_impl import ManhattanBoxGoal, ManhattanBoxPlayer, BoxesOnGoal, AlphaZeroHeuristic, NeighboringBoxes
from astar import AStar

LEGAL_ACTIONS = [1,2,3,4]

ACTION_MAP = {
    'push up': "U",
    'push down': "D",
    'push left': "L",
    'push right': "R",
}

def write_solution(args, file, actions):
    """
    Write the actions taken to solve board into log file.
    """
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "{}.log".format(file.stem), mode="w") as log:
        print("{}".format(len(actions)), file=log, end="")
        for action in actions:
            print(" {}".format(action), file=log, end="")

def extract_heuristics(args):
    """
    Maps different huristics with their correcponding implementations and weights.
    """
    def get_heuristic(heuristic_name: str) -> Heuristic:
        if heuristic_name == "manhattan_box_player":
            heuristic = ManhattanBoxPlayer()
        elif heuristic_name == "boxes_on_goal":
            heuristic = BoxesOnGoal()
        elif heuristic_name == "manhattan_box_goal":
            heuristic = ManhattanBoxGoal()
        elif heuristic_name == "alpha_zero":
            heuristic = AlphaZeroHeuristic()
        elif heuristic_name == "neighboring_boxes":
            heuristic = NeighboringBoxes()
        else:
            # Default to Manhattan Heuristic
            print(f"Invalid heuristic name: {heuristic_name}. Defaulting to ManhattanBoxGoal")
            heuristic = ManhattanBoxGoal()
        return heuristic

    if not args.heuristic:
        args.heuristic = [["manhattan_box_goal", 1]]

    heuristics_with_weights = []
    for heuristic_name, weight in args.heuristic:
        heuristic = get_heuristic(heuristic_name)
        try:
            heuristics_with_weights.append((heuristic, float(weight)))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid weight value: {weight} for heuristic: {heuristic_name}")

    return heuristics_with_weights

def create_env(args, file) -> gym.Env:
    """
    Creating and initilalizing sokoban environment with supplied arguments.
    """
    if args.render_mode == "raw":
        observation_mode = "raw"
    elif "tiny" in args.render_mode:
        observation_mode = "tiny_rgb_array"
    elif args.render_mode == "human":
        observation_mode = "human"
    else:
        observation_mode = "rgb_array"
    dim_room, n_boxes, map = parse(filename=file)
    env = gym.make("Sokoban-v1",
                   original_map=map,
                   dim_room=dim_room,
                   num_boxes=n_boxes,
                   max_steps=args.max_steps,
                   render_mode=observation_mode,
                   step_penalty=0
                   )
    env = env.unwrapped
    env.reset()

    return env

def mcts_solve(args, file):
    """
    Tries to solve a Sokoban board using MCTS and trains an Alphazero Model if training is enabled.
    """
    env = create_env(args, file)
    dim_room, _, _ = parse(filename=file)
    heuristics_with_weights = extract_heuristics(args)

    exploration = args.exploration
    step_penalty = args.step_penalty
    depth_penalty = args.depth_penalty
    johgust_score = args.johgust_score

    actions = []
 
    # Create trainer if in training mode with given arguments
    train_model: bool = (args.train_time != 0)
    embedding_dim: int = args.embedding_dim
    lr: float = args.lr
    betas: tuple[float, float] = tuple(args.betas)
    weight_decay: float = args.weight_decay
    neural_network = AlphaZero(dim_room, embedding_dim=embedding_dim, lr=lr, betas=betas, weight_decay=weight_decay) if train_model else None

    solver = MCTS(env=env,
                  heuristics_with_weights=heuristics_with_weights,
                  neural_network=neural_network,
                  train_model=train_model,
                  n_playout=args.max_rollouts,
                  max_depth=args.max_depth,
                  exploration=exploration,
                  depth_penalty=depth_penalty,
                  eval_mode=args.eval_mode,
                  johgust_score = johgust_score
                  )

    allocated_time = args.time_limit * 60
    training_time = args.train_time * 60
    start_time = time()
    finished = False
    while not finished:
        now = time()
        elapsed_time = now - start_time
        if train_model and (elapsed_time >= training_time):
            print("Saving model, please wait...")
            torch.save(solver.neural_network, "models/trained_model")
            print("Model saved to 'models/trained_model'")
            break
        if elapsed_time > allocated_time:
            break
        for _, _, done, info in solver.take_best_actions():
            if "action.name" in info:
                actions.append(ACTION_MAP[info["action.name"]])
            if done and "mcts_giveup" in info:
                env.reset()
                actions.clear()
                break
            elif done:
                if info["maxsteps_used"]:
                    env.reset()
                    actions.clear()
                else:
                    if train_model:
                        print("Saving model, please wait...")
                        torch.save(solver.neural_network, "models/trained_model")
                        print("Model saved to 'models/trained_model'")
                        print("Solved in {:.0f} seconds".format((now - start_time)))
                    actions.append("Solved in {:.0f} seconds".format((now - start_time)))
                    finished = True
    env.close()
    write_solution(args, file, actions)

def astar_solve(args, file):
    """
    Solve a sokoban board using A* search.
    """
    env = create_env(args, file)
    heuristics_with_weights = extract_heuristics(args)

    print("Solving with AStar")
    astar = AStar(env.room_state, heuristics_with_weights, env)
    start_time = time()
    solution = astar.search()
    if not solution:
        print("No solution found")
        env.close()
        return

    # A solution is found
    print("Solved in {:.0f} seconds".format((time() - start_time)))
    for action in solution:
        env.step(action, observation_mode=str(env.render_mode))

    env.close()
    write_solution(args, file, solution)

def main(args):
    """
    Entry point for sokoban solver.
    """
    if args.config:
        config_data = json.load(open(args.config))
        args = SimpleNamespace(**config_data)

    if args.file:
        for file in args.file:
            if args.eval_mode:
                # Use A* with alpha zero heuristic
                astar_solve(args, Path(file))
            else:
                mcts_solve(args, Path(file))
    else:
        for file in Path(args.folder).iterdir():
            if args.eval_mode:
                # Use A* with alpha zero heuristic
                astar_solve(args, file)
            else:
                mcts_solve(args, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", nargs = "+", help= "File that defines the sokoban map")
    group.add_argument("--folder", help= "Folder that contains files which define the sokoban map")
    group.add_argument("--config", help= "Configuration (.json) file that sets parameters")
    parser.add_argument("--render_mode", help="Obversation mode for the game. Use human to see a render on the screen", default="raw")
    parser.add_argument("--heuristic", help="Heuristic(s) to use for the MCTS together with their weights", nargs=2, action="append", metavar=("HEURISTIC_NAME", "WEIGHT"), default=[])
    parser.add_argument("--max_rollouts", type=int, help="Number of rollouts per move", default=100)
    parser.add_argument("--max_depth", type=int, help="Depth of each rollout", default=30)
    parser.add_argument("--max_steps", type=int, help="Max moves before game is lost", default=120)
    parser.add_argument("--time_limit", type=int, help="Allocated time (in minutes) per board", default=60)
    parser.add_argument("--log_dir", type=str, help="Directory to log solve information", default="./solve_log")
    parser.add_argument("--exploration", type=float, help="Exploration for UCB, The square root is applied to the input value", default=2)
    parser.add_argument("--step_penalty", type=float, help="Penalty for each step", default=0.1)
    parser.add_argument("--depth_penalty", type=float, help="Penalty for depth in each simulation", default=0.01)
    parser.add_argument("--train_time", type=int, help="Train AlphaZero model for given time (in minutes) then save the model. Time of 0 represents training turned off", default=0)
    parser.add_argument("--eval_mode", type=bool, help="Turn off MCTS and choose moves based on heuristic", default=False)
    parser.add_argument("--lr", type=float, help="Learning rate for the AlphaZero trainer", default=0.001)
    parser.add_argument("--betas", type=float, help="Beta values (lower and upper bound, between 0 and 1) for the AlphaZero trainer", nargs=2, default=[0.9,0.999])
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the AlphaZero trainer", default=0.01)
    parser.add_argument("--embedding_dim", type=int, help="Embedding dimension for the AlphaZero trainer", default=16)
    parser.add_argument("--johgust_score", type=bool, help="Use Johan and Augusts improved score", default = True)
    args = parser.parse_args()
    main(args)

