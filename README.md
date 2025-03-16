# Solve Sokoban with AlphaZero

This project aims to solve the Sokoban game using the AlphaZero-inspired model. The project is part of the course TDDE19 at Linköping University completed during Autumn 2024.

## Installation instruction

Follow the steps below to install the project and all its dependencies.

### Prerequisites

- Python >=3.9,<3.12
- 'pip' (Python package installer)

### Step 1: Clone the repository

First, clone the repository to your local machine using the following command:
```bash
git clone https://gitlab.liu.se/TDDE19_teachers/2024/solve-sokoban-with-alphazero.git
cd solve-sokoban-with-alphazero
```
### Step 2: Create a virtual environment

It is recommended to create a virtual environment to install the project dependencies. To create a virtual environment, run the following command:
```bash
# For macOS/Linux
python3 -m venv venv

# For Windows
python -m venv venv
```
Then, activate the virtual environment:
```bash
# For macOS/Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

### Step 3: Install dependencies

Install the required dependencies using the following command:
```bash
pip install -e .
```

## Files

The most important files are:
- main.py
- mcts.py
- models/alphazero.py
- heuristics/heuristic\_impl.py
- cycle_detection.py
- gym\_sokoban/envs/sokoban\_env.py

## Running instruction

For running the project, use the following command:
```bash
python3 main.py --file sokoban_boards/sokoban01.txt --max_rollouts 100
```
More boards can be found in the sokoban_boards/ directory.


If you want a graphical representation while it's running, add the following flag:
```bash
--render_mode human
```

Training of a model is done for a set amount of time, specified by the user. To train the model for 25 minutes, add the flag:
```bash
--train_time 25
```

There are also flags for changing certain network parameters, heuristics, and more! To see what arguments are available, run:
```bash
python3 main.py --help
``` 

Instead of specifying flags and parameters as command line arguments, one can specify them using a .json config file. Some examples can be found in the configs/ directory. Run using the --config flag:
```bash
python3 main.py --config configs/default.json
``` 

## License
The project is licensed under the MIT license (https://opensource.org/license/mit).
