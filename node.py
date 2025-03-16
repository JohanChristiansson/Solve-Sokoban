from typing import Any, Union
import numpy as np # type: ignore
from anytree import NodeMixin # type: ignore

class Node(NodeMixin):
    """
    Represent a node in tree structure of MCTS.
    """
    def __init__(
        self,
        name: str,
        state: np.ndarray,
        done: bool = False,
        actions: Union[list[Any], None] = None,
        parent: Union["Node", None] = None,
        children: list["Node"] = []
    ):
        super(Node, self).__init__()
        self.name = name
        self.state = state
        self.score: float = 1.0
        self.total_score: float = 0.0
        self.steps: int = 0
        self.visits: int = 1
        self.done = done
        self.parent = parent
        self.actions = actions
        if children:
            self.children = children

