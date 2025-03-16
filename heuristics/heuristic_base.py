from abc import ABC, abstractmethod

class Heuristic(ABC):
    @abstractmethod
    def run(self, room_state, env):
        return 0.0
