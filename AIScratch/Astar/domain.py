from abc import ABC, abstractmethod
from AIScratch.Astar import State, Action

class Domain(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_initial_state(self) -> State:
        pass

    @abstractmethod
    def get_goal_states(self) -> list[State]:
        pass

    @abstractmethod
    def is_goal(self, state : State) -> bool:
        pass

    @abstractmethod
    def is_terminal(self, state : State) -> bool:
        pass

    @abstractmethod
    def generate_actions(self, state : State) -> list[Action]:
        pass

    @abstractmethod
    def generate_state(self, current_state : State, action : Action) -> State:
        pass

    @abstractmethod
    def get_transition_value(self, current_state : State, action : Action, next_state : State) -> float:
        pass

class HeuristicDomain(Domain):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_heuristic_value(self, current_state : State, action : Action, next_state : State) -> float:
        pass

class Renderable(ABC):
    @abstractmethod
    def render(self):
        pass