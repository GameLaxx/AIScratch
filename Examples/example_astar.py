from AIScratch.Astar import HeuristicDomain, Astar, Action, State, Renderable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 

class MazeState(State):
    def __init__(self, x, y, depth = 0):
        self.x = x
        self.y = y
        self.depth = depth

    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        # # return (self.x - other.x) ** 2 + (self.y - other.y) ** 2
        # return min(abs(self.x - other.x), abs(self.y - other.y))
    
    def __eq__(self, value):
        return self.x == value.x and self.y == value.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"S({self.x}, {self.y})"

class MazeAction(Action):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def __repr__(self):
        return f"A({self.dx}, {self.dy})"

class MazeDomain(HeuristicDomain, Renderable):
    def __init__(self, start : MazeState, goal : MazeState, maze):
        self.start = start
        self.goal = goal
        self.maze = maze

    def get_initial_state(self):
        return self.start
    
    def get_goal_states(self):
        return [self.goal]
    
    def is_goal(self, state):
        return state == self.goal
    
    def is_terminal(self, state):
        return self.is_goal(state)

    def generate_actions(self, state : MazeState):
        # top is bit 1, right is bit 2, bottom is bit 3 and left is bit 4 (1 means wall)
        maze_value = int(self.maze[state.y][state.x])
        ret = []
        if not maze_value & 1:
            ret.append(MazeAction(0,-1))
        if not maze_value & 2:
            ret.append(MazeAction(1,0))
        if not maze_value & 4:
            ret.append(MazeAction(0,1))
        if not maze_value & 8:
            ret.append(MazeAction(-1,0))
        return ret
    
    def generate_state(self, current_state, action):
        return MazeState(current_state.x + action.dx, current_state.y + action.dy)
    
    def get_transition_value(self, current_state, action, next_state):
        if current_state == next_state:
            return 50
        return 1
    
    def get_heuristic_value(self, current_state : MazeState, action, next_state):
        return current_state.distance(self.goal)
    
    def render(self):
        fig, ax = plt.subplots(figsize=self.maze.shape)
        ax.set_xlim(0, self.maze.shape[0])
        ax.set_ylim(0, self.maze.shape[1])
        ax.add_patch(patches.Rectangle((self.start.x, self.start.y), 1, 1, facecolor='green'))
        ax.add_patch(patches.Rectangle((self.goal.x, self.goal.y), 1, 1, facecolor='red'))
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                maze_value = int(self.maze[i][j])
                if maze_value & 1:
                    ax.plot([j, j + 1], [i, i], "k", linewidth=2)
                if maze_value & 2:
                    ax.plot([j + 1, j + 1], [i, i + 1], "k", linewidth=2)
                if maze_value & 4:
                    ax.plot([j, j + 1], [i + 1, i + 1], "k", linewidth=2)
                if maze_value & 8:
                    ax.plot([j, j], [i, i + 1], "k", linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

domain = MazeDomain(MazeState(0,0), MazeState(5,5), np.zeros((6,6)))
astar = Astar(domain)
for i in range(6):
    domain.maze[0][i] += 1
    domain.maze[i][0] += 8
    domain.maze[5][i] += 4
    domain.maze[i][5] += 2
domain.render()
print(astar.solve())