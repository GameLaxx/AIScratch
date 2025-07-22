import itertools
from AIScratch.Astar import HeuristicDomain
import heapq

class Astar():
    def __init__(self, domain : HeuristicDomain):
        self.domain = domain

    def solve(self):
        start = self.domain.get_initial_state()
        g_score = {start: 0}
        came_from = {}
        open_set = []
        visited = set()
        
        counter = itertools.count()
        
        f_start = self.domain.get_heuristic_value(start, None, start)
        heapq.heappush(open_set, (f_start, next(counter), start))
        
        while open_set:
            _, _, current = heapq.heappop(open_set)

            if self.domain.is_goal(current):
                return self.reconstruct_path(came_from, current)

            if self.domain.is_terminal(current):
                continue

            visited.add(current)

            for action in self.domain.generate_actions(current):
                neighbor = self.domain.generate_state(current, action)
                if neighbor in visited:
                    continue

                tentative_g = g_score[current] + self.domain.get_transition_value(current, action, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.domain.get_heuristic_value(current, action, neighbor)
                    heapq.heappush(open_set, (f_score, next(counter), neighbor)) 
        return []
    
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path