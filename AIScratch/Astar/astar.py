from AIScratch.Astar import HeuristicDomain

class Node():
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList():
    def __init__(self):
        self.head = None

    def add(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node 
        
    def pop(self):
        if not self.head:
            raise IndexError("*-* A* error : Pop from empty list.")
        popped_data = self.head.data
        self.head = self.head.next
        return popped_data
    
    def get(self):
        if not self.head:
            raise IndexError("*-* A* error : Get from empty list.")
        return self.head.data
    
    def is_empty(self):
        return self.head is None
    
    def to_list(self):
        ret = []
        current_node = self.head
        while current_node is not None:
            ret.append(current_node.data)
            current_node = current_node.next
        return ret

class Astar():
    def __init__(self, domain : HeuristicDomain):
        self.domain = domain
        self.g = {}
        self.visited = set()
        self.queue = LinkedList()

    def solve(self):
        self.queue.add(self.domain.get_initial_state())
        self.visited.add(self.domain.get_initial_state())
        while not self.queue.is_empty() and not self.domain.is_goal(self.queue.get()) and not self.domain.is_terminal(self.queue.get()):
            actions = self.domain.generate_actions(self.queue.get())
            if len(actions) == 0:
                self.queue.pop()
                continue
            next_state = None
            lowest_value = -1
            for action in actions:
                tmp_state = self.domain.generate_state(self.queue.get(), action)
                if tmp_state in self.visited:
                    continue
                tmp_value = self.domain.get_transition_value(self.queue.get(), action, tmp_state) + self.domain.get_heuristic_value(self.queue.get(), action, tmp_state)
                if lowest_value >= 0 and tmp_value >= lowest_value:
                    continue
                lowest_value = tmp_value
                next_state = tmp_state
            if lowest_value == -1:
                self.queue.pop()
                continue
            self.queue.add(next_state)
            self.visited.add(next_state)
        ret = self.queue.to_list()
        ret.reverse()
        return ret