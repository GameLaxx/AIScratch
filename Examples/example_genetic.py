from AIScratch.Genetic.genetic_algorithm import GeneticElement, GeneticSolver
import numpy as np
import random

"""
Example of usage of the genetic algorithm of the library.
In this case, we are trying to find the best run possible.
The possible parameters are the distance, the different speeds
that one is able to run and the effort while maintaining this speed.
The best run is defined using flags.
NO_CONSTRAINT tries to find the run that is the closest to the distance.
QUICK tries to find the run that arrives at the distance the fastest.
EASY tries to find the run that asks for the less efforts.
BOTH tries to find the run that is the fastest while asking for the less efforts.
Each element is a list of time in h and each speed is in km/h.
"""

# flags
QUICK = 0
EASY = 1
BOTH = 2
NO_CONSTRAINT = 3

class Run(GeneticElement):
    def __init__(self, element):
        super().__init__()
        self.element = np.array(element)

    def __mul__(self, other : int) -> "Run":
        return Run(other * self.element)
    __rmul__ = __mul__ 
    
    def __add__(self, other : "Run"):
        return Run(self.element + other.element)
    
    def __repr__(self):
        return str(self.element)

class Solver(GeneticSolver):
    def __init__(self, population_size, number_of_generations, solver_type, distance, speeds, efforts = None, tol=0.01, elite_rate=0.2, exploration_rate=0.2, mutation_rate=0.2):
        super().__init__(population_size, number_of_generations, tol, elite_rate, exploration_rate, mutation_rate)
        if solver_type == EASY or solver_type == BOTH:
            assert efforts != None, "*-* Problem occured : efforts asked but not given"
        self.distance = distance
        self.speeds = speeds
        self.efforts = efforts
        self.solver_type = solver_type

    def _init_elem(self):
        ret = []
        for _ in range(len(self.speeds)):
            ret.append(random.random() * 0.5) # random number between 0 and 30min for each speeds
        return Run(ret)
    
    def _breeding(self, elm1, elm2):
        alpha = random.random()
        return alpha * elm1 + (1 - alpha) * elm2
    
    def _selection(self, population, fitness, elite_rate):
        nb_elites = int(len(population) * elite_rate)
        indexes_sorted = np.argsort(fitness)
        return [population[i] for i in indexes_sorted[:nb_elites]]
    
    def _mutation(self, elem : Run):
        ret = elem.element.copy()
        for i in range(len(elem.element)):
            mutation_size = random.random() * 0.4 - 0.2
            ret[i] += mutation_size
            ret[i] = max(ret[i], 0)
        return Run(ret)
    
    def __constraint_quick(self, elem : Run):
        ret = 0
        for i in range(len(self.speeds)):
            ret -= elem.element[i] # add all the times
        return abs(5 / self.speeds[-1] - ret) # difference between the fastest time and the current time
    
    def __constraint_easy(self, elem : Run):
        ret = 0
        for i in range(len(self.speeds)):
            ret += elem.element[i] * self.efforts[i] # t * effort
        return ret # difference between the easiest and the current difficulty
    
    def constraint(self, elem : Run):
        if self.solver_type == NO_CONSTRAINT:
            return 0
        ret = 0
        if self.solver_type == BOTH or self.solver_type == QUICK:
            ret += self.__constraint_quick(elem)
        if self.solver_type == BOTH or self.solver_type == EASY:
            ret += self.__constraint_easy(elem)
        return ret
    
    def _objective(self, elem : Run):
        ret = 0
        for i in range(len(self.speeds)):
            ret += elem.element[i] * self.speeds[i] # d = t * v
        return abs(ret - self.distance) + self.constraint(elem)
    
    def _debug(self):
        return str(min(self._GeneticSolver__last_fitness))

# parameters
_speeds = [7,7.5,8,8.5,9,9.5,10,10.5,11] # speeds one is able to run (km/h)
_efforts = [0.1,0.2,0.4,0.5,0.6,0.8,0.9,0.98,1] # how hard it is to keep the speed (0 to 1)
solver = Solver(1000, 100, BOTH, 5, _speeds, _efforts)
elem, fit = solver.solve()
print("Final results")
print(f"Best element : {elem}, best fitness : {fit}")
print(f"Distance : {solver.distance - fit + solver.constraint(elem)}km")
time = 0
for i in range(len(elem.element)):
    time += elem.element[i]
print(f"Time : {time * 60} min ({time}h)")