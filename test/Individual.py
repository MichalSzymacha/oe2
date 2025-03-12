from typing import Tuple
from Chromosome import Chromosome
import numpy as np

class Individual:
    def __init__(self, size : int, vars_number : int):
        self.size = size
        self.vars_number = vars_number
        self.chromosomes = Chromosome(size, vars_number)
        self.fitness = None

    def mutate(self, mutation_rate):
        pass

    def crossover(self, other_individual):
        pass

    def inversion(self, inversion_rate):
        pass

    def evaluate(self, objective_function, var_bounds):
        self.fitness = objective_function([chromosome.decode(var_bounds) for chromosome in self.chromosomes])



def func(x):
    return x[0] + x[1]

if __name__ == "__main__":
    ind = Individual(3, 2)
