from typing import Callable, List, Tuple
from genetic_algorithm.Individual import Individual
import math
import numpy as np

# from Individual import Individual

class Population:
    def __init__(
            self,
            var_bounds: Tuple[float, float],
            vars_number: int,
            pop_size: int,
            ):
        """
        :param var_bounds:krotka (min, max) dla zmiennej
        :param precision: dokładność (np. jeśli kodujemy zmienne binarnie, decyduje o liczbie bitów)
        :param vars_number: liczba zmiennych
        :param pop_size: rozmiar populacji"
        """
        self.var_bounds = var_bounds
        self.vars_number = vars_number
        self.population = [Individual(self.var_bounds, self.vars_number) for _ in range(pop_size)]

    def __str__(self) -> str:
        for ind in self.population:
            print(ind)
        return ""
    
    def crossover_arithmetical(self, individual: Individual, other_individual: Individual):
        a = np.random.uniform(0, 1)
        b = 1 - a
        child1 = a * individual.genes + b * other_individual.genes
        child2 = b * individual.genes + a * other_individual.genes
        if np.any(child1 < self.var_bounds[0]) or np.any(child1 > self.var_bounds[1]) or np.any(child2 < self.var_bounds[0]) or np.any(child2 > self.var_bounds[1]):
            return
        return (Individual(self.var_bounds, self.vars_number, child1),
                Individual(self.var_bounds, self.vars_number, child2))
        
    def crossover_linear(self, individual: Individual, other_individual: Individual):
        child1 = 0.5*(individual.genes + other_individual.genes)
        child2 = 1.5*individual.genes - 0.5*other_individual.genes
        child3 = -0.5*individual.genes + 1.5*other_individual.genes
        
        if np.any(child1 < self.var_bounds[0]) or np.any(child1 > self.var_bounds[1]) or np.any(child2 < self.var_bounds[0]) or np.any(child2 > self.var_bounds[1]) or np.any(child3 < self.var_bounds[0]) or np.any(child3 > self.var_bounds[1]):
            return
        
        Z = Individual(self.var_bounds, self.vars_number, child1)
        V = Individual(self.var_bounds, self.vars_number, child2)
        W = Individual(self.var_bounds, self.vars_number, child3)
        children = [Z, V, W]
        for child in children:
            child.evaluate(func)
        children.sort(key=lambda x: x.fitness, reverse=True)
        return (children[0], children[1])
        
    def crossover_blendalpha(self, individual: Individual, other_individual: Individual):
        a = np.random.uniform(0, 1)
        d = abs(individual.genes - other_individual.genes)
        min_array = np.minimum(individual.genes, other_individual.genes) - a*d
        max_array = np.maximum(individual.genes, other_individual.genes) + a*d
        child1 = np.random.uniform(min_array, max_array)
        child2 = np.random.uniform(min_array, max_array)
        if np.any(child1 < self.var_bounds[0]) or np.any(child1 > self.var_bounds[1]) or np.any(child2 < self.var_bounds[0]) or np.any(child2 > self.var_bounds[1]):
            return
        return (Individual(self.var_bounds, self.vars_number, child1),
                Individual(self.var_bounds, self.vars_number, child2))
        
    def crossover_blendalphabeta(self, individual: Individual, other_individual: Individual):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)
        d = abs(individual.genes - other_individual.genes)
        min_array = np.minimum(individual.genes, other_individual.genes) - a*d
        max_array = np.maximum(individual.genes, other_individual.genes) + b*d
        child1 = np.random.uniform(min_array, max_array)
        child2 = np.random.uniform(min_array, max_array)
        if np.any(child1 < self.var_bounds[0]) or np.any(child1 > self.var_bounds[1]) or np.any(child2 < self.var_bounds[0]) or np.any(child2 > self.var_bounds[1]):
            return
        return (Individual(self.var_bounds, self.vars_number, child1),
                Individual(self.var_bounds, self.vars_number, child2))
        
    def crossover_average(self, individual: Individual, other_individual: Individual):
        child = 0.5*(individual.genes + other_individual.genes)
        if np.any(child < self.var_bounds[0]) or np.any(child > self.var_bounds[1]):
            return
        return [Individual(self.var_bounds, self.vars_number, child)]
        
        
    
    def evaluate(self, objective_function: Callable[[List[float]], float]):
        """
        Ocena populacji
        """
        for individual in self.population:
            individual.evaluate(objective_function)

    

def func(x):
    return x[0] + x[1]

if __name__ == "__main__":
    var_bounds = (-1, 1)
    pop = Population(var_bounds, 3, 2)
    # for ind in pop.population:
    #     print(ind.decode(var_bounds))
    print(pop)
    pop2 = pop.crossover_average(pop.population[0], pop.population[1])
    print(pop2[0])

    
