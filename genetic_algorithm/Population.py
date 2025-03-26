from typing import Callable, List, Tuple
from genetic_algorithm.Individual import Individual
import math
import numpy as np

class Population:
    def __init__(
            self,
            var_bounds: Tuple[float, float],
            precision: float,
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
        self.precision = precision
        self.vars_number = vars_number
        self.size = self.compute_gene_size()
        self.genes_size = self.size * self.vars_number
        self.population = [Individual(self.size, self.vars_number) for _ in range(pop_size)]

    def compute_gene_size(self) -> int:
        """
        Oblicza długość chromosomu dla pojedynczej zmiennej
        """
        p = self.convert_precision_to_int()
        num_of_steps = (self.var_bounds[1] - self.var_bounds[0]) * 10**p + 1
        size = math.ceil(math.log2(num_of_steps))
        return size
    
    def convert_precision_to_int(self) -> int:
        """
        Zmienia precyzję dla całej populacji
        """
        p = 0
        local_precision = self.precision
        while local_precision < 1.0:
            local_precision *= 10
            p += 1
        return p
    
    def crossover_one_point(self, individual: Individual, other_individual: Individual):
        """
        Krzyżowanie jednopunktowe
        """
        i = np.random.randint(self.size * self.vars_number)
        return (Individual(self.size, self.vars_number, genes=np.concatenate((individual.genes[:i], other_individual.genes[i:]))),
                Individual(self.size, self.vars_number, genes=np.concatenate((other_individual.genes[:i], individual.genes[i:]))))

    def crossover_two_point(self, individual: Individual, other_individual: Individual):        
        """
        Krzyżowanie dwupunktowe
        """
        i, j = np.random.choice(self.genes_size, 2, replace=False)
        if i > j:
            i, j = j, i
        return (Individual(self.size, self.vars_number, genes=np.concatenate((individual.genes[:i], other_individual.genes[i:j], individual.genes[j:]))),
                Individual(self.size, self.vars_number, genes=np.concatenate((other_individual.genes[:i], individual.genes[i:j], other_individual.genes[j:]))))
    
    def crossover_uniform(self, individual: Individual, other_individual: Individual, crossover_prob: float):
        """
        Krzyżowanie jednorodne
        """
        genes1, genes2 = individual.genes, other_individual.genes
        for i in range(self.genes_size):
            if np.random.rand() < crossover_prob:
                genes1[i], genes2[i] = genes2[i], genes1[i]
        return (Individual(self.size, self.vars_number, genes=genes1),
                Individual(self.size, self.vars_number, genes=genes2))

                
    def crossover_grain(self, individual: Individual, other_individual: Individual, crossover_prob: float, grain_size: int = 2):
        """
        Krzyżowanie ziarniste
        """
        genes1, genes2 = individual.genes, other_individual.genes
        for i in range(0, self.genes_size, grain_size):
            if np.random.rand() < crossover_prob:
                genes1[i:i + grain_size], genes2[i:i + grain_size] = genes2[i:i + grain_size], genes1[i:i + grain_size]
        return (Individual(self.size, self.vars_number, genes=genes1),
                Individual(self.size, self.vars_number, genes=genes2))
    
    def evaluate(self, objective_function: Callable[[List[float]], float]):
        """
        Ocena populacji
        """
        for individual in self.population:
            individual.evaluate(objective_function, self.var_bounds)

    

def func(x):
    return x[0] + x[1]

if __name__ == "__main__":
    var_bounds = (0, 31)
    pop = Population(var_bounds, 1, 2, 10)
    # for ind in pop.population:
    #     print(ind.decode(var_bounds))

    pop.evaluate(func)
