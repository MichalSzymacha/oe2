from typing import Callable, List, Tuple
from Individual import Individual
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
    
    def crossover_one_point(self, chromosome: Individual, other_chromosome: Individual):
        """
        Krzyżowanie jednopunktowe
        """
        i = np.random.randint(self.size * self.vars_number)
        print(i)
        # print(f"{chromosome.chromosomes.genes}\n{other_chromosome.chromosomes.genes}")
        chromosome.chromosomes.genes[i:], other_chromosome.chromosomes.genes[i:] = other_chromosome.chromosomes.genes[i:], chromosome.chromosomes.genes[i:]
        print()
        # print(f"{chromosome.chromosomes.genes}\n{other_chromosome.chromosomes.genes}")

if __name__ == "__main__":
    pop = Population((0, 1), 0.01, 2, 2)
    print(pop.population[0].chromosomes.genes)
    print(pop.population[1].chromosomes.genes)
    pop.crossover_one_point(pop.population[0], pop.population[1])
    print(pop.population[0].chromosomes.genes)
    print(pop.population[1].chromosomes.genes)