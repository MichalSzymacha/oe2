from typing import Tuple
import numpy as np
import numpy.typing as npt


"""
TODO:

-dodanie strategii elitarnej
algorytm genetyczny - zaimpelmentować samo wykonanie 
"""

class Individual:
    def __init__(self, var_bounds : Tuple[float, float], vars_number : int, genes : npt.NDArray[np.float64] = None):
        self.var_bounds = var_bounds
        self.vars_number = vars_number
        self.fitness = None
        if genes is not None:
            if len(genes) != self.vars_number:
                raise ValueError(f"Genes size should be {self.vars_number}, but got {len(genes)}")
            if np.any(genes < var_bounds[0]) or np.any(genes > var_bounds[1]):
                raise ValueError(f"Genes should be in range {var_bounds}, but got {genes}")
            self.genes = genes
        else:
            self.genes = np.random.uniform(low = var_bounds[0], high = var_bounds[1], size = self.vars_number)

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}"

    def evaluate(self, objective_function):
        self.objective_value = objective_function(self.genes)
        self.fitness = self.objective_value
        
    def mutate_uniform(self):
        index = np.random.randint(0, self.vars_number)
        mutation_value = np.random.uniform(self.var_bounds[0], self.var_bounds[1])
        self.genes[index] = mutation_value

    def mutate_gaussian(self):
        scale_range = (self.var_bounds[0] + self.var_bounds[1])
        mutation_value = np.random.normal(0, scale = scale_range*0.1, size = self.vars_number)
        self.genes += mutation_value
        np.clip(self.genes, self.var_bounds[0], self.var_bounds[1], out = self.genes)
       

def func(x):
    return x[0] + x[1]

if __name__ == "__main__":
    ind = Individual((5, 10), 5)
    print(ind)
    ind.mutate_gaussian()
    print(ind)
