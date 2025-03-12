import numpy as np
from typing import Tuple

class Chromosome:
    def __init__(self, size : int, vars_number : int):
        self.size = size
        self.vars_number = vars_number
        self.genes_size = size * vars_number
        self.genes = np.random.choice([0, 1], size=self.genes_size)
    '''
    Funkcja tworząca losowy chromosom o zadanej długości
    :param size: int - długość chromosomu
    '''

    def mutate(self):
        """
        Mutacja pojedynczego bitu
        """
        i = np.random.randint(self.genes_size)
        self.genes[i] = self.genes[i] ^ 1

    def crossover_one_point(self, other_chromosome):
        """
        Krzyżowanie jednopunktowe
        """
        i = np.random.randint(self.genes_size)
        self.genes[i:], other_chromosome.genes[i:] = other_chromosome.genes[i:], self.genes[i:]


    def crossover_two_point(self, other_chromosome):
        """
        Krzyżowanie dwupunktowe
        """
        i, j = np.random.choice(self.genes_size, 2, replace=False)
        if i > j:
            i, j = j, i
        self.genes[i:j], other_chromosome.genes[i:j] = other_chromosome.genes[i:j], self.genes[i:j]

    def crossover_uniform(self, other_chromosome, crossover_prob):
        """
        Krzyżowanie jednorodne
        """
        for i in range(self.genes_size):
            if np.random.rand() < crossover_prob:
                self.genes[i], other_chromosome.genes[i] = other_chromosome.genes[i], self.genes[i]
    
    def crossover_grain(self, other_chromosome, crossover_prob):
        """
        Krzyżowanie ziarniste
        """
        grain_size = 2
        for i in range(0, self.genes_size, grain_size):
            if np.random.rand() < crossover_prob:
                self.genes[i:i+grain_size], other_chromosome.genes[i:i+grain_size] = other_chromosome.genes[i:i+grain_size], self.genes[i:i+grain_size]

    def inversion(self):
        """
        Inwersja
        """
        i, j = np.random.choice(self.genes_size, 2, replace=False)
        if i > j:
            i, j = j, i
        for k in range((j-i)//2):
            self.genes[i+k], self.genes[j-k-1] = self.genes[j-k-1], self.genes[i+k]    

    def decode(self, var_bounds):
        genes = np.split(self.genes, self.vars_number)
        print(genes)
        return [self.decode_gene(gene, var_bounds) for gene in genes]
        
    def decode_gene(self, gene, var_bounds):
        return var_bounds[0] + int("".join(map(str, gene)), 2) * (var_bounds[1] - var_bounds[0]) / (2**self.size - 1)



if __name__ == "__main__":
    chrom1 = Chromosome(12, 2)
    chrom2 = Chromosome(12, 2)

    chrom1.crossover_one_point(chrom2)
    print()
    # print(f'chrom1 {chrom1.genes}')  
    # print(f'chrom2 {chrom2.genes}')
