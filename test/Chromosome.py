import numpy as np
from typing import Tuple
import numpy.typing as npt

class Chromosome:
    def __init__(self, size : int, vars_number : int, genes : npt.NDArray[np.int32] = None):
        self.size = size
        self.vars_number = vars_number
        self.genes_size = size * vars_number
        if genes is not None:
            self.genes = genes
            if len(genes) != self.genes_size:
                raise ValueError(f"Genes size should be {self.genes_size}, but got {len(genes)}")
        else:
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
  

    def decode(self, var_bounds):
        genes = np.split(self.genes, self.vars_number)
        print(genes)
        return [self.decode_gene(gene, var_bounds) for gene in genes]
        
    def decode_gene(self, gene, var_bounds):
        return var_bounds[0] + int("".join(map(str, gene)), 2) * (var_bounds[1] - var_bounds[0]) / (2**self.size - 1)



if __name__ == "__main__":
    
    chrom2 = Chromosome(12, 2)
    print(chrom2.genes)
    
    print()
    # print(f'chrom1 {chrom1.genes}')  
    # print(f'chrom2 {chrom2.genes}')
