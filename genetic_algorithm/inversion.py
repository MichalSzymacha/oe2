import numpy as np

def inversion(chromosome, prob):
    """
    Odwrócenie kolejności fragmentu chromosomu z prawdopodobieństwem prob.
    """
    if np.random.rand() > prob:
        return chromosome.copy()
    csize = len(chromosome)
    start, end = sorted(np.random.choice(range(csize), 2, replace=False))
    inv_part = chromosome[start:end+1][::-1]
    new_chrom = chromosome.copy()
    new_chrom[start:end+1] = inv_part
    return new_chrom
