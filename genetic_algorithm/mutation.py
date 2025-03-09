import numpy as np

def mutation_boundary(individual, prob):
    """
    Mutacja brzegowa - np. ustawienie wartości bitu skrajnego (0->1 albo 1->0) 
    tylko na początku i końcu segmentu? 
    Można też interpretować "mutację brzegową" inaczej.
    """
    chromosome = individual.chromosome.copy()
    # Zmieniamy pierwszego i/lub ostatniego bitu z prawdopodobieństwem prob
    if np.random.rand() < prob:
        chromosome[0] = 1 - chromosome[0]
    if np.random.rand() < prob:
        chromosome[-1] = 1 - chromosome[-1]
    return chromosome

def mutation_one_point(individual, prob):
    chromosome = individual.chromosome.copy()
    for i in range(len(chromosome)):
        if np.random.rand() < prob:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def mutation_two_points(individual, prob):
    chromosome = individual.chromosome.copy()
    # Losujemy 2 pozycje i mutujemy tylko je
    # (zamiast iterować po całości)
    idxs = np.random.choice(range(len(chromosome)), 2, replace=False)
    for idx in idxs:
        if np.random.rand() < prob:
            chromosome[idx] = 1 - chromosome[idx]
    return chromosome
