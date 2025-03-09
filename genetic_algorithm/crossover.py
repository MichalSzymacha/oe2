import numpy as np

def crossover_one_point(parent1, parent2, prob):
    if np.random.rand() > prob:
        # Bez krzyżowania - zwracamy kopie
        return parent1.chromosome.copy(), parent2.chromosome.copy()

    point = np.random.randint(1, len(parent1.chromosome))
    child1 = np.concatenate((parent1.chromosome[:point], parent2.chromosome[point:]))
    child2 = np.concatenate((parent2.chromosome[:point], parent1.chromosome[point:]))
    return child1, child2

def crossover_two_point(parent1, parent2, prob):
    if np.random.rand() > prob:
        return parent1.chromosome.copy(), parent2.chromosome.copy()

    csize = len(parent1.chromosome)
    point1, point2 = sorted(np.random.choice(range(1, csize), 2, replace=False))
    child1 = np.concatenate((parent1.chromosome[:point1],
                             parent2.chromosome[point1:point2],
                             parent1.chromosome[point2:]))

    child2 = np.concatenate((parent2.chromosome[:point1],
                             parent1.chromosome[point1:point2],
                             parent2.chromosome[point2:]))

    return child1, child2

def crossover_uniform(parent1, parent2, prob):
    if np.random.rand() > prob:
        return parent1.chromosome.copy(), parent2.chromosome.copy()
    
    csize = len(parent1.chromosome)
    mask = np.random.randint(0, 2, csize)  # 0/1 w losowych miejscach
    child1 = np.where(mask, parent1.chromosome, parent2.chromosome)
    child2 = np.where(mask, parent2.chromosome, parent1.chromosome)
    return child1, child2

def crossover_grain(parent1, parent2, prob, grain_size=2):
    """
    Tzw. "krzyżowanie ziarniste" – dzielimy chromosom na bloki o długości grain_size.
    Losowo wybieramy, z którego rodzica bierzemy dany blok.
    """
    if np.random.rand() > prob:
        return parent1.chromosome.copy(), parent2.chromosome.copy()
    
    csize = len(parent1.chromosome)
    child1 = np.zeros(csize, dtype=int)
    child2 = np.zeros(csize, dtype=int)

    for i in range(0, csize, grain_size):
        # wylosuj, od którego rodzica pobieramy ziarno
        if np.random.rand() < 0.5:
            child1[i:i+grain_size] = parent1.chromosome[i:i+grain_size]
            child2[i:i+grain_size] = parent2.chromosome[i:i+grain_size]
        else:
            child1[i:i+grain_size] = parent2.chromosome[i:i+grain_size]
            child2[i:i+grain_size] = parent1.chromosome[i:i+grain_size]

    return child1, child2
