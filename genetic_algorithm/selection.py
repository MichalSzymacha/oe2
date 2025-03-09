import numpy as np

def selection_elite(population, elite_size):
    """
    Selektuje najlepszych osobników (elitarnych) o rozmiarze `elite_size`.
    Zakładamy, że populacja jest posortowana wg fitness od najlepszego do najgorszego.
    """
    return population[:elite_size]

def selection_roulette(population):
    """
    Selekcja ruletkowa.
    Zakładamy, że fitness obliczamy tak, żeby im wyższy tym lepszy.
    Dla minimalizacji można np. użyć 1/(1+wartość_funkcji) lub inne podejście.
    """
    total_fitness = sum(ind.fitness for ind in population)
    pick = np.random.rand() * total_fitness
    current = 0
    for ind in population:
        current += ind.fitness
        if current > pick:
            return ind

def selection_tournament(population, tournament_size=3):
    """
    Selekcja turniejowa.
    Losujemy 'tournament_size' osobników i wybieramy najlepszego spośród nich.
    """
    chosen = np.random.choice(population, size=tournament_size, replace=False)
    chosen_sorted = sorted(chosen, key=lambda x: x.fitness, reverse=True)
    # reverse=True - jeśli fitness większy = lepszy
    return chosen_sorted[0]
