import numpy as np
import time

from genetic_algorithm.individual import Individual
from genetic_algorithm.selection import selection_elite, selection_roulette, selection_tournament
from genetic_algorithm.crossover import (crossover_one_point, crossover_two_point,
                                        crossover_uniform, crossover_grain)
from genetic_algorithm.mutation import (mutation_boundary, mutation_one_point, mutation_two_points)
from genetic_algorithm.inversion import inversion


class GAManager:
    def __init__(self, 
                 var_bounds, 
                 bits_per_var,
                 pop_size,
                 epochs,
                 selection_method='roulette',
                 crossover_method='one_point',
                 crossover_prob=0.7,
                 mutation_method='one_point',
                 mutation_prob=0.01,
                 inversion_prob=0.0,
                 elite_percentage=0.1,
                 objective_function=None,  # funkcja celu
                 maximize=False):
        """
        var_bounds: lista krotek (min, max) dla każdej zmiennej
        bits_per_var: int - ile bitów koduje 1 zmienną (uprośćmy, że każda ma tyle samo)
        ...
        """
        self.var_bounds = var_bounds
        self.bits_per_var = bits_per_var
        self.pop_size = pop_size
        self.epochs = epochs

        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.crossover_prob = crossover_prob
        self.mutation_method = mutation_method
        self.mutation_prob = mutation_prob
        self.inversion_prob = inversion_prob
        
        self.elite_size = max(1, int(elite_percentage * pop_size)) \
            if elite_percentage < 1 else int(elite_percentage)

        self.objective_function = objective_function
        self.maximize = maximize

        # Przygotowanie populacji startowej
        self.chromosome_length = bits_per_var * len(var_bounds)
        self.population = self._init_population()

        # Do zbierania statystyk
        self.history_best = []
        self.history_mean = []
        self.history_std = []

    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            chromosome = np.random.randint(0, 2, self.chromosome_length)
            ind = Individual(chromosome, precision=1.0, var_bounds=self.var_bounds)
            population.append(ind)
        return population

    def evaluate_population(self):
        for ind in self.population:
            vals = ind.decode()
            f = self.objective_function(vals)
            # jeśli minimalizujemy, a chcemy selekcję typu ruletka, warto dać: ind.fitness = 1/(1+f)
            # lub wprost przechowywać "fit" = -f przy minimalizacji
            if self.maximize:
                ind.fitness = f
            else:
                # np. w celu selekcji ruletki: im mniejsza funkcja tym większy fitness
                # prosta transformacja (ale uwaga na f<0!)
                ind.fitness = 1 / (1 + f*f)  # cokolwiek, by uniknąć 0
                # Można to zmienić na inną formułę w zależności od typu funkcji

    def run(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            self.evaluate_population()
            # Sortujemy populację wg fitness malejąco
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            best_fitness = self.population[0].fitness
            mean_fitness = np.mean([ind.fitness for ind in self.population])
            std_fitness = np.std([ind.fitness for ind in self.population])

            self.history_best.append(best_fitness)
            self.history_mean.append(mean_fitness)
            self.history_std.append(std_fitness)

            new_population = []

            # Elitarna część
            elite = selection_elite(self.population, self.elite_size)
            new_population.extend(elite)

            # Tworzymy resztę populacji
            while len(new_population) < self.pop_size:
                # Selekcja rodziców
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                # Krzyżowanie
                child1_bits, child2_bits = self._crossover(parent1, parent2)
                # Mutacja
                child1_bits = self._mutation(child1_bits)
                child2_bits = self._mutation(child2_bits)
                # Inwersja
                child1_bits = inversion(child1_bits, self.inversion_prob)
                child2_bits = inversion(child2_bits, self.inversion_prob)
                # Nowi osobnicy
                child1 = Individual(chromosome=child1_bits,
                                    precision=1.0,
                                    var_bounds=self.var_bounds)
                child2 = Individual(chromosome=child2_bits,
                                    precision=1.0,
                                    var_bounds=self.var_bounds)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            self.population = new_population

        end_time = time.time()
        total_time = end_time - start_time
        return total_time

    def _select_parent(self):
        if self.selection_method == 'roulette':
            return selection_roulette(self.population)
        elif self.selection_method == 'tournament':
            return selection_tournament(self.population, tournament_size=3)
        elif self.selection_method == 'best':
            # bierzemy najlepszego (pierwszy w posortowanej populacji)
            return self.population[0]
        else:
            # Domyślnie ruletka
            return selection_roulette(self.population)

    def _crossover(self, parent1, parent2):
        if self.crossover_method == 'one_point':
            return crossover_one_point(parent1, parent2, self.crossover_prob)
        elif self.crossover_method == 'two_point':
            return crossover_two_point(parent1, parent2, self.crossover_prob)
        elif self.crossover_method == 'uniform':
            return crossover_uniform(parent1, parent2, self.crossover_prob)
        elif self.crossover_method == 'grain':
            return crossover_grain(parent1, parent2, self.crossover_prob, grain_size=2)
        else:
            return crossover_one_point(parent1, parent2, self.crossover_prob)

    def _mutation(self, chromosome_bits):
        # Tworzymy tymczasowego osobnika do funkcji mutacji
        fake_ind = Individual(chromosome_bits, precision=1.0, var_bounds=self.var_bounds)
        if self.mutation_method == 'boundary':
            return mutation_boundary(fake_ind, self.mutation_prob)
        elif self.mutation_method == 'one_point':
            return mutation_one_point(fake_ind, self.mutation_prob)
        elif self.mutation_method == 'two_points':
            return mutation_two_points(fake_ind, self.mutation_prob)
        else:
            return mutation_one_point(fake_ind, self.mutation_prob)
