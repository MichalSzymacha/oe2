import copy
import os
import random
import numpy as np
from typing import Callable, List, Tuple
from genetic_algorithm.Population import Population
from genetic_algorithm.Individual import Individual
from utils.file_saver import save_results_to_csv, clear_file, save_results_to_json

# from Population import Population
# from Individual import Individual


class GeneticAlgorithm:
    def __init__(
        self,
        var_bounds: Tuple[float, float],
        vars_number: int,
        pop_size: int,
        epochs: int,
        selection_method: str,
        selection_percentage: float,
        tournament_size: int,
        crossover_method: str,
        crossover_prob: float,
        mutation_method: str,
        mutation_prob: float,
        elite_percentage: float,
        objective_function: Callable[[List[float]], float],
        maximize: bool = True,
    ):
        """
        Zarządza podstawowymi ustawieniami i przebiegiem algorytmu genetycznego.

        :param var_bounds:krotka (min, max) dla zmiennej
        :param vars_number: liczba zmiennych
        :param pop_size: rozmiar populacji
        :param epochs: liczba epok (iteracji)
        :param selection_method: metoda selekcji ("roulette", "tournament", "best",)
        :param selection_percentage: procent osobników branych do selekcji najlepszych i selekcji ruletkowej
        :param tournament_size: rozmiar turnieju (dla selekcji turniejowej)
        :param crossover_method: metoda krzyżowania ("one_point", "two_point", "uniform", "grain", ...)
        :param crossover_prob: prawdopodobieństwo krzyżowania
        :param mutation_method: metoda mutacji ("boundary", "one_point", "two_points", ...)
        :param mutation_prob: prawdopodobieństwo mutacji
        :param inversion_prob: prawdopodobieństwo inwersji
        :param elite_percentage: procent najlepszych osobników przenoszonych bez zmian (elityzm)
        :param objective_function: funkcja celu f(x1, x2, ..., xN) -> float
        :param maximize: czy maksymalizujemy (True) czy minimalizujemy (False) funkcję celu
        """
        self.var_bounds = var_bounds
        self.vars_number = vars_number
        self.pop_size = pop_size
        self.epochs = epochs

        self.selection_method = selection_method
        self.selection_percentage = selection_percentage
        self.tournament_size = tournament_size

        self.crossover_method = crossover_method
        self.crossover_prob = crossover_prob

        self.mutation_method = mutation_method
        self.mutation_prob = mutation_prob

        self.elite_percentage = elite_percentage

        self.objective_function = objective_function
        self.maximize = maximize

        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Inicjalizuje populację. Tutaj decydujemy, jak kodujemy zmienne (np. binarnie).
        Na przykład można zapisać 'chromosom' jako listę bitów lub liczb rzeczywistych.
        """
        return Population(
            self.var_bounds, self.vars_number, self.pop_size
        )

    def _evaluate(self):
        """
        Oblicza wartość funkcji celu dla danego chromosomu i zwraca fitness.
        Jeśli maximize = True, to fitness może być bezpośrednio wynikiem objective_function.
        Jeśli maximize = False (czyli minimalizacja), to często liczymy fitness = 1 / (1 + f(x))
        lub stosujemy inną transformację, by móc używać podobnych mechanizmów selekcji.
        """

        self.population.evaluate(self.objective_function)

        for inv in self.population.population:
            if self.maximize:
                inv.fitness = inv.objective_value
            else:
                try:
                    inv.fitness = 1 / (1 + abs(inv.objective_value))
                except ZeroDivisionError:
                    inv.fitness = float("inf")

    def _selection(self) -> list:
        """
        Na podstawie self.selection_method dokonuje wyboru rodziców/nowej populacji.
        Zwraza listę wybranych osobników (słowników z chromosome, fitness).
        """
        if self.selection_method == "roulette":
            return self._selection_roulette()
        elif self.selection_method == "tournament":
            return self._selection_tournament()
        elif self.selection_method == "best":
            return self._selection_best()
        else:
            raise ValueError(f"Nieznana metoda selekcji: {self.selection_method}")

    def _selection_roulette(self) -> list:
        """
        Przykładowa ruletka (prawdopodobieństwo proporcjonalne do fitness).
        Zwraca listę osobników (tyle, ile wynosi pop_size).
        """
        self._evaluate()
        fitnesses = [ind.fitness for ind in self.population.population]
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        new_population = random.choices(
            self.population.population,
            weights=probabilities,
            k=int(self.pop_size * self.selection_percentage),
        )
        return new_population

    def _selection_tournament(self) -> list:
        """
        Przykładowa selekcja turniejowa.
        Zwraca nową populację (kopie osobników).
        """
        self._evaluate()
        new_population = []
        for i in range(0, self.pop_size, self.tournament_size):
            winner = max(
                self.population.population[i : i + self.tournament_size],
                key=lambda x: x.fitness,
            )
            new_population.append(winner)
        return new_population

    def _selection_best(self) -> list:
        """
        Wybieramy pewien % najlepszych i z nich tworzymy nową populację (bądź dociągamy do pop_size).
        """
        self._evaluate()
        self.population.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = self.population.population[
            : int(self.selection_percentage * self.pop_size)
        ]
        return new_population

    def _crossover(self, parent1, parent2):
        """
        Zwraca nowy chromosom (potomek) utworzony z dwóch rodziców (chromosome list).
        """
        if np.random.rand() < self.crossover_prob:
            if self.crossover_method == "arithmetical":
                return self.population.crossover_arithmetical(parent1, parent2)
            elif self.crossover_method == "linear":
                return self.population.crossover_linear(parent1, parent2)
            elif self.crossover_method == "blendalpha":
                return self.population.crossover_blendalpha(
                    parent1, parent2)
            elif self.crossover_method == "blendalphabeta":
                return self.population.crossover_blendalphabeta(parent1, parent2)
            elif self.crossover_method == "average":
                return self.population.crossover_average(parent1, parent2)
        else:
            return parent1, parent2

    def _mutation(self, chromosome):
        """
        Mutuje dany chromosom według self.mutation_method.
        """
        if np.random.rand() < self.mutation_prob:
            if self.mutation_method == "uniform":
                chromosome.mutate_uniform()
            elif self.mutation_method == "gaussian":
                chromosome.mutate_gaussian()
            else:
                pass
        else:
            return chromosome


    def _elite(self):
        """
        Dodaje elityzm do nowej populacji.
        """
        self._evaluate()
        self.population.population.sort(key=lambda x: x.fitness, reverse=True)
        elite_size = int(self.elite_percentage * self.pop_size)

        array = copy.deepcopy(self.population.population[:elite_size])

        return array


    def run(self, dir_name="results"):
        """
        Główna pętla – uruchamia algorytm na 'epochs' epok.
        """

        os.makedirs(dir_name, exist_ok=True)
        clear_file(f"{dir_name}/wyniki.csv")
        save_results_to_json(f"{dir_name}/settings.json", self.__dict__)

        for epoch in range(self.epochs):
            new_population = []
            new_population = self._elite()
            parents = self._selection()
            parents_len = len(parents)
            while len(new_population) < self.pop_size:
                parent1, parent2 = (
                    parents[np.random.randint(parents_len)],
                    parents[np.random.randint(parents_len)],
                )
                children = self._crossover(parent1, parent2)
                if children is None:
                    continue
                for child in children:
                    self._mutation(child)
                    new_population.append(child)
            if len(new_population) > self.pop_size:
                new_population = new_population[: self.pop_size]

            self.population.population = new_population.copy()
            self._evaluate()
            self.population.population.sort(key=lambda x: x.fitness, reverse=True)




            mean = np.mean([ind.fitness for ind in self.population.population])
            std = np.std([ind.fitness for ind in self.population.population])
            best_individual = max(self.population.population, key=lambda x: x.fitness)
            save_results_to_csv(f"{dir_name}/wyniki.csv", epoch, best_individual.fitness, mean, std)

        best_individual = max(self.population.population, key=lambda x: x.fitness)
        if not self.maximize:
            best_individual.fitness = 1 / best_individual.fitness - 1
            
        return best_individual.fitness, best_individual.genes


if __name__ == "__main__":

    def func(x):
        return x[0] ** 2 + x[1] ** 2

    ga = GeneticAlgorithm(
        var_bounds=(-10, 10),
        vars_number=2,
        pop_size=100,
        epochs=100,
        selection_method="best",
        selection_percentage=0.5,
        tournament_size=3,
        crossover_method="linear",
        crossover_prob=0.9,
        mutation_method="uniform",
        mutation_prob=0.1,
        elite_percentage=0.1,
        objective_function=func,
        maximize=False,
    )

    best = ga.run()
    print(f"Best solution: {best}")
