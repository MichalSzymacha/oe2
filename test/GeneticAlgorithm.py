import random
from typing import Callable, List, Tuple
from Population import Population
from Individual import Individual

class GeneticAlgorithm:
    def __init__(
        self,
        var_bounds: Tuple[float, float],
        precision: float,
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
        inversion_prob: float,
        elite_percentage: float,
        objective_function: Callable[[List[float]], float],
        maximize: bool = True
    ):
        """
        Zarządza podstawowymi ustawieniami i przebiegiem algorytmu genetycznego.
        
        :param var_bounds:krotka (min, max) dla zmiennej
        :param precision: dokładność (np. jeśli kodujemy zmienne binarnie, decyduje o liczbie bitów)
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
        self.precision = precision
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

        self.inversion_prob = inversion_prob
        self.elite_percentage = elite_percentage
        
        self.objective_function = objective_function
        self.maximize = maximize

        self.population = self.initialize_population()


    def initialize_population(self):
        """
        Inicjalizuje populację. Tutaj decydujemy, jak kodujemy zmienne (np. binarnie).
        Na przykład można zapisać 'chromosom' jako listę bitów lub liczb rzeczywistych.
        """
        return Population(self.var_bounds, self.precision, self.vars_number, self.pop_size)

    def _evaluate(self):
        """
        Oblicza wartość funkcji celu dla danego chromosomu i zwraca fitness.
        Jeśli maximize = True, to fitness może być bezpośrednio wynikiem objective_function.
        Jeśli maximize = False (czyli minimalizacja), to często liczymy fitness = 1 / (1 + f(x))
        lub stosujemy inną transformację, by móc używać podobnych mechanizmów selekcji.
        """
        
        if self.maximize:
            self.population.evaluate(self.objective_function)
        else:
            self.population.evaluate(self.objective_function)
            for inv in self.population.population:
                inv.fitness = -inv.fitness

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
        new_population = random.choices(self.population.population, weights=probabilities, k=int(self.pop_size*self.selection_percentage))
        return new_population
    
    def _selection_tournament(self) -> list:
        """
        Przykładowa selekcja turniejowa. 
        Zwraca nową populację (kopie osobników).
        """
        self._evaluate()
        new_population = []
        for i in range(0, self.pop_size, self.tournament_size):
            winner = max(self.population.population[i:i+self.tournament_size], key=lambda x: x.fitness)
            new_population.append(winner)
        return new_population
        
    def _selection_best(self) -> list:
        """
        Wybieramy pewien % najlepszych i z nich tworzymy nową populację (bądź dociągamy do pop_size).
        """
        self._evaluate()
        self.population.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = self.population.population[:int(self.selection_percentage * self.pop_size)]
        return new_population

    def _crossover(self, parent1, parent2):
        """
        Zwraca nowy chromosom (potomek) utworzony z dwóch rodziców (chromosome list).
        """
        if self.crossover_method == "one_point":
            return self._crossover_one_point(parent1, parent2)
        elif self.crossover_method == "two_point":
            return self._crossover_two_point(parent1, parent2)
        elif self.crossover_method == "uniform":
            return self._crossover_uniform(parent1, parent2)
        elif self.crossover_method == "grain":
            return self._crossover_grain(parent1, parent2)
        else:
            # Możesz tu dodać własne metody...
            return parent1[:]  # np. brak krzyżowania

    def _crossover_one_point(self, p1, p2):
        """
        Przykład krzyżowania jednopunktowego.
        """
        if random.random() > self.crossover_prob:
            # Bez krzyżowania
            return p1[:]
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:]

    def _crossover_two_point(self, p1, p2):
        """
        Krzyżowanie dwupunktowe.
        """
        if random.random() > self.crossover_prob:
            return p1[:]
        length = len(p1)
        c1, c2 = sorted(random.sample(range(1, length), 2))
        return p1[:c1] + p2[c1:c2] + p1[c2:]

    def _crossover_uniform(self, p1, p2):
        """
        Uniform crossover (dla każdego genu losujemy pochodzenie).
        """
        child = []
        for i in range(len(p1)):
            if random.random() < 0.5:
                child.append(p1[i])
            else:
                child.append(p2[i])
        return child

    def _crossover_grain(self, p1, p2):
        """
        Przykład innego krzyżowania: 
        np. dzielimy chromosom na kilka segmentów i mieszamy je między rodzicami.
        """
        # Tu wstaw własną logikę. Tymczasowo – identyczny do uniform.
        return self._crossover_uniform(p1, p2)

    def _mutation(self, chromosome):
        """
        Mutuje dany chromosom według self.mutation_method.
        """
        if self.mutation_method == "boundary":
            self._mutation_boundary(chromosome)
        elif self.mutation_method == "one_point":
            self._mutation_one_point(chromosome)
        elif self.mutation_method == "two_points":
            self._mutation_two_points(chromosome)
        else:
            pass  # brak mutacji

    def _mutation_boundary(self, chromosome):
        """
        Przykładowa mutacja boundary – czyli "wyskakujemy" na kraniec przedziału.
        """
        for i in range(len(chromosome)):
            if random.random() < self.mutation_prob:
                min_b, max_b = self.var_bounds[i]
                # losowo wybierz czy min czy max
                chromosome[i] = random.choice([min_b, max_b])

    def _mutation_one_point(self, chromosome):
        """
        Mutacja jednopunktowa (np. zmiana wartości na losową w obrębie [min, max]).
        """
        for i in range(len(chromosome)):
            if random.random() < self.mutation_prob:
                min_b, max_b = self.var_bounds[i]
                chromosome[i] = random.uniform(min_b, max_b)

    def _mutation_two_points(self, chromosome):
        """
        Mutacja w dwóch losowo wybranych pozycjach.
        """
        # Możemy losować dwa indeksy i w tych pozycjach wprowadzić zmiany
        idxs = [i for i in range(len(chromosome))]
        random.shuffle(idxs)
        chosen = idxs[:2]
        for i in chosen:
            if random.random() < self.mutation_prob:
                min_b, max_b = self.var_bounds[i]
                chromosome[i] = random.uniform(min_b, max_b)

    def _inversion(self, chromosome):
        """
        Przykładowa inwersja – odwrócenie kolejności genów w wylosowanym przedziale.
        """
        if random.random() < self.inversion_prob:
            c1, c2 = sorted(random.sample(range(len(chromosome)), 2))
            chromosome[c1:c2] = reversed(chromosome[c1:c2])

    def run(self):
        """
        Główna pętla – uruchamia algorytm na 'epochs' epok.
        """
        
if __name__ == '__main__':
    def func(x):
        return x[0] + x[1]

    ga = GeneticAlgorithm(
        var_bounds=(0, 7),
        precision=3,
        vars_number=2,
        pop_size=10,
        epochs=10,
        selection_method="best",
        selection_percentage=0.5,
        tournament_size=3,
        crossover_method="one_point",
        crossover_prob=0.9,
        mutation_method="one_point",
        mutation_prob=0.1,
        inversion_prob=0.01,
        elite_percentage=0.1,
        objective_function=func,
        maximize=True
    )
    ga._selection_roulette()