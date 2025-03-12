import random
from typing import Callable, List, Tuple

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
        objective_function: Callable,
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
        :param selection_percentage: procent osobników branych do selekcji najlepszych (w niektórych metodach)
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
        Tworzymy populacje z klasy population

    def _evaluate(self, chromosome) -> float:
        """
        Oblicza wartość funkcji celu dla danego chromosomu i zwraca fitness.
        Jeśli maximize = True, to fitness może być bezpośrednio wynikiem objective_function.
        Jeśli maximize = False (czyli minimalizacja), to często liczymy fitness = 1 / (1 + f(x))
        lub stosujemy inną transformację, by móc używać podobnych mechanizmów selekcji.
        """
        raw_value = self.objective_function(chromosome)
        if self.maximize:
            return raw_value
        else:
            # Jedna z popularnych transformacji do minimalizacji:
            # unikamy dzielenia przez zero, stąd (abs(raw_value) + 1)
            return 1.0 / (abs(raw_value) + 1.0)

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
        # Suma fitnessów
        total_fitness = sum(ind['fitness'] for ind in self.population)
        new_population = []
        
        for _ in range(self.pop_size):
            pick = random.uniform(0, total_fitness)
            current = 0
            for ind in self.population:
                current += ind['fitness']
                if current > pick:
                    # Tworzymy kopię, żeby nie modyfikować oryginału
                    new_population.append({
                        'chromosome': ind['chromosome'][:],
                        'fitness': ind['fitness']
                    })
                    break
        return new_population

    def _selection_tournament(self) -> list:
        """
        Przykładowa selekcja turniejowa. 
        Zwraca nową populację (kopie osobników).
        """
        new_population = []
        for _ in range(self.pop_size):
            # Losujemy turniej
            tournament = random.sample(self.population, self.tournament_size)
            # Wybieramy najlepszego (lub najgorszego, jeśli minimize – ale tu mamy fitness przetransformowane)
            best = max(tournament, key=lambda ind: ind['fitness'])
            new_population.append({
                'chromosome': best['chromosome'][:],
                'fitness': best['fitness']
            })
        return new_population

    def _selection_best(self) -> list:
        """
        Wybieramy pewien % najlepszych i z nich tworzymy nową populację (bądź dociągamy do pop_size).
        """
        sorted_pop = sorted(self.population, key=lambda ind: ind['fitness'], reverse=True)
        # Ile osobników bierzemy z top:
        k = int(self.pop_size * self.selection_percentage)
        best_part = sorted_pop[:k]

        new_population = []
        # Uzupełniamy populację wielokrotnie powielając najlepszych,
        # lub dobieramy losowo spośród najlepszych (zależy od strategii).
        while len(new_population) < self.pop_size:
            ind = random.choice(best_part)
            new_population.append({
                'chromosome': ind['chromosome'][:],
                'fitness': ind['fitness']
            })
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
        # Krok 1: inicjalizacja populacji
        self.initialize_population()

        # Główna pętla
        for epoch in range(self.epochs):
            # Selekcja
            new_population = self._selection()

            # Elityzm – zachowanie pewnego procenta najlepszych
            elites = []
            if self.elite_percentage > 0:
                elite_count = max(1, int(self.pop_size * self.elite_percentage))
                sorted_pop = sorted(self.population, key=lambda ind: ind['fitness'], reverse=True)
                elites = sorted_pop[:elite_count]

            # Krzyżowanie, mutacja, inwersja
            next_pop = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    parent1 = new_population[i]
                    parent2 = new_population[i+1]
                else:
                    # Nieparzysta liczba osobników, ostatni rodzic kopiuje się sam
                    parent1 = new_population[i]
                    parent2 = new_population[i]

                # Krzyżowanie
                child1_chrom = self._crossover(parent1['chromosome'], parent2['chromosome'])
                child2_chrom = self._crossover(parent2['chromosome'], parent1['chromosome'])

                # Mutacja i inwersja
                self._mutation(child1_chrom)
                self._mutation(child2_chrom)
                self._inversion(child1_chrom)
                self._inversion(child2_chrom)

                # Obliczamy nowe fitnessy
                child1_fit = self._evaluate(child1_chrom)
                child2_fit = self._evaluate(child2_chrom)

                next_pop.append({'chromosome': child1_chrom, 'fitness': child1_fit})
                if len(next_pop) < self.pop_size:
                    next_pop.append({'chromosome': child2_chrom, 'fitness': child2_fit})

            # Dodaj elity
            if elites:
                # Zastępujemy najsłabszych w next_pop elitami lub łączymy i sortujemy
                # Tutaj – prosta metoda: wstawiamy je i przycinamy do pop_size
                next_pop.extend(elites)
                # Sortujemy i bierzemy pop_size najlepszych (jeśli maximize)
                next_pop = sorted(next_pop, key=lambda ind: ind['fitness'], reverse=True)[:self.pop_size]

            # Aktualizujemy populację
            self.population = next_pop

            # Opcjonalnie: diagnostyka / print
            best_ind = max(self.population, key=lambda ind: ind['fitness'])
            print(f"Epoch {epoch+1}/{self.epochs} | Best fitness: {best_ind['fitness']}")

        # Po zakończeniu zwracamy najlepszego osobnika
        best_ind = max(self.population, key=lambda ind: ind['fitness'])
        return best_ind

    def get_best_individual(self):
        """
        Zwraca najlepszego osobnika z bieżącej populacji.
        """
        return max(self.population, key=lambda ind: ind['fitness'])
