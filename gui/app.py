import tkinter as tk
from tkinter import ttk
import numpy as np
from genetic_algorithm.ga_manager import GAManager
from functions.function_selector import get_function_by_name, get_suggested_bounds, FUNCTIONS_MAP

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Genetic Algorithm")
        self.geometry("500x650")

        # --- Wybór funkcji benchmarkowej ---
        self.function_var = tk.StringVar(value="rastrigin")
        self.create_dropdown("Choose function", list(FUNCTIONS_MAP.keys()), self.function_var)

        self.num_parameters_var = tk.IntVar(value=2)
        self.create_input_field("Number of parameters", self.num_parameters_var)

        # --- Konfiguracja pól wejściowych ---
        self.begin_range_var = tk.DoubleVar(value=-500)
        self.end_range_var = tk.DoubleVar(value=500)
        self.create_input_field("Begin of the range", self.begin_range_var)
        self.create_input_field("End of the range", self.end_range_var)

        self.precision_var = tk.DoubleVar(value=0.0001)
        self.create_input_field("Precision", self.precision_var)

        self.population_var = tk.IntVar(value=15)
        self.epochs_var = tk.IntVar(value=1000)
        self.create_input_field("Population", self.population_var)
        self.create_input_field("Epochs", self.epochs_var)

        self.elite_percentage_var = tk.DoubleVar(value=0.15)
        self.cross_prob_var = tk.DoubleVar(value=0.5)
        self.mutation_prob_var = tk.DoubleVar(value=0.01)
        self.inversion_prob_var = tk.DoubleVar(value=0.015)
        self.create_input_field("Elite percentage", self.elite_percentage_var)
        self.create_input_field("Cross probability", self.cross_prob_var)
        self.create_input_field("Mutation probability", self.mutation_prob_var)
        self.create_input_field("Inversion probability", self.inversion_prob_var)

        # --- Wybór metody selekcji ---
        self.selection_method_var = tk.StringVar(value="roulette")
        self.create_dropdown("Selection method", ["roulette", "tournament", "best"], self.selection_method_var)

        self.tournament_size_var = tk.IntVar(value=3)
        self.create_input_field("Tournament size", self.tournament_size_var)

        # --- Wybór metody krzyżowania ---
        self.crossover_method_var = tk.StringVar(value="one_point")
        self.create_dropdown("Crossover method",
                             ["one_point", "two_point", "uniform", "grain"],
                             self.crossover_method_var)

        # --- Wybór metody mutacji ---
        self.mutation_method_var = tk.StringVar(value="one_point")
        self.create_dropdown("Mutation method",
                             ["boundary", "one_point", "two_points"], 
                             self.mutation_method_var)

        # --- Minimalizacja / Maksymalizacja ---
        self.maximize_var = tk.BooleanVar(value=False)
        tk.Radiobutton(self, text="Minimization", variable=self.maximize_var, value=False).pack()
        tk.Radiobutton(self, text="Maximization", variable=self.maximize_var, value=True).pack()

        # --- Przycisk Start ---
        start_button = tk.Button(self, text="Start", command=self.run_ga)
        start_button.pack(pady=10)

        # --- Pole na wyniki ---
        self.result_label = tk.Label(self, text="Best solution: -", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def create_input_field(self, label_text, variable):
        frame = tk.Frame(self)
        frame.pack(fill="x", padx=10, pady=2)

        label = tk.Label(frame, text=label_text, width=25, anchor="w")
        label.pack(side="left")

        entry = tk.Entry(frame, textvariable=variable)
        entry.pack(side="right", fill="x", expand=True)

    def create_dropdown(self, label_text, options, variable):
        frame = tk.Frame(self)
        frame.pack(fill="x", padx=10, pady=2)

        label = tk.Label(frame, text=label_text, width=25, anchor="w")
        label.pack(side="left")

        dropdown = ttk.Combobox(frame, textvariable=variable, values=options, state="readonly")
        dropdown.pack(side="right", fill="x", expand=True)

    def run_ga(self):
        print("Uruchamianie algorytmu genetycznego...")

        # Pobranie parametrów z GUI
        function_name = self.function_var.get()
        num_vars = self.num_parameters_var.get()

        # Pobranie funkcji benchmarkowej
        objective_function = get_function_by_name(function_name, num_vars)

        # Pobranie sugerowanych granic
        bounds = get_suggested_bounds(function_name, num_vars)
        
        pop_size = self.population_var.get()
        epochs = self.epochs_var.get()
        selection_method = self.selection_method_var.get()
        crossover_method = self.crossover_method_var.get()
        mutation_method = self.mutation_method_var.get()
        crossover_prob = self.cross_prob_var.get()
        mutation_prob = self.mutation_prob_var.get()
        inversion_prob = self.inversion_prob_var.get()
        elite_percentage = self.elite_percentage_var.get()
        maximize = self.maximize_var.get()
        tournament_size = self.tournament_size_var.get()

        # Uruchomienie algorytmu
        ga = GAManager(
            var_bounds=bounds,
            bits_per_var=16,  # Stała wartość precyzji binarnej
            pop_size=pop_size,
            epochs=epochs,
            selection_method=selection_method,
            crossover_method=crossover_method,
            crossover_prob=crossover_prob,
            mutation_method=mutation_method,
            mutation_prob=mutation_prob,
            inversion_prob=inversion_prob,
            elite_percentage=elite_percentage,
            objective_function=objective_function,
            maximize=maximize
        )

        elapsed_time = ga.run()
        best_solution = ga.population[0].decode()  # Pobranie najlepszego rozwiązania

        # Wyświetlenie wyniku w GUI
        self.result_label.config(text=f"Best solution: {best_solution}, Time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    app = App()
    app.mainloop()
