import multiprocessing
import hashlib
import json
import os
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import time
from genetic_algorithm.GeneticAlgorithm import GeneticAlgorithm
from functions.function_selector import (
    get_function_by_name,
    get_suggested_bounds,
    FUNCTIONS_MAP,
)
from utils.file_saver import save_results_to_csv, save_results_to_json
from utils.plotter import plot_iterations, plot_mean_std


# Funkcja uruchamiająca pojedynczy test w osobnym procesie
def run_single_test_mp(args):
    # args: (config, i, parameters)
    config, i, parameters = args
    config_id = config.get("config_id", "config")
    temp_dir = os.path.join("temp_results", f"{config_id}_{i}")
    os.makedirs(temp_dir, exist_ok=True)
    start_time = time.time()

    # Tworzymy instancję GeneticAlgorithm z przekazanymi parametrami
    ga = GeneticAlgorithm(
        var_bounds=parameters["bounds"],
        vars_number=parameters["vars_number"],
        pop_size=parameters["pop_size"],
        epochs=parameters["epochs"],
        selection_method=parameters["selection_method"],
        selection_percentage=parameters["selection_percentage"],
        tournament_size=parameters["tournament_size"],
        crossover_method=parameters["crossover_method"],
        crossover_prob=parameters["crossover_prob"],
        mutation_method=parameters["mutation_method"],
        mutation_prob=parameters["mutation_prob"],
        elite_percentage=parameters["elite_percentage"],
        objective_function=parameters["objective_function"],
        maximize=parameters["maximize"],
    )

    best_solution = ga.run(temp_dir)
    end_time = time.time()
    run_time = end_time - start_time

    result_dict = {"best_solution_vector": best_solution, "time": run_time}
    save_results_to_json(os.path.join(temp_dir, "wyniki.json"), result_dict)

    return {"best_solution": best_solution, "time": run_time, "run_time": run_time}


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Genetic Algorithm")
        self.geometry("500x650")

        # --- Wybór funkcji benchmarkowej ---
        self.function_var = tk.StringVar(value="ackley")
        self.create_dropdown(
            "Choose function", list(FUNCTIONS_MAP.keys()), self.function_var
        )

        self.num_parameters_var = tk.IntVar(value=10)
        self.create_input_field("Number of parameters", self.num_parameters_var)

        # --- Konfiguracja pól wejściowych ---
        self.begin_range_var = tk.DoubleVar(value=-32)
        self.end_range_var = tk.DoubleVar(value=32)
        self.create_input_field("Begin of the range", self.begin_range_var)
        self.create_input_field("End of the range", self.end_range_var)


        self.population_var = tk.IntVar(value=100)
        self.epochs_var = tk.IntVar(value=100)
        self.create_input_field("Population", self.population_var)
        self.create_input_field("Epochs", self.epochs_var)

        self.elite_percentage_var = tk.DoubleVar(value=0.01)
        self.cross_prob_var = tk.DoubleVar(value=0.5)
        self.mutation_prob_var = tk.DoubleVar(value=0.3)
        self.create_input_field("Elite percentage", self.elite_percentage_var)
        self.create_input_field("Cross probability", self.cross_prob_var)
        self.create_input_field("Mutation probability", self.mutation_prob_var)

        # --- Wybór metody selekcji ---
        self.selection_method_var = tk.StringVar(value="tournament")
        self.create_dropdown(
            "Selection method",
            ["roulette", "tournament", "best"],
            self.selection_method_var,
        )
        self.selection_percentage_var = tk.DoubleVar(value=0.5)
        self.create_input_field("Selection percentage", self.selection_percentage_var)
        self.tournament_size_var = tk.IntVar(value=5)
        self.create_input_field("Tournament size", self.tournament_size_var)

        # --- Wybór metody krzyżowania ---
        self.crossover_method_var = tk.StringVar(value="arithmetical")
        self.create_dropdown(
            "Crossover method",
            ["arithmetical", "linear", "blendalpha", "blendalphabeta", "average"],
            self.crossover_method_var,
        )

        # --- Wybór metody mutacji ---
        self.mutation_method_var = tk.StringVar(value="uniform")
        self.create_dropdown(
            "Mutation method",
            ["uniform", "gaussian"],
            self.mutation_method_var,
        )

        # --- Minimalizacja / Maksymalizacja ---
        self.maximize_var = tk.BooleanVar(value=False)
        tk.Radiobutton(
            self, text="Minimization", variable=self.maximize_var, value=False
        ).pack()
        tk.Radiobutton(
            self, text="Maximization", variable=self.maximize_var, value=True
        ).pack()

        # --- Przycisk Start ---
        start_button = tk.Button(self, text="Start", command=self.run_ga)
        start_button.pack(pady=10)

        # --- Przycisk Automate Testing ---
        auto_test_button = tk.Button(
            self, text="Automate Testing", command=self.automate_testing
        )
        auto_test_button.pack(pady=10)

        # --- Pole na wyniki ---
        self.result_label = tk.Label(
            self, text="Best solution: -", font=("Arial", 12), wraplength=480
        )
        self.result_label.pack(pady=10)
        copy_button = tk.Button(self, text="Copy result", command=self.copy_result)
        copy_button.pack(pady=5)

    def copy_result(self):
        text = self.result_label.cget("text")
        self.clipboard_clear()
        self.clipboard_append(text)

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
        dropdown = ttk.Combobox(
            frame, textvariable=variable, values=options, state="readonly"
        )
        dropdown.pack(side="right", fill="x", expand=True)

    def run_ga(self):
        print("Uruchamianie algorytmu genetycznego...")
        function_name = self.function_var.get()
        num_vars = self.num_parameters_var.get()
        objective_function = get_function_by_name(function_name, num_vars)
        bounds = (self.begin_range_var.get(), self.end_range_var.get())
        pop_size = self.population_var.get()
        epochs = self.epochs_var.get()
        selection_method = self.selection_method_var.get()
        selection_percentage = self.selection_percentage_var.get()
        crossover_method = self.crossover_method_var.get()
        mutation_method = self.mutation_method_var.get()
        crossover_prob = self.cross_prob_var.get()
        mutation_prob = self.mutation_prob_var.get()
        elite_percentage = self.elite_percentage_var.get()
        maximize = self.maximize_var.get()
        tournament_size = self.tournament_size_var.get()

        ga = GeneticAlgorithm(
            var_bounds=bounds,
            vars_number=num_vars,
            pop_size=pop_size,
            epochs=epochs,
            selection_method=selection_method,
            selection_percentage=selection_percentage,
            tournament_size=tournament_size,
            crossover_method=crossover_method,
            crossover_prob=crossover_prob,
            mutation_method=mutation_method,
            mutation_prob=mutation_prob,
            elite_percentage=elite_percentage,
            objective_function=objective_function,
            maximize=maximize,
        )

        params = {
            "bounds": bounds,
            "pop_size": pop_size,
            "epochs": epochs,
            "selection_method": selection_method,
            "selection_percentage": selection_percentage,
            "crossover_method": crossover_method,
            "mutation_method": mutation_method,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
            "elite_percentage": elite_percentage,
            "maximize": maximize,
            "tournament_size": tournament_size,
        }
        params_str = str(sorted(params.items()))
        hash_digest = hashlib.md5(params_str.encode()).hexdigest()[:8]
        print(f"Hash: {hash_digest}", params_str)
        root_dir = f"results/{self.function_var.get()}_{hash_digest}"
        os.makedirs(root_dir, exist_ok=True)
        dir_name = f"{root_dir}/{len(os.listdir(root_dir))}"
        os.makedirs(dir_name, exist_ok=True)
        time_start = time.time()
        best_solution = ga.run(dir_name)
        time_stop = time.time()
        self.result_label.config(
            text=f"Best solution: {best_solution} \n Time: {time_stop - time_start}s"
        )
        result_dict = {
            "best_solution_vector": best_solution,
            "time": time_stop - time_start,
        }
        save_results_to_json(f"{dir_name}/wyniki.json", result_dict)
        plot_iterations(
            f"{dir_name}/wyniki.csv", f"{dir_name}/wykres_iteracji.png", True
        )
        plot_mean_std(
            f"{dir_name}/wyniki.csv", f"{dir_name}/wykres_sredniej_odchylenia.png", True
        )

    def run_ga_with_dir(self, dir_name):
        function_name = self.function_var.get()
        num_vars = self.num_parameters_var.get()
        objective_function = get_function_by_name(function_name, num_vars)
        bounds = (self.begin_range_var.get(), self.end_range_var.get())
        pop_size = self.population_var.get()
        epochs = self.epochs_var.get()
        selection_method = self.selection_method_var.get()
        selection_percentage = self.selection_percentage_var.get()
        tournament_size = self.tournament_size_var.get()
        crossover_method = self.crossover_method_var.get()
        mutation_method = self.mutation_method_var.get()
        crossover_prob = self.cross_prob_var.get()
        mutation_prob = self.mutation_prob_var.get()
        elite_percentage = self.elite_percentage_var.get()
        maximize = self.maximize_var.get()

        ga = GeneticAlgorithm(
            var_bounds=bounds,
            vars_number=num_vars,
            pop_size=pop_size,
            epochs=epochs,
            selection_method=selection_method,
            selection_percentage=selection_percentage,
            tournament_size=tournament_size,
            crossover_method=crossover_method,
            crossover_prob=crossover_prob,
            mutation_method=mutation_method,
            mutation_prob=mutation_prob,
            elite_percentage=elite_percentage,
            objective_function=objective_function,
            maximize=maximize,
        )
        params = {
            "bounds": bounds,
            "pop_size": pop_size,
            "epochs": epochs,
            "selection_method": selection_method,
            "selection_percentage": selection_percentage,
            "crossover_method": crossover_method,
            "mutation_method": mutation_method,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
            "elite_percentage": elite_percentage,
            "maximize": maximize,
            "tournament_size": tournament_size,
        }
        params_str = str(sorted(params.items()))
        hash_digest = hashlib.md5(params_str.encode()).hexdigest()[:8]
        print(f"Hash: {hash_digest}", params_str)
        root_dir = f"results/{self.function_var.get()}_{hash_digest}"
        os.makedirs(root_dir, exist_ok=True)
        dir_name = f"{root_dir}/{len(os.listdir(root_dir))}"
        os.makedirs(dir_name, exist_ok=True)
        time_start = time.time()
        best_solution = ga.run(dir_name)
        time_stop = time.time()
        self.result_label.config(
            text=f"Best solution: {best_solution} \n Time: {time_stop - time_start}s"
        )
        result_dict = {
            "best_solution_vector": best_solution,
            "time": time_stop - time_start,
        }
        save_results_to_json(f"{dir_name}/wyniki.json", result_dict)
        plot_iterations(
            f"{dir_name}/wyniki.csv", f"{dir_name}/wykres_iteracji.png", True
        )
        plot_mean_std(
            f"{dir_name}/wyniki.csv",
            f"{dir_name}/wykres_sredniej_odchylenia.png",
            True,
        )
        return best_solution

    def automate_testing(self):
        def run_tests():
            # Upewnij się, że folder "temp_results" istnieje
            os.makedirs("temp_results", exist_ok=True)
            with open("all_configs.json", "r") as f:
                configurations = json.load(f)

            summary_results = []
            total_configs = len(configurations)
            config_counter = 0

            for config in configurations:
                config_counter += 1
                # Uaktualniamy pola GUI według konfiguracji
                self.begin_range_var.set(config["bounds"][0])
                self.end_range_var.set(config["bounds"][1])
                self.population_var.set(config["pop_size"])
                self.epochs_var.set(config["epochs"])
                self.selection_method_var.set(config["selection_method"])
                self.selection_percentage_var.set(config["selection_percentage"])
                self.tournament_size_var.set(config["tournament_size"])
                self.crossover_method_var.set(config["crossover_method"])
                self.cross_prob_var.set(config["crossover_prob"])
                self.mutation_method_var.set(config["mutation_method"])
                self.mutation_prob_var.set(config["mutation_prob"])
                self.elite_percentage_var.set(config["elite_percentage"])
                self.maximize_var.set(config["maximize"])
                self.update_idletasks()

                config_results = []
                total_time = 0.0

                for i in range(10):
                    # Tworzymy unikalny folder tymczasowy dla uruchomienia testu
                    config_id = config.get("config_id", "config")
                    temp_dir = os.path.join("temp_results", f"{config_id}_{i}")
                    os.makedirs(temp_dir, exist_ok=True)

                    # Wywołujemy run_ga_with_dir, która generuje wyniki i wykresy tak samo jak run_ga
                    start_time = time.time()
                    best_solution = self.run_ga_with_dir(temp_dir)
                    end_time = time.time()
                    run_time = end_time - start_time
                    total_time += run_time

                    # Odczytujemy zapisany wynik (wyniki.json)
                    wyniki_path = os.path.join(temp_dir, "wyniki.json")
                    try:
                        with open(wyniki_path, "r") as jf:
                            result_data = json.load(jf)
                    except Exception:
                        result_data = {"best_solution_vector": None, "time": run_time}

                    config_results.append(
                        {
                            "best_solution": result_data.get("best_solution_vector"),
                            "time": result_data.get("time"),
                            "run_time": run_time,
                        }
                    )

                    print(f"Konfiguracja {config_id}: test {i+1}/10 zakończony.")

                avg_time = total_time / 10.0

                summary_results.append(
                    {
                        "configuration": config,
                        "results": config_results,
                        "avg_time": avg_time,
                    }
                )

                print(
                    f"Konfiguracja {config.get('config_id', 'config')} ukończona ({config_counter}/{total_configs})."
                )

            with open("automated_test_summary.json", "w") as outfile:
                json.dump(summary_results, outfile, indent=4)

            print(
                "Automated testing completed. Summary saved to automated_test_summary.json"
            )

        # Uruchamiamy testy w osobnym wątku, by nie blokować GUI
        threading.Thread(target=run_tests, daemon=True).start()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Przydatne na Windows
    app = App()
    app.mainloop()
