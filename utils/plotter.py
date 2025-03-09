import matplotlib.pyplot as plt

def plot_iterations(best_list, mean_list, std_list):
    # Wykres wartości najlepszej
    plt.figure()
    plt.title("Najlepsza wartość funkcji w kolejnych epokach")
    plt.plot(best_list)
    plt.xlabel("Epoka")
    plt.ylabel("Fitness (lub wartość funkcji)")

    plt.figure()
    plt.title("Średnia wartość funkcji i odchylenie standardowe")
    plt.plot(mean_list, label="Średnia")
    plt.plot(std_list, label="Odchylenie standardowe")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość")
    plt.legend()

    plt.show()
