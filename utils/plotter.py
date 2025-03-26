import matplotlib.pyplot as plt
import pandas as pd

def plot_iterations(filename):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(filename)
    
    # Utworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['best'], marker='o', label='Best')
    
    # Dodanie etykiet oraz tytułu
    plt.xlabel('Epoch')
    plt.ylabel('Wartość funkcji celu')
    plt.title('Zależność wartości funkcji celu od kolejnych iteracji')
    plt.legend()
    plt.grid(True)
    
    # Wyświetlenie wykresu
    plt.show()

def plot_mean_std(filename):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(filename)
    
    # Utworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['mean'], marker='o', label='Średnia wartość funkcji celu')
    plt.plot(df['epoch'], df['std'], marker='o', label='Odchylenie standardowe')
    
    # Dodanie etykiet, tytułu oraz legendy
    plt.xlabel('Iteracja (epoch)')
    plt.ylabel('Wartość')
    plt.title('Średnia wartość funkcji celu i odchylenie standardowe w kolejnych iteracjach')
    plt.legend()
    plt.grid(True)
    
    # Wyświetlenie wykresu
    plt.show()