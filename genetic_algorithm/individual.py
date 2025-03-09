import numpy as np

class Individual:
    """
    Klasa reprezentująca pojedynczego osobnika w populacji
    z binarną reprezentacją chromosomu.
    """

    def __init__(self, chromosome: np.ndarray, precision: float, var_bounds: list):
        """
        :param chromosome: numpy array z wartościami 0/1 (binarna reprezentacja)
        :param precision: dokładność zakodowania zmiennej (np. liczba bitów na zmienną)
        :param var_bounds: lista krotek (min_val, max_val) dla każdej zmiennej
        """
        self.chromosome = chromosome
        self.precision = precision
        self.var_bounds = var_bounds
        self.fitness = None  # wartość funkcji (lub ocena)

    def decode(self):
        """
        Z binarnej reprezentacji odczytujemy rzeczywiste wartości zmiennych.
        Zakładamy np., że 1 zmienna = kilkanaście bitów i mamy N zmiennych.
        """
        # Przykład: załóżmy, że mamy stałą liczbę bitów per zmienna
        # i znamy liczbę zmiennych na podstawie var_bounds.
        decoded_values = []
        bits_per_var = int(len(self.chromosome) / len(self.var_bounds))
        
        for i, bounds in enumerate(self.var_bounds):
            min_val, max_val = bounds
            # Wydziel fragment chromosomu dla zmiennej i
            var_bits = self.chromosome[i*bits_per_var:(i+1)*bits_per_var]
            # Zamiana binarnych bitów na liczbę całkowitą
            int_val = 0
            for bit in var_bits:
                int_val = (int_val << 1) | bit
            # Normalizacja do zakresu
            # np. mapujemy [0, (2^bits_per_var)-1] -> [min_val, max_val]
            max_int = (1 << bits_per_var) - 1  # 2^bits_per_var - 1
            real_val = min_val + (max_val - min_val) * (int_val / max_int)
            decoded_values.append(real_val)

        return np.array(decoded_values, dtype=float)
