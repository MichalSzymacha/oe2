import benchmark_functions as bf
from functions import cec2014

FUNCTIONS_MAP = {
    "sphere": bf.Hypersphere,
    "ellipsoid": bf.Hyperellipsoid,
    "schwefel": bf.Schwefel,
    "ackley": bf.Ackley,
    "michalewicz": bf.Michalewicz,
    "rastrigin": bf.Rastrigin,
    "rosenbrock": bf.Rosenbrock,
    "de_jong_3": bf.DeJong3,
    "de_jong_5": bf.DeJong5,
    "martin_gaddy": bf.MartinGaddy,
    "griewank": bf.Griewank,
    "easom": bf.Easom,
    "styblinski_tang": bf.StyblinskiTang,
    "mc_cormick": bf.McCormick,
    "rana": bf.Rana,
    "egg_holder": bf.EggHolder,
    "keane": bf.Keane,
    "schaffer_2": bf.Schaffer2,
    "himmelblau": bf.Himmelblau,
    "pits_and_holes": bf.PitsAndHoles,
    # Dodajemy funkcje z cec2014
    "cec2014_f1": cec2014.F12014,
    "cec2014_f2": cec2014.F22014,
    "cec2014_f3": cec2014.F32014,
    "cec2014_f4": cec2014.F42014,
    "cec2014_f5": cec2014.F52014,
    "cec2014_f6": cec2014.F62014,
    "cec2014_f7": cec2014.F72014,
    "cec2014_f8": cec2014.F82014,
    "cec2014_f9": cec2014.F92014,
    "cec2014_f10": cec2014.F102014,
    "cec2014_f11": cec2014.F112014,
    "cec2014_f12": cec2014.F122014,
    "cec2014_f13": cec2014.F132014,
    "cec2014_f14": cec2014.F142014,
    "cec2014_f15": cec2014.F152014,
    "cec2014_f16": cec2014.F162014,
    "cec2014_f17": cec2014.F172014,
    "cec2014_f18": cec2014.F182014,
    "cec2014_f19": cec2014.F192014,
    "cec2014_f20": cec2014.F202014,
    "cec2014_f21": cec2014.F212014,
    "cec2014_f22": cec2014.F222014,
    "cec2014_f23": cec2014.F232014,
    "cec2014_f24": cec2014.F242014,
    "cec2014_f25": cec2014.F252014,
    "cec2014_f26": cec2014.F262014,
    "cec2014_f27": cec2014.F272014,
}

def get_function_by_name(name, n_dimensions=2, opposite=False):
    """
    Pobiera funkcję benchmarkową na podstawie nazwy.
    
    :param name: Nazwa funkcji (np. "sphere", "rastrigin", "cec2014_f1").
    :param n_dimensions: Liczba zmiennych funkcji (dla funkcji wielowymiarowych).
    :param opposite: Jeśli True, zwraca funkcję jako maksymalizację zamiast minimalizacji.
    :return: Instancja funkcji benchmarkowej.
    """
    name = name.lower()
    
    if name not in FUNCTIONS_MAP:
        raise ValueError(f"Nieznana funkcja: {name}. Dostępne: {list(FUNCTIONS_MAP.keys())}")

    function_class = FUNCTIONS_MAP[name]

    # Sprawdzenie, czy funkcja obsługuje różne liczby wymiarów
    try:
        if name.startswith("cec2014"):
            function_instance = function_class(ndim=n_dimensions)
        else:
            function_instance = function_class(n_dimensions=n_dimensions, opposite=opposite)
    except TypeError:
        # Niektóre funkcje mogą być tylko 2D i nie akceptują `n_dimensions`
        function_instance = function_class(opposite=opposite)
    
    return function_instance

def evaluate_function(name, point, n_dimensions=2, opposite=False):
    """
    Oblicza wartość funkcji dla podanego punktu.

    :param name: Nazwa funkcji.
    :param point: Lista wartości dla zmiennych funkcji.
    :param n_dimensions: Liczba zmiennych (jeśli funkcja obsługuje różne liczby wymiarów).
    :param opposite: Czy zamienić funkcję na maksymalizację.
    :return: Wartość funkcji dla danego punktu.
    """
    function = get_function_by_name(name, n_dimensions, opposite)
    
    if len(point) != function.n_dimensions:
        raise ValueError(f"Podano {len(point)} wartości, ale funkcja {name} wymaga {function.n_dimensions} zmiennych.")

    return function(point)

def get_suggested_bounds(name, n_dimensions=2):
    """
    Pobiera sugerowane przedziały dla funkcji.

    :param name: Nazwa funkcji.
    :param n_dimensions: Liczba wymiarów.
    :return: Krotka (dolna_granica, górna_granica).
    """
    function = get_function_by_name(name, n_dimensions)
    return function.suggested_bounds()

def get_known_minimum(name, n_dimensions=2):
    """
    Pobiera znane minimum funkcji, jeśli jest dostępne.

    :param name: Nazwa funkcji.
    :param n_dimensions: Liczba zmiennych.
    :return: Pozycja i wartość minimum lub None, jeśli brak danych.
    """
    function = get_function_by_name(name, n_dimensions)
    min_optimum = function.minimum

    if min_optimum:
        return min_optimum.position, min_optimum.score
    return None, None

if __name__ == "__main__":
    # Przykładowe wywołania:
    print(bf.Hyperellipsoid(n_dimensions=2)([1, 2]))
    print(get_function_by_name("sphere", n_dimensions=2)([1, 2]))
    # Przykład użycia funkcji z cec2014:
    cec_func = get_function_by_name("cec2014_f1", n_dimensions=30)
    print(cec_func([0.0]*30))
