import benchmark_functions as bf

# Mapa nazw funkcji do klas w bibliotece benchmark_functions
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
    "pits_and_holes": bf.PitsAndHoles
}

def get_function_by_name(name, n_dimensions=2, opposite=False):
    """
    Pobiera funkcję benchmarkową na podstawie nazwy.
    
    :param name: Nazwa funkcji (np. "sphere", "rastrigin").
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
        function_instance = function_class(n_dimensions=n_dimensions, opposite=opposite)
    except TypeError:
        # Niektóre funkcje są tylko 2D i nie akceptują `n_dimensions`
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
    :param n_dimensions: Liczba wymiarów.
    :return: Pozycja i wartość minimum lub None, jeśli brak danych.
    """
    function = get_function_by_name(name, n_dimensions)
    min_optimum = function.minimum

    if min_optimum:
        return min_optimum.position, min_optimum.score
    return None, None
