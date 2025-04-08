import csv
import os
import json


def clear_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "best", "mean", "std"])


def save_results_to_csv(filename, *args):
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(args)


def convert_to_json_serializable(obj):
    """
    Rekurencyjnie konwertuje obiekt do formatu, który można zapisać jako JSON.
    Jeśli napotka obiekt, którego nie potrafi zserializować, zwraca jego stringową reprezentację.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            str(key): convert_to_json_serializable(value) for key, value in obj.items()
        }
    else:
        # Dla obiektów nie serializowalnych, używamy ich stringowej reprezentacji
        return str(obj)


def save_results_to_json(filename, data):
    # Konwertujemy dane do formatu JSON-friendly
    json_data = convert_to_json_serializable(data)
    with open(filename, mode="w", newline="") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)
        jsonfile.write("\n")
