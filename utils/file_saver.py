import csv

def clear_file(filename):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "best", "mean", "std"])

def save_results_to_csv(filename, i, best, mean, std):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([i, best, mean, std])

