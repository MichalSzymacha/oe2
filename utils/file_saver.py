import csv

def save_results_to_csv(filename, history_best, history_mean, history_std):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Iteration", "Best", "Mean", "Std"])
        for i, (b, m, s) in enumerate(zip(history_best, history_mean, history_std)):
            writer.writerow([i+1, b, m, s])
