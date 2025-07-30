import pandas as pd
import time

def filter_data(data, threshold=1000):
    return [d for d in data if d > threshold]

def sort_data(data):
    return sorted(data)

def process_sequential(data, threshold=1000):
    start_time = time.time()
    filtered = filter_data(data, threshold)
    sorted_data = sort_data(filtered)
    end_time = time.time()
    return sorted_data, end_time - start_time

def load_data(path='train.csv'):
    df = pd.read_csv(path)
    trip_duration = df['trip_duration'].tolist()
    return trip_duration

def run_experiments():
    data = load_data()
    data_sizes = [0.25, 0.5, 0.75, 1.0]

    print("Sequential Processing Results:")
    for size in data_sizes:
        subset_len = int(len(data) * size)
        subset = data[:subset_len]
        _, duration = process_sequential(subset)
        print(f"{int(size*100)}% data -> Time: {duration:.4f} seconds")

if __name__ == '__main__':
    run_experiments()
