import pandas as pd
import time
import threading

filtered_result = []
sorted_result = []

def filter_data_thread(data, threshold=1000):
    global filtered_result
    filtered_result = [d for d in data if d > threshold]

def sort_data_thread():
    global filtered_result, sorted_result
    sorted_result = sorted(filtered_result)

def process_threading(data, threshold=1000):
    global filtered_result, sorted_result
    filtered_result = []
    sorted_result = []

    start_time = time.time()

    t1 = threading.Thread(target=filter_data_thread, args=(data, threshold))


    t1.start()
    t1.join()  

    t2 = threading.Thread(target=sort_data_thread)
    t2.start()
    t2.join()  

    end_time = time.time()
    return sorted_result, end_time - start_time

def load_data(path='train.csv'):
    df = pd.read_csv(path)
    trip_duration = df['trip_duration'].tolist()
    return trip_duration

def run_experiments():
    data = load_data()
    data_sizes = [0.25, 0.5, 0.75, 1.0]

    print("Threading Processing Results:")
    for size in data_sizes:
        subset_len = int(len(data) * size)
        subset = data[:subset_len]
        _, duration = process_threading(subset)
        print(f"{int(size*100)}% data -> Time: {duration:.4f} seconds")

if __name__ == '__main__':
    run_experiments()
