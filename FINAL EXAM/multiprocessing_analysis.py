import pandas as pd
import time
from multiprocessing import Pool
import numpy as np

# --- INITIAL SETUP ---
FILE_PATH = 'NYC.csv'
# Number of processes to use, ideally <= your number of CPU cores
NUM_PROCESSES = 4 

# --- WORKER FUNCTIONS (EXECUTED IN SEPARATE PROCESSES) ---
def worker_sort(data_chunk):
    """Sorts a chunk of data."""
    return data_chunk.sort_values()

def worker_filter(data_chunk):
    """Filters a chunk of data."""
    threshold = 1000  # Threshold defined inside the worker
    return data_chunk[data_chunk > threshold]

# --- MAIN FUNCTIONS (MANAGING THE PROCESS POOL) ---
def multiprocessing_sort(data, num_processes):
    """Measures sorting time using a process pool."""
    start_time = time.time()
    # 'with' statement ensures the pool is properly closed
    with Pool(processes=num_processes) as pool:
        chunks = np.array_split(data, num_processes)
        # 'map' applies the worker function to each chunk in parallel
        pool.map(worker_sort, chunks)
    end_time = time.time()
    return end_time - start_time

def multiprocessing_filter(data, num_processes):
    """Measures filtering time using a process pool."""
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        chunks = np.array_split(data, num_processes)
        pool.map(worker_filter, chunks)
    end_time = time.time()
    return end_time - start_time

# --- MAIN EXECUTION BLOCK ---
# This check is crucial for multiprocessing to work correctly on all platforms.
if __name__ == '__main__':
    print("Loading dataset...")
    try:
        df = pd.read_csv(FILE_PATH)
        trip_duration_data = df['trip_duration']
        print(f"Dataset loaded successfully. Total rows: {len(trip_duration_data)}")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{FILE_PATH}'.")
        exit()

    DATA_SPLITS = [0.25, 0.50, 0.75, 1.00]
    results = []
    
    print(f"\nStarting multiprocessing analysis with {NUM_PROCESSES} processes...")

    for split_ratio in DATA_SPLITS:
        split_size = int(len(trip_duration_data) * split_ratio)
        data_subset = trip_duration_data.iloc[:split_size]
        split_label = f"{int(split_ratio * 100)}%"
        
        print(f"\n--- Processing {split_label} of data ({len(data_subset)} rows) ---")
        
        time_for_sort = multiprocessing_sort(data_subset, NUM_PROCESSES)
        print(f"Sorting completed in {time_for_sort:.4f} seconds.")
        results.append({
            'Data Size': split_label,
            'Operation': 'Sorting',
            'Execution Time (s)': time_for_sort
        })
        
        time_for_filter = multiprocessing_filter(data_subset, NUM_PROCESSES)
        print(f"Filtering completed in {time_for_filter:.4f} seconds.")
        results.append({
            'Data Size': split_label,
            'Operation': 'Filtering',
            'Execution Time (s)': time_for_filter
        })

    print("\n======================================")
    print(f" MULTIPROCESSING ANALYSIS RESULTS ({NUM_PROCESSES} Processes) ")
    print("======================================")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("\nMultiprocessing analysis complete.")