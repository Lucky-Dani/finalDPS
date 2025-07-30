import pandas as pd
import time
import threading
import numpy as np

# --- INITIAL SETUP ---
FILE_PATH = 'NYC.csv'
NUM_THREADS = 4  # Number of threads to use
DATA_SPLITS = [0.25, 0.50, 0.75, 1.00]
FILTER_THRESHOLD = 1000

# --- STEP 1: LOAD DATASET ---
print("Loading dataset...")
try:
    df = pd.read_csv(FILE_PATH)
    trip_duration_data = df['trip_duration']
    print(f"Dataset loaded successfully. Total rows: {len(trip_duration_data)}")
except FileNotFoundError:
    print(f"ERROR: File not found at '{FILE_PATH}'.")
    exit()

# --- STEP 2: THREADING PROCESSING FUNCTIONS ---

def worker_sort(data_chunk):
    """The task performed by each sorting thread."""
    _ = data_chunk.sort_values()

def worker_filter(data_chunk, threshold):
    """The task performed by each filtering thread."""
    _ = data_chunk[data_chunk > threshold]

def thread_based_sort(data, num_threads):
    """Manages threads for the sorting task."""
    start_time = time.time()
    chunks = np.array_split(data, num_threads)
    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=worker_sort, args=(chunk,))
        threads.append(thread)
        thread.start()
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    end_time = time.time()
    return end_time - start_time

def thread_based_filter(data, threshold, num_threads):
    """Manages threads for the filtering task."""
    start_time = time.time()
    chunks = np.array_split(data, num_threads)
    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=worker_filter, args=(chunk, threshold))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    end_time = time.time()
    return end_time - start_time

# --- STEP 3: EXECUTION & MEASUREMENT ---
results = []
print(f"\nStarting threading analysis with {NUM_THREADS} threads...")

for split_ratio in DATA_SPLITS:
    split_size = int(len(trip_duration_data) * split_ratio)
    data_subset = trip_duration_data.iloc[:split_size]
    split_label = f"{int(split_ratio * 100)}%"
    
    print(f"\n--- Processing {split_label} of data ({len(data_subset)} rows) ---")
    
    time_for_sort = thread_based_sort(data_subset, NUM_THREADS)
    print(f"Sorting completed in {time_for_sort:.4f} seconds.")
    results.append({
        'Data Size': split_label,
        'Operation': 'Sorting',
        'Execution Time (s)': time_for_sort
    })
    
    time_for_filter = thread_based_filter(data_subset, FILTER_THRESHOLD, NUM_THREADS)
    print(f"Filtering completed in {time_for_filter:.4f} seconds.")
    results.append({
        'Data Size': split_label,
        'Operation': 'Filtering',
        'Execution Time (s)': time_for_filter
    })

# --- STEP 4: DISPLAY RESULTS ---
print("\n======================================")
print(f"    THREADING ANALYSIS RESULTS ({NUM_THREADS} Threads)   ")
print("======================================")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\nThreading analysis complete.")