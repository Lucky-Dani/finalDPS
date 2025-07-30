import pandas as pd
import time

# --- INITIAL SETUP ---
FILE_PATH = 'NYC.csv'
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

# --- STEP 2: SEQUENTIAL PROCESSING FUNCTIONS ---
def sequential_sort(data):
    """Sorts data and returns execution time."""
    start_time = time.time()
    _ = data.sort_values()
    end_time = time.time()
    return end_time - start_time

def sequential_filter(data, threshold):
    """Filters data and returns execution time."""
    start_time = time.time()
    _ = data[data > threshold]
    end_time = time.time()
    return end_time - start_time

# --- STEP 3: EXECUTION & MEASUREMENT ---
results = []
print("\nStarting sequential analysis...")

for split_ratio in DATA_SPLITS:
    split_size = int(len(trip_duration_data) * split_ratio)
    data_subset = trip_duration_data.iloc[:split_size]
    split_label = f"{int(split_ratio * 100)}%"
    
    print(f"\n--- Processing {split_label} of data ({len(data_subset)} rows) ---")
    
    # Measure sorting time
    time_for_sort = sequential_sort(data_subset)
    print(f"Sorting completed in {time_for_sort:.4f} seconds.")
    results.append({
        'Data Size': split_label,
        'Operation': 'Sorting',
        'Execution Time (s)': time_for_sort
    })
    
    # Measure filtering time
    time_for_filter = sequential_filter(data_subset, FILTER_THRESHOLD)
    print(f"Filtering completed in {time_for_filter:.4f} seconds.")
    results.append({
        'Data Size': split_label,
        'Operation': 'Filtering',
        'Execution Time (s)': time_for_filter
    })

# --- STEP 4: DISPLAY RESULTS ---
print("\n======================================")
print("   SEQUENTIAL ANALYSIS RESULTS      ")
print("======================================")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("\nSequential analysis complete.")