import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def get_optimal_thread_count(cpu_cores, max_threads=36):
    """
    Determine the optimal thread count based on CPU cores.
    For I/O-bound operations, we can use more threads than cores.
    """
    return min(cpu_cores * 4, max_threads)

def run_threaded_operation(operation, items, num_threads):
    """
    Run an operation on items using specified number of threads.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        return list(executor.map(operation, items))

def benchmark_threading(operation, items, cpu_cores):
    """
    Benchmark the operation with different thread counts and return the optimal.
    """
    optimal_threads = get_optimal_thread_count(cpu_cores)
    
    print(f"CPU cores: {cpu_cores}")
    print(f"Estimated optimal threads: {optimal_threads}")
    
    # Single-threaded baseline
    start = time.time()
    single_result = [operation(item) for item in items]
    single_time = time.time() - start
    print(f"Single-threaded time: {single_time:.2f} seconds")
    
    best_time = single_time
    best_threads = 1
    
    # Test different thread counts
    for thread_count in [4, 8, 12, 16, 20, 24, 28, 32, 36]:
        if thread_count > optimal_threads:
            break
        
        start = time.time()
        multi_result = run_threaded_operation(operation, items, thread_count)
        multi_time = time.time() - start
        speedup = single_time / multi_time
        
        print(f"Threads: {thread_count}, Time: {multi_time:.2f} seconds, Speedup: {speedup:.2f}x")
        
        if multi_time < best_time:
            best_time = multi_time
            best_threads = thread_count
    
    print(f"\nOptimal thread count: {best_threads}")
    print(f"Best speedup: {single_time / best_time:.2f}x")
    
    return best_threads

# Example usage
if __name__ == "__main__":
    # Simulated I/O-bound operation
    def io_bound_operation(x):
        time.sleep(0.1)  # Simulate I/O operation
        return x * 2

    items = list(range(1000))  # 1000 items to process
    cpu_cores = 8  # Adjust this based on your system

    optimal_threads = benchmark_threading(io_bound_operation, items, cpu_cores)
    
    print("\nUsing optimal thread count for final run:")
    start = time.time()
    final_result = run_threaded_operation(io_bound_operation, items, optimal_threads)
    final_time = time.time() - start
    print(f"Final run time: {final_time:.2f} seconds")
