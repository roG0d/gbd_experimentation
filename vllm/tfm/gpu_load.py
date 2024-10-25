import torch
import time
import threading
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo

# Initialize NVML
nvmlInit()

# Number of GPUs to test
num_gpus = 1

# Function to monitor GPU usage
def monitor_gpu(device_id, stop_event, interval=0.01, results=None):
    # Handle to GPU
    handle = nvmlDeviceGetHandleByIndex(device_id)
    
    # Lists to store GPU usage metrics
    gpu_usage = []
    memory_usage = []
    
    start_time = time.time()
    while not stop_event.is_set():  # Continue until stop_event is set:
        # Get GPU utilization
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        
        # Append data to lists
        gpu_usage.append(util.gpu)         # GPU usage in %
        memory_usage.append(mem_info.used / 1024 ** 2)  # Memory usage in MB
        
        # Wait for the next interval
        time.sleep(interval)

    if results is not None:
        results['gpu_usage'] = gpu_usage
        results['memory_usage'] = memory_usage

# Function to create a dummy load on a GPU
def gpu_test(device_id, stop_event):
    device = torch.device(f'cuda:{device_id}')
    print(f"Testing GPU {device_id}...")

    try:
        # Allocate a large tensor
        size = (100000, 10000)
        tensor = torch.randn(size, device=device)

        # Perform dummy operations
        start_time = time.time()
        for _ in range(10):
            tensor = tensor * 2.0  # Multiply
            tensor = tensor + tensor  # Add
            tensor = tensor - tensor / 2.0  # Subtract

        # Check performance time
        end_time = time.time()
        print(f"GPU {device_id} passed. Time taken: {end_time - start_time:.2f} seconds")
        print(torch.cuda.memory_summary(device=device))

        # Free memory
        del tensor
        torch.cuda.empty_cache()

        # Signal that the GPU load test is done
        stop_event.set()

    except Exception as e:
        print(f"GPU {device_id} encountered an error: {e}")

# Run tests on all available GPUs
if __name__ == "__main__":
    # Check if there are enough GPUs
    available_gpus = torch.cuda.device_count()
    interval = 0.001

    if available_gpus < num_gpus:
        print(f"Only {available_gpus} GPUs found, expected {num_gpus}")
    else:
        print(f"{available_gpus} GPUs found. Beginning test...")

    # Test each GPU
    for gpu_id in range(num_gpus):

        # Create an Event to signal when the GPU load test is finished
        stop_event = threading.Event()

        results = {}
        # Start the GPU load test and monitoring in parallel
        monitor_thread = threading.Thread(target=monitor_gpu, args=(gpu_id, stop_event, interval, results))
        monitor_thread.start()
        # Run the GPU load test
        gpu_test(gpu_id, stop_event)
    
        # Wait for monitoring to finish
        monitor_thread.join()

        # Output the collected GPU usage and memory usage data
        print("GPU Usage (%):", results["gpu_usage"])
        print("Memory Usage (MB):", results["memory_usage"])

        print("Max GPU usage: ", max(results["gpu_usage"]))
        print("Max Memory Usage (MB): ", max(results["memory_usage"]))



    

    

    

    