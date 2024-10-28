import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Auxiliar functions
class Generation:
    def __init__(self, seed, elapsed_time, gen):
        """
        arrival_time: The time when the request arrived. 
         first_scheduled_time: The time when the request was first scheduled. 
         first_token_time: The time when the first token was generated. 
         time_in_queue: The time the request spent in the queue. 
         finished_time: The time when the request was finished. 
        """
        # Elapsed time its based on vLLM values finished_time - arrival_time
        self.seed = seed
        self.elapsed_time = elapsed_time
        self.gen = gen

    def load(model, experiment, samples, max_tokens):
        model_foldername = model.replace("/","::")
        with open(f"./samples/{model_foldername}/{experiment}/{samples}e_{max_tokens}t.jsonl", "r") as file:
            results = []
            for line in file:
                result = json.loads(line, object_hook=gen_encoder)
                results.append(result)
            return results
        
    def __str__(self):
        print = json.dumps(self.__dict__)
        return print
    
    def __repr__(self):
        print = json.dumps(self.__dict__)
        return print

class Result:
    def __init__(self, seeds: list[int], elapsed_time_gen: list[int], syntax_validation: list[int], semantic_validation: list[int]):
        self.model = MODEL.replace("/","::")
        self.experiment_type = EXPERIMENT_TYPE
        self.samples = SAMPLES
        self.max_tokens= MAX_TOKENS
        self.seeds = [seed for seed in seeds[:SAMPLES]]
        self.elapsed_time_gen = [elapsed for elapsed in elapsed_time_gen[:SAMPLES]]
        self.syntax_validation = syntax_validation
        self.semantic_validation = semantic_validation

    def __init__(self, model:str, experiment_type: str, samples:int, max_tokens:int, seeds: list[int], elapsed_time_gen: list[int], syntax_validation: list[int], semantic_validation: list[int]):
        self.model = model
        self.experiment_type = experiment_type
        self.samples = samples
        self.max_tokens= max_tokens
        self.seeds = [seed for seed in seeds[:samples]]
        self.elapsed_time_gen = [elapsed for elapsed in elapsed_time_gen[:samples]]
        self.syntax_validation = syntax_validation
        self.semantic_validation = semantic_validation

    def save(self):
        with open(f"./results/{self.model}/{self.experiment_type}/{self.samples}e_{self.max_tokens}t.jsonl", 'w') as file:
            json_line = json.dumps(self.__dict__)
            file.write(json_line + "\n")
        
    
    def load(model, experiment, samples, max_tokens):
        model_foldername = model.replace("/","::")
        with open(f"./results/{model_foldername}/{experiment}/{samples}e_{max_tokens}t.jsonl", "r") as file:
            for line in file:
                results = json.loads(line, object_hook=result_encoder)
            return results
        
    def __str__(self):
        print = json.dumps(self.__dict__)
        return print
    
    def __repr__(self):
        print = json.dumps(self.__dict__)
        return print
    
def result_encoder(r):
    return Result(model=r['model'], experiment_type=r['experiment_type'], samples=r['samples'], max_tokens=r['max_tokens'],
                    seeds=r['seeds'], elapsed_time_gen=r['elapsed_time_gen'], syntax_validation= r['syntax_validation'], semantic_validation=r['semantic_validation'])

def gen_encoder(g):
    return Generation(seed=g['seed'], elapsed_time=g['elapsed_time'], gen=g['gen'])

def fixed_seeds():
    # Taking the 100 samples seed from the first 100 samples experiment
    fixed_seeds = []
    with open(f"./seeds/100.jsonl", "r") as file:
        for line in file:
            fixed_seeds = json.loads(line)
    return fixed_seeds

# Function to plot GPU usage and memory usage and save the plots
def plot_gpu_monitoring(results, save_path):
    # Extract data from the results dictionary
    gpu_usage = results.get('gpu_usage', [])
    memory_usage = results.get('memory_usage', [])
    
    # Generate time points based on the sampling interval (0.1 seconds)
    time_points = [i * 0.1 for i in range(len(gpu_usage))]

    # Plot GPU Usage
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, gpu_usage, label='GPU Usage (%)', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{save_path}_gpu_usage.png")
    plt.close()
    
    # Plot Memory Usage
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, memory_usage, label='Memory Usage (MB)', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    memory_usage_path = save_path
    plt.savefig(f"{save_path}_memory_usage.png")
    plt.close()

    print(f"Plots saved to {save_path}")

# Function to plot combined GPU usage and memory usage and save the plot
def plot_combined_gpu_monitoring(results, save_path):
    # Extract data from the results dictionary
    gpu_usage = results.get('gpu_usage', [])
    memory_usage = results.get('memory_usage', [])
    
    # Generate time points based on the sampling interval (0.1 seconds)
    time_points = [i * 0.1 for i in range(len(gpu_usage))]

    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot GPU Usage on the first y-axis
    ax1.set_xlabel('Tiempo (Segundos)')
    ax1.set_ylabel('Uso de GPU (%)', color='blue')
    ax1.plot(time_points, gpu_usage, label='Uso de GPU (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Create a second y-axis for Memory Usage
    ax2 = ax1.twinx()
    ax2.set_ylabel('Uso de memoria (MB)', color='green')
    ax2.plot(time_points, memory_usage, label='Uso de memoria (MB)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title and legend
    plt.title('Uso de GPU y de memoria')
    fig.tight_layout()  # Adjust layout to prevent clipping

    # Save the plot
    combined_plot_path = save_path
    plt.savefig(combined_plot_path)
    plt.close()

    print(f"Combined plot saved to {combined_plot_path}")

# Reading results and gens
SAMPLES = 100
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL = "study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8"
experiments = ["nogbd","gbd","gbd+fewshots"]
tokens = [50,100,200] 
total_results = {}
total_gens = {}
total_gpu_monitoring = {}

seeds = fixed_seeds()
model_foldername = MODEL.replace("/","::")

for experiment in experiments:
    EXPERIMENT_TYPE = experiment
    for token in tokens:
        MAX_TOKENS = token
        print(f"==== Results for type experiment: {experiment} and max_tokens: {token} ====")
        results = Result.load(MODEL, experiment, SAMPLES, token)
        total_results[f"{experiment}-{token}"] = results
        
        gens = Generation.load(MODEL, experiment, SAMPLES, token)
        total_gens[f"{experiment}-{token}"] = gens

        #print(str(total_results[f"{experiment}-{token}"].syntax_validation.count(1)))
        #print(str(total_results[f"{experiment}-{token}"].semantic_validation.count(1)))


        # Elapsed time and more metrics
        elapsed_time_per_experiment = total_results[f"{experiment}-{token}"].elapsed_time_gen
        print(sum(elapsed_time_per_experiment))
        import statistics
        print(statistics.mean(elapsed_time_per_experiment))

        times_series = pd.Series(elapsed_time_per_experiment)
        rolling_mean = times_series.rolling(window=3).mean()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot(times_series)
        plt.title('Diagrama de cajas para el tiempo por generación')
        plt.ylabel('Tiempo (segundos)')
        plt.xlabel('Generaciones')

        # Plot 2: Line Plot with Rolling Mean
        plt.subplot(1, 2, 2)
        plt.plot(times_series, label='Tiempo')
        plt.plot(rolling_mean, label='Medias móviles (ventana=3)', color='orange', linestyle='--')
        plt.title('Tiempo por generación con Medias móviles')
        plt.xlabel('Generación')
        plt.ylabel('Tiempo (segundos)')
        plt.legend()

        # Show and Save
        plt.tight_layout()
        os.makedirs(f"./results/{model_foldername}/{experiment}/times/", exist_ok=True)
        plt.savefig(f"./results/{model_foldername}/{experiment}/times/{experiment}_{token}_time.png")
        plt.show()


        # Reading gpu monitoring
        with open(f"./results/{model_foldername}/{experiment}/gpu/{SAMPLES}e_{token}t_gpu.jsonl", "r") as file:
            for line in file:
                gpu_monitoring = json.loads(line)
                total_gpu_monitoring[f"{experiment}-{token}"] = gpu_monitoring
                #plot_gpu_monitoring(total_gpu_monitoring[f"{experiment}-{token}"], f"./results/{model_foldername}/{experiment}/gpu/{experiment}_{token}")
                #plot_combined_gpu_monitoring(total_gpu_monitoring[f"{experiment}-{token}"],f"./results/{model_foldername}/{experiment}/gpu/{experiment}_{token}_gpu_combined.png")


            

print("===============================================")
#print(total_gpu_monitoring[f"gbd-50"])

# Ploting gpu data




#print(total_results[f"gbd+fewshots-200"].elapsed_time_gen)
#plot_gpu_monitoring(total_results[f"gbd+fewshots-200"].elapsed_time_gen, "./another_plot/")
#plot_combined_gpu_monitoring(total_gpu_monitoring[f"gbd+fewshots-200"], "./another_plot/")




        
