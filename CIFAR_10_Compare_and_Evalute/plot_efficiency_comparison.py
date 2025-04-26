import matplotlib.pyplot as plt

# Metrics collected from your run
models = ['Teacher (ResNet-50)', 'Student (ResNet-18)']

model_sizes = [90.06, 42.73]        # in MB
parameters = [23528522, 11181642]   # number of parameters
inference_times = [17.29, 6.42]     # in milliseconds
carbon_emissions = [0.00013, 0.00005] # in kg CO2

# Plotting function
def plot_metric(title, values, ylabel, save_name):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, color=['blue', 'green'])
    
    # Annotate values on bars
    for idx, val in enumerate(values):
        plt.text(idx, val + (max(values) * 0.02), f'{val:.5f}' if ylabel == 'Carbon Emissions (kg CO₂)' else f'{val:.2f}', 
                 ha='center', fontweight='bold')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

# Plot all comparisons
plot_metric('Model Size Comparison', model_sizes, 'Size (MB)', 'model_size_comparison.png')
plot_metric('Parameter Count Comparison', parameters, 'Number of Parameters', 'parameter_comparison.png')
plot_metric('Inference Time Comparison', inference_times, 'Inference Time (ms)', 'inference_time_comparison.png')
plot_metric('Carbon Emissions Comparison', carbon_emissions, 'Carbon Emissions (kg CO₂)', 'carbon_emissions_comparison.png')

print("✅ All comparison plots saved!")