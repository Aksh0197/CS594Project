import matplotlib.pyplot as plt

# Replace these numbers with your actual outputs after running evaluate_efficiency_nlp.py
models = ['Teacher (BERT)', 'Student (DistilBERT)']

# Placeholder values (you will replace these after your evaluation)
model_sizes = [420.00, 250.00]        # in MB (example)
parameters = [109482240, 66897024] # number of parameters (example for BERT and DistilBERT)
inference_times = [35.2, 18.5]         # in milliseconds (example)
carbon_emissions = [0.00045, 0.00023]  # in kg CO₂ (example)

# Plotting function
def plot_metric(title, values, ylabel, save_name):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, color=['blue', 'green'])
    
    # Annotate values on bars
    for idx, val in enumerate(values):
        plt.text(idx, val + (max(values) * 0.02), 
                 f'{val:.5f}' if ylabel == 'Carbon Emissions (kg CO₂)' else f'{val:.2f}',
                 ha='center', fontweight='bold')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

# Plot all comparisons
plot_metric('Model Size Comparison (NLP)', model_sizes, 'Size (MB)', 'nlp_model_size_comparison.png')
plot_metric('Parameter Count Comparison (NLP)', parameters, 'Number of Parameters', 'nlp_parameter_comparison.png')
plot_metric('Inference Time Comparison (NLP)', inference_times, 'Inference Time (ms)', 'nlp_inference_time_comparison.png')
plot_metric('Carbon Emissions Comparison (NLP)', carbon_emissions, 'Carbon Emissions (kg CO₂)', 'nlp_carbon_emissions_comparison.png')

print("✅ NLP comparison plots saved!")