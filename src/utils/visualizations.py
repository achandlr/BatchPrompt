import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


def save_plot(description):
    description = description.replace("(", "_").replace(")", "_").replace(":", "_").replace(" ", "_")

    # Create directories if they don't exist
    if not os.path.exists(os.path.join("data", "visualizations")):
        os.makedirs(os.path.join("data", "visualizations"))

    # Save the plot
    plt.savefig(os.path.join("data", "visualizations", description))


def plot_batch_results(batch_sizes, accuracies, parsing_error_rates, title="Batch Prompting Results", sample_size=None):
    """
    Plot accuracies and parsing error rates against batch sizes.

    Parameters:
    - batch_sizes: List of batch sizes.
    - accuracies: List of accuracies corresponding to batch sizes.
    - parsing_error_rates: List of parsing error rates corresponding to batch sizes.
    - title: Title of the plot.
    - sample_size: The number of samples represented in the plot. Default is None.

    Returns:
    None
    """
    # Initialize the plot and primary y-axis
    fig, ax1 = plt.subplots()

    # Plot accuracies against batch sizes on the primary y-axis
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(batch_sizes, accuracies, 'o-', color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Set integer labels on the x-axis
    ax1.set_xticks(np.arange(min(batch_sizes), max(batch_sizes) + 1, step=1))
    # Set y-axis limits and ticks for ax1
    ax1.set_ylim([0, 1])
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    
    # Initialize the secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylim([0, 1])
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    # Plot parsing error rates against batch sizes on the secondary y-axis
    ax2.set_ylabel('Parsing Error Rate', color='tab:red')
    ax2.plot(batch_sizes, parsing_error_rates, 'x-', color='tab:red', label='Parsing Error Rate')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add labels, title, and legend
    if sample_size is not None:
        title += f" (Sample Size: {sample_size})"
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    # Show the plot
    # plt.show()

    # Save plot
    save_plot(title+".png")
    return



if __name__ == "__main__":
    with open(r"task_to_stats_rte_incomplete_50", "rb") as input_file:
        stats_to_task = pickle.load(input_file)

rte_stats = stats_to_task["rte"]
# rte_batched_model_sizes = rte_stats.keys()
# rte_accuracies = [rte_stats[batch_size]["stat"]['Accuracy'] for batch_size in rte_batched_model_sizes]
# rte_proportion_cant_parse = [rte_stats[batch_size]["stat"]['proportion_cant_parse'] for batch_size in rte_batched_model_sizes]
# rte_sample_size = 50

rte_batched_model_sizes = [1,2,4,8]
rte_accuracies = [.8,.78,.82,.8]
rte_proportion_cant_parse = [0,0,0,0]#[rte_stats[batch_size]["stat"]['proportion_cant_parse'] for batch_size in rte_batched_model_sizes]
rte_sample_size = 50

with open(r"task_to_stats_rte_gsm8k", "rb") as input_file:
        stats_to_task_gsm8k  = pickle.load(input_file)
gsm8k_stats = stats_to_task_gsm8k["GSM8K"]
gsm8k_batched_model_sizes = gsm8k_stats.keys()
gsm8k_accuracies = [gsm8k_stats[batch_size]["stat"]['Accuracy'] for batch_size in rte_batched_model_sizes]
gsm8k_proportion_cant_parse = [gsm8k_stats[batch_size]["stat"]['proportion_cant_parse'] for batch_size in rte_batched_model_sizes]
gsm8k_sample_size = 50

plot_batch_results(batch_sizes = rte_batched_model_sizes, accuracies = rte_accuracies, parsing_error_rates = rte_proportion_cant_parse, title="RTE Zero-Shot BatchPrompt Results", sample_size = rte_sample_size)

plot_batch_results(batch_sizes = gsm8k_batched_model_sizes, accuracies = gsm8k_accuracies, parsing_error_rates = gsm8k_proportion_cant_parse, title="GSM8K Zero-Shot BatchPrompt Results", sample_size = gsm8k_sample_size)