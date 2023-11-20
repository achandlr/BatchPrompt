import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
import pandas as pd
import random
def build_dataframe(task_to_stats):
    """
    Constructs a DataFrame from the task_to_stats dictionary, handling nested accuracy values.

    :param task_to_stats: Dictionary containing experiment statistics.
    :return: A DataFrame with all combinations of variables and their corresponding statistics.
    """
    data_for_df = []

    for task_name, k_shot_data in task_to_stats.items():
        for k_shot_size, batch_data in k_shot_data.items():
            for batch_size, model_data in batch_data.items():
                for model_name, stats in model_data.items():
                    accuracy_dict = stats.get('stat', {})
                    cannot_parse_proportion = accuracy_dict.get('cannot_parse_proportion', None)
                    none_is_skipped = accuracy_dict.get('none_is_skipped', {}).get('Accuracy', None)
                    none_is_wrong = accuracy_score([x if x!=None else -123456789 for x in stats["pred"]], stats["ground_truth"])
                    valid_answers = list(set(stats["ground_truth"]))
                    none_is_random = accuracy_score([x if x!=None else random.choice(valid_answers) for x in stats["pred"]], stats["ground_truth"])
                    # none_is_wrong = accuracy_dict.get('none_is_wrong', {}).get('Accuracy', None)
                    none_is_random = accuracy_dict.get('none_is_random', {}).get('Accuracy', None)

                    data_point = {
                        'Task Name': task_name,
                        'K-shot Size': k_shot_size,
                        'Batch Size': batch_size,
                        'Model Name': model_name,
                        'Cannot Parse Proportion': cannot_parse_proportion,
                        'Accuracy - Skipped': none_is_skipped,
                        'Accuracy - Wrong': none_is_wrong,
                        'Accuracy - Random': none_is_random
                    }
                    data_for_df.append(data_point)

    return pd.DataFrame(data_for_df)

# Create the DataFrame

# def extract_and_plot_stats(task_to_stats, control_vars, plot_var, plot_title='', x_label='', y_label=''):
#     """
#     Extracts a 2D list of data points and plots statistics based on controlled and variable parameters.

#     :param task_to_stats: Dictionary containing experiment statistics.
#     :param control_vars: Dictionary specifying variables to control and their values.
#     :param plot_var: Variable to plot.
#     :param plot_title: Title for the plot.
#     :param x_label: Label for the X-axis.
#     :param y_label: Label for the Y-axis.
#     """
#     # Extract and prepare data
#     data_for_plotting = []

#     for task_name, k_shot_data in task_to_stats.items():
#         if control_vars.get('task_name') and task_name != control_vars['task_name']:
#             continue

#         for k_shot_size, batch_data in k_shot_data.items():
#             if control_vars.get('k_shot_size') and k_shot_size != control_vars['k_shot_size']:
#                 continue

#             for batch_size, model_data in batch_data.items():
#                 if control_vars.get('batch_size') and batch_size != control_vars['batch_size']:
#                     continue

#                 for model_name, stats in model_data.items():
#                     if control_vars.get('model_name') and model_name != control_vars['model_name']:
#                         continue

#                     data_point = [task_name, k_shot_size, batch_size, model_name, stats.get(plot_var)]
#                     data_for_plotting.append(data_point)

#     # Convert to DataFrame for easier plotting
#     df = pd.DataFrame(data_for_plotting, columns=['Task Name', 'K-shot Size', 'Batch Size', 'Model Name', plot_var])
#     # Plotting
#     plt.figure(figsize=(12, 8))
#     sns.lineplot(data=df, x='Batch Size', y=plot_var, hue='Model Name', style='K-shot Size', markers=True)
#     plt.title(plot_title if plot_title else f'{plot_var} across different configurations')
#     plt.xlabel(x_label if x_label else 'Batch Size')
#     plt.ylabel(y_label if y_label else plot_var)
#     plt.legend(title='Models / K-shot Size')
#     plt.grid(True)
#     plt.show()
#     return

def plot_2_vars(df, x_var="Batch Size", y_var="Accuracy - Wrong", control_variables = {"K-shot Size": 1, "Model Name" : "LLAMA-2-70B"}):
    # Update this list based on available models
    assert "Model Name" in control_variables.keys()

    assert control_variables["Model Name"] in ['gpt-3.5-turbo-16k', 'LLAMA-2-70B']
    # Filtering the DataFrame
    filtered_df = df
    for key, value in control_variables.items():
        filtered_df = filtered_df[filtered_df[key] == value]


    # Creating the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=filtered_df, x=x_var, y=y_var, hue='Task Name', marker='o')

    # Dynamic title based on y-variable (accuracy or error rate)
    if "Accuracy" in y_var:
        plot_aspect = 'Accuracy'
        ylabel = 'Accuracy (%)'
    else:
        plot_aspect = 'Error Rate'
        ylabel = 'Error Rate (%)'

    title = f'{plot_aspect} over {x_var}'

    for key, value in control_variables.items():
        title += f" with {key} = {value}"
    
    
    ''' 
        # if "K-shot Size" in control_variables.keys():
        # k_shot = control_variables["K-shot Size"] #TODO: fix this
        # # plt.title(f'{plot_aspect} over {x_var} for {control_variables["Model Name"]} ({k_shot}-Shot)')
        # else:
        #     plt.title(f'{plot_aspect} over {x_var} for {control_variables["Model Name"]}')

    ''' 
    
    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel(ylabel)

    # Setting y-axis limits
    plt.ylim(0, 1)

    if "K-shot Size" in control_variables.keys():
        k_shot = control_variables["K-shot Size"]
        # Enhancing the legend
        plt.legend(title=f'Tasks ({k_shot}-Shot)')

    # Grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    with open("task_to_stats_rte_gsm8k_mnli_commonsense_all_models", "rb") as input_file:
        task_to_stats = pickle.load(input_file)

    df = build_dataframe(task_to_stats)
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Random", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Random", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})

    plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})

    plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-70B"})


    plot_2_vars(df, x_var = "K-shot Size", y_var = "Cannot Parse Proportion", control_variables = {"Batch Size" : 4  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "K-shot Size", y_var = "Cannot Parse Proportion", control_variables = {"Batch Size" : 4  , "Model Name" : "LLAMA-2-70B"})

    print("DONE")
    # plot_1  = extract_and_plot_stats(task_to_stats, control_vars = [], plot_var, plot_title='', x_label='', y_label='')

    # with open(r"task_to_stats_rte_incomplete_50", "rb") as input_file:
    #     stats_to_task = pickle.load(input_file)

    # rte_stats = stats_to_task["rte"]
    # # rte_batched_model_sizes = rte_stats.keys()
    # # rte_accuracies = [rte_stats[batch_size]["stat"]['Accuracy'] for batch_size in rte_batched_model_sizes]
    # # rte_proportion_cant_parse = [rte_stats[batch_size]["stat"]['proportion_cant_parse'] for batch_size in rte_batched_model_sizes]
    # # rte_sample_size = 50

    # rte_batched_model_sizes = [1,2,4,8]
    # rte_accuracies = [.8,.78,.82,.8]
    # rte_proportion_cant_parse = [0,0,0,0]#[rte_stats[batch_size]["stat"]['proportion_cant_parse'] for batch_size in rte_batched_model_sizes]
    # rte_sample_size = 50

    # with open(r"task_to_stats_rte_gsm8k", "rb") as input_file:
    #         stats_to_task_gsm8k  = pickle.load(input_file)
    # gsm8k_stats = stats_to_task_gsm8k["GSM8K"]
    # gsm8k_batched_model_sizes = gsm8k_stats.keys()
    # gsm8k_accuracies = [gsm8k_stats[batch_size]["stat"]['Accuracy'] for batch_size in rte_batched_model_sizes]
    # gsm8k_proportion_cant_parse = [gsm8k_stats[batch_size]["stat"]['proportion_cant_parse'] for batch_size in rte_batched_model_sizes]
    # gsm8k_sample_size = 50

    # plot_batch_results(batch_sizes = rte_batched_model_sizes, accuracies = rte_accuracies, parsing_error_rates = rte_proportion_cant_parse, title="RTE Zero-Shot BatchPrompt Results", sample_size = rte_sample_size)

    # plot_batch_results(batch_sizes = gsm8k_batched_model_sizes, accuracies = gsm8k_accuracies, parsing_error_rates = gsm8k_proportion_cant_parse, title="GSM8K Zero-Shot BatchPrompt Results", sample_size = gsm8k_sample_size)