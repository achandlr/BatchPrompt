import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import random

def save_plot(description):
    description = description.replace("(", "_").replace(")", "_").replace(":", "_").replace(" ", "_").replace(".","_").replace("\n","_")

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


def plot_2_vars(df, x_var="Batch Size", y_var="Accuracy - Wrong", control_variables={"K-shot Size": 1, "Model Name": "LLAMA-2-70B"}, y_range=(0, 1), show= False):
    plt.clf()
 
    # Update this list based on available models
    assert "Model Name" in control_variables.keys()
    # assert control_variables["Model Name"] in ['gpt-3.5-turbo-16k', 'LLAMA-2-70B']

    # Filtering the DataFrame
    filtered_df = df
    for key, value in control_variables.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    # Creating the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=filtered_df, x=x_var, y=y_var, hue='Task Name', marker='o')

    # Dynamic title based on y-variable (accuracy or error rate)
    plot_aspect = 'Accuracy' if "Accuracy" in y_var else 'Answer Parsing Error Rate'
    ylabel = plot_aspect

    if control_variables["Model Name"] == "gpt-3.5-turbo-16k":
        model_name = "GPT-3.5 Turbo (16k)"
    elif control_variables["Model Name"] == "LLAMA-2-70B":
        model_name = "LLAMA-2 (70B)"
    elif control_variables["Model Name"] == "LLAMA-2-13B":
        model_name = "LLAMA-2 (13B)"
    title = f'{model_name} {plot_aspect} over {x_var}'
    for key, value in control_variables.items():
        if key == "Model Name":
            continue
        title += f" with {key} = {value}"

    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel(ylabel)

    # Setting y-axis limits using the provided y_range
    plt.ylim(y_range)

    if "K-shot Size" in control_variables.keys():
        k_shot = control_variables["K-shot Size"]
        plt.legend(title=f'Tasks ({k_shot}-Shot)')

    plt.grid(True)
    if show:
        plt.show()
    save_plot(title)
    plt.close()

def get_task_name_to_size_accuracy(data, desired_accuracy = "none_is_wrong"):
    plot_data = {}

    # Iterate through each batched task combination
    for tasks, task_data in data.items():
        num_batched_tasks = len(tasks)  # Number of tasks batched together

        # Iterate through each task in the batch
        for task_name, metrics in task_data.items():

            
            if metrics and desired_accuracy in metrics:  # Check if accuracy data is available

                accuracy = metrics[desired_accuracy]['Accuracy']

                if task_name not in plot_data:
                    plot_data[task_name] = {}

                if num_batched_tasks not in plot_data[task_name]:

                    plot_data[task_name][num_batched_tasks] = [] 

                plot_data[task_name][num_batched_tasks].append(accuracy)
            else:
                pass

    task_name_to_size_accuracy = {}

    for task_name, task_data in plot_data.items():
        task_name_to_size_accuracy[task_name] = {}

        for task_size, accuracies in task_data.items():
            if accuracies:  # Ensure there are accuracies to average
                average_accuracy = sum(accuracies) / len(accuracies)
                task_name_to_size_accuracy[task_name][task_size] = average_accuracy
    return task_name_to_size_accuracy


def get_task_name_to_cannot_parse_proportion(data):
    plot_data = {}

    # Iterate through each batched task combination
    for tasks, task_data in data.items():
        num_batched_tasks = len(tasks)  # Number of tasks batched together

        # Iterate through each task in the batch
        for task_name, metrics in task_data.items():
            if 'cannot_parse_proportion' in metrics:  # Check if accuracy data is available
                cannot_parse_proportion = metrics['cannot_parse_proportion']

                if task_name not in plot_data:
                    plot_data[task_name] = {}

                if num_batched_tasks not in plot_data[task_name]:

                    plot_data[task_name][num_batched_tasks] = [] 

                plot_data[task_name][num_batched_tasks].append(cannot_parse_proportion)
            else:
                pass

    task_name_to_cannot_parse_proportion = {}

    for task_name, task_data in plot_data.items():
        task_name_to_cannot_parse_proportion[task_name] = {}

        for task_size, cannot_parse_data in task_data.items():
            if cannot_parse_data:  # Ensure there are accuracies to average
                average_accuracy = sum(cannot_parse_data) / len(cannot_parse_data)
                task_name_to_cannot_parse_proportion[task_name][task_size] = average_accuracy
    return task_name_to_cannot_parse_proportion

def plot_task_accuracies(data, title='Task Accuracies', xlabel='Number of Tasks', ylabel='Average Accuracy', legend_title='Tasks'):
    # Create a figure and an axis for the plot
    fig, ax = plt.subplots()

    max_tasks = max(max(accuracies.keys()) for accuracies in data.values())

    # Iterate over each task and plot their accuracies
    for task_name, accuracies in data.items():
        x_values = list(accuracies.keys())
        
        y_values = list(accuracies.values())
        ax.plot(x_values, y_values, label=task_name, marker='o')  # Plot with markers at each point

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a legend
    ax.legend(title=legend_title)
    ax.set_xticks(range(1, max_tasks + 1))
    plt.ylim((0,1))

    # Show the plot
    # plt.show()
    save_plot(title)
    plt.close()
    return

def plot_task_parse_error(data, title="Impact of Question Variety in Batched Prompts \non Answer Parsing Error Rates", xlabel='Number of Unique Tasks in Batched Prompt', ylabel='Answer Parsing Error Rate', legend_title='Tasks', y_range = (0,.2)):
    # Create a figure and an axis for the plot
    fig, ax = plt.subplots()

    max_tasks = max(max(accuracies.keys()) for accuracies in data.values())

    # Iterate over each task and plot their accuracies
    for task_name, accuracies in data.items():
        x_values = list(accuracies.keys())
        y_values = list(accuracies.values())
        ax.plot(x_values, y_values, label=task_name, marker='o')  # Plot with markers at each point

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a legend
    ax.legend(title=legend_title)
    ax.set_xticks(range(1, max_tasks + 1))
    plt.ylim(y_range)


    # Show the plot
    # plt.show()
    save_plot(title)
    plt.close()
    return
if __name__ == "__main__":




    with open("task_to_stats_rte_gsm8k_mnli_commonsense_all_models", "rb") as input_file:
        task_to_stats = pickle.load(input_file)

    df = build_dataframe(task_to_stats)
    df.drop('Accuracy - Skipped', axis=1, inplace=True)
    df.drop('Accuracy - Random', axis=1, inplace=True)
    df.to_csv("Stage 1 Results all three models.csv")

    # Stage 1 Plotting
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-13B"}, show = False)
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-13B"}, show = False)
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-13B"}, show = False)
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-13B"}, show = False)
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 6  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 6  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 6  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"}, y_range=(0, 0.3))
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 6  , "Model Name" : "gpt-3.5-turbo-16k"}, y_range=(0, 0.3))
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Cannot Parse Proportion", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "K-shot Size", y_var = "Cannot Parse Proportion", control_variables = {"Batch Size" : 4  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "K-shot Size", y_var = "Cannot Parse Proportion", control_variables = {"Batch Size" : 4  , "Model Name" : "LLAMA-2-70B"})


    # These plots treat answers that cannot be parsed differently
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Random", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Random", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Random", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Wrong", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 1  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 0  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 6  , "Model Name" : "gpt-3.5-turbo-16k"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 0  , "Model Name" : "LLAMA-2-70B"})
    # plot_2_vars(df, x_var = "Batch Size", y_var = "Accuracy - Skipped", control_variables = {"K-shot Size" : 1  , "Model Name" : "LLAMA-2-70B"})



    # # Stage 2: Plotting Line Plot for the data
    from datasets import load_dataset, Dataset
    from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
    from enum import Enum, auto
    from tqdm import tqdm
    from pathlib import Path
    class DatasetType(Enum):
        GSM8K_HARD = auto()
        GSM8K_HARD_CoT = auto()
        COMMON_SENSE = "COMMON_SENSE"
        COMMON_SENSE_CoT = auto()
        GSM8K = "GSM8K"
        MBPP = "MBPP"
        RTE = "RTE"
        MNLI = "MNLI"
    model_to_dataset_path = {"GPT-3.5 Turbo (16K)" : 'dataset_combination_to_stats', "LLAMA-2 (70B)" : 'dataset_combination_to_stats_llama', "LLAMA-2 (13B)" : 'dataset_combination_to_stats_llama_13B'}
    model_name_to_data = {}

    for model_name, pickle_file_path in model_to_dataset_path.items():

        # Load the data from the pickle file
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        task_name_to_size_accuracy = get_task_name_to_size_accuracy(data, desired_accuracy = "none_is_wrong")
        plot_task_accuracies(task_name_to_size_accuracy, title=f'{model_name} Accuracy\nby Number of Unique Task Types in Batched Prompts', xlabel='Number of Unique Task Types in Batched Prompts', ylabel='Average Accuracy Across Experiments', legend_title='Tasks')

        task_name_to_cannot_parse_proportion = get_task_name_to_cannot_parse_proportion(data)

        plot_task_parse_error(task_name_to_cannot_parse_proportion, title=f"{model_name} Impact of Question Variety in Batched Prompts \non Answer Parsing Error Rates", legend_title='Tasks')

        if model_name not in model_name_to_data:
             model_name_to_data[model_name] = {"task_name_to_size_accuracy" : {}, "task_name_to_cannot_parse_proportion" : {}}
        model_name_to_data[model_name]["task_name_to_size_accuracy"] = task_name_to_size_accuracy
        model_name_to_data[model_name]["task_name_to_cannot_parse_proportion"] = task_name_to_cannot_parse_proportion
        continue

    pickle.dump((model_name_to_data), open("stage_2_model_name_to_multi_task_stats", 'wb')) 
