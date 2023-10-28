from src.experiments.k_shot_experiment_configs import config_param_list, oai_gen_params
from src.experiments.k_shot_experiment import * # BatchPromptExperiment,BatchPromptingExperimentConfig
from src.utils.evaluation import CodeEvaluator, Evaluation
import re
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict

# from src.utils.parsing_functions import extract_answers_batch

# TODO: For now just use OPEN_AI, but soon we will loop through API calls.
 # TODO: Move this back to parsing functions, but just keep it here for now while debug
def extract_answers_batch(output_str: str, answer_type = None) -> List[int]:
    answer_type = answer_type.upper()
    if answer_type == "COMMON_SENSE":
                # Initialize an empty list to store the extracted answers.
        answers = []
        
        # Step 1: Split the string by newlines to process each line individually.
        lines = output_str.strip().split("\n")
        
        # Modified regex pattern to extract either an integer or single uppercase letter
        # after ": ".
        primary_pattern = r": ([A-D]|\d+)"
        
        # Backup regex pattern to extract any number in the line.
        backup_pattern = r"([A-D]|\d+)"
        
        if answer_type == "COMMON_SENSE":
            raise None  # Raise appropriate Exception or modify as needed

        # Step 2: Loop through each line to extract the answer.
        for line in lines:
            # Try primary regex pattern first.
            match = re.search(primary_pattern, line)
            if match:
                answers.append(match.group(1))
            else:
                # If primary fails, try the backup pattern.
                match = re.search(backup_pattern, line)
                if match:
                    answers.append(match.group(1))

        return answers
    elif answer_type ==  "MBPP":
        raise None
    # elif answer_type == "MNLI":
    #     raise None
    elif answer_type in ["GSM8K", "GSM8K_HARD", "RTE", "MNLI"]:
        answers = []
        
        # Split the string by newlines to process each line individually.
        lines = output_str.strip().split("\n")
        
        # General regex pattern to extract a potential answer.
        # general_pattern = r": (\d+)|(\d+)"
        
        # Loop through each line to extract the answer.
        for line in lines:
            answer = extract_last_number(line)
            answers.append(answer)
            # found = False
            # for match in re.finditer(general_pattern, line):
            #     start, end = match.span()
                
            #     # Check the preceding characters to make sure the found digit is not within brackets.
            #     preceding_text = line[max(0, start - 5):start]
            #     if not re.search(r"\[\d+\]", preceding_text):
            #         answers.append(int(match.group().split(' ')[-1]))
            #         found = True
            #         break

            # # If no answer was found
            # if not found:
            #     answers.append(None)

        return answers

    else: 
        raise NotImplementedError()

def convert_to_int(input_str: str) -> int:
    # Step 1: Remove unnecessary characters like spaces and commas
    cleaned_str = re.sub(r"[^\d.-]", "", input_str)
    
    # Step 2: Convert the cleaned string to float
    float_val = float(cleaned_str)
    
    # Step 3: Convert the float to integer
    int_val = int(float_val)
    
    return int_val
    
def get_index_to_ground_truth(answers_dict, task_name):
    task_name = task_name.upper()
    index_to_ground_truth = {}
    for answer_index, answer in answers_dict.items():
        if task_name == "MBPP":
            raise NotImplementedError()
        elif task_name == "GSM8K":
            index_to_ground_truth[answer_index] = convert_to_int(answer["answer"].split("####")[-1])
        elif task_name == "GSM8K_HARD":
            raise NotImplementedError()
        elif task_name == "COMMON_SENSE":
            raise NotImplementedError()
        elif task_name == "MNLI":
            raise NotImplementedError()
        elif task_name == "RTE":
            index_to_ground_truth[answer_index] = int(answer["label"])
        else:
            raise ValueError("Task name not recognized.")
    return index_to_ground_truth

def get_index_to_pred(batched_model_inputs, batched_model_outputs, task_name):
    index_to_pred = {}
    for batch_input, batch in zip(batched_model_inputs, batched_model_outputs):
        indices, LLM_output = batch
        answers = extract_answers_batch(LLM_output, task_name)
        # answers = parse_batched_answers(LLM_output, task_name)
        if len(answers) == len(batch[0]):
            for index, answer in zip(indices, answers):
                index_to_pred[index] = answer
        elif len(answers) > len(batch[0]):
            for index, answer in zip(indices, answers[0:len(batch[0])]):
                index_to_pred[index] = answer 
        else:
            for index, answer in zip(indices, answers[0:len(indices)]):
                index_to_pred[index] = answer
            
            for index in indices[len(answers):]:
                index_to_pred[index] = None
                # TODO: Maybe throw in new lines
 
    return index_to_pred
from typing import Optional

def extract_last_number(text: str) -> Optional[int]:
    # Define the regex pattern to capture numbers with optional negative sign, dollar sign, and commas
    pattern = r"[-$]?[\d,]+"

    # Find all matching numbers in the string
    matches = re.findall(pattern, text)

    # If no match is found, return None
    if not matches:
        return None
    try:
        # Grab the last match
        last_match = matches[-1]

        # Remove dollar sign and commas, if any
        cleaned_match = last_match.replace("$", "").replace(",", "")

        # Convert to int
        final_number = int(cleaned_match)
        return final_number
    except Exception as e:
        return None


def get_ordered_lists(index_to_pred: dict, index_to_ground_truth: dict) -> (list, list):
    # Initialize empty lists for predictions and ground truth values.
    pred = []
    ground_truth = []
    
    # Ensure both dictionaries have the same keys, otherwise raise an exception.
    if set(index_to_pred.keys()) > set(index_to_ground_truth.keys()):
        raise ValueError("The keys in both dictionaries should match.")
    
    # Sort the keys to ensure the values are ordered.
    sorted_keys = sorted(index_to_pred.keys())
    
    # Populate the 'pred' list with prediction values in sorted order of keys.
    for key in sorted_keys:
        pred.append(index_to_pred[key])
        
    # Populate the 'ground_truth' list with ground truth values in sorted order of keys.
    for key in sorted_keys:
        ground_truth.append(index_to_ground_truth[key])
        
    return pred, ground_truth

def get_pred_ground_truth(batched_model_inputs, batched_model_outputs, answers_dict, task_name):
    index_to_pred = get_index_to_pred(batched_model_inputs, batched_model_outputs, task_name)
    index_to_ground_truth = get_index_to_ground_truth(answers_dict, task_name)
    pred, ground_truth = get_ordered_lists(index_to_pred, index_to_ground_truth)
    return pred, ground_truth


config_to_answer_type = {"GSM8K": "numerical", 
                "GSM8K_HARD": "numerical", 
                "COMMON_SENSE": "categorical", 
                "MBPP": "code",
                "MNLI": "binary",
                "RTE": "categorical"}

task_to_stats ={}
for task_name, configs in config_param_list.items():
    if task_name != "GSM8K":
        continue
    oai_gen_params = OpenAIGenerationParameters(
            model_name='gpt-3.5-turbo',
            temperature=0.2,
            max_tokens=512,
            frequency_penalty=1.0,
        )
    questions_config, examples_config, task_description, question_format, answer_format = configs

    config = BatchPromptingExperimentConfig(
    questions_dataset_config=questions_config,
    examples_dataset_config=examples_config,
    task_description=task_description,
    k_shot=4,
    example_selection=ExampleSelectionType.SEMANTIC,
    example_question_format=question_format,
    example_answer_format=answer_format,
    batch_size=1,
    model_api=ModelAPIType.OPEN_AI,
    generation_params=oai_gen_params,
    random_seed=0,
    debug=BatchPromptingDebugConfig(
            truncate_examples=True,
            truncate_batch_queries=True,
            save_batched_model_inputs=None,
            save_batched_model_outputs=None,
        )
)
    experiment = BatchPromptExperiment(config)
    batched_model_inputs, batched_model_outputs, answers_dict = experiment.execute()
    if task_name == "MBPP":
        evaluator = CodeEvaluator()
        raise NotImplementedError()
        # mbpp_code_example = mbpp['train']['code'][index]
        # mbpp_test_cases_example = mbpp['train']['test_list'][index]
        # result = evaluator.run_code_and_tests(mbpp_code_example, mbpp_test_cases_example)
    else:
        pred, ground_truth = get_pred_ground_truth(batched_model_inputs, batched_model_outputs, answers_dict, task_name)
        evaluator = Evaluation()

        stat = evaluator.get_stats(y_pred=pred, y_true=ground_truth, answer_type = config_to_answer_type[task_name.upper()])
    task_to_stats[task_name] = stat

print(task_to_stats)
with open("task_to_stats", 'wb') as f:
    pickle.dump(task_to_stats, f)