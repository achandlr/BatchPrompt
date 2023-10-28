from src.experiments.k_shot_experiment_configs import config_param_list, oai_gen_params
from src.experiments.k_shot_experiment import * # BatchPromptExperiment,BatchPromptingExperimentConfig
from src.utils.evaluation import CodeEvaluator, Evaluation
import re
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
        # Initialize an empty list to store the extracted answers.
        answers = []
        
        # Step 1: Split the string by newlines to process each line individually.
        lines = output_str.strip().split("\n")
        
        # Primary regex pattern to extract number after ": ".
        primary_pattern = r": (\d+)"
        
        # Backup regex pattern to extract any number in the line.
        backup_pattern = r"(\d+)"
        
        # Step 2: Loop through each line to extract the answer.
        for line in lines:
            # Try primary regex pattern first.
            match = re.search(primary_pattern, line)
            if match:
                answers.append(int(match.group(1)))
            else:
                # If primary fails, try the backup pattern.
                match = re.search(backup_pattern, line)
                if match:
                    answers.append(int(match.group(1)))

        return answers
    else: 
        raise NotImplementedError()
    
def get_index_to_ground_truth(answers_dict, task_name):
    task_name = task_name.upper()
    index_to_ground_truth = {}
    for answer_index, answer in answers_dict.items():
        if task_name == "MBPP":
            raise NotImplementedError()
        elif task_name == "GSM8K":
            raise NotImplementedError()
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

def get_index_to_pred(batched_model_outputs, task_name):
    index_to_pred = {}
    for batch in batched_model_outputs:
        indices, LLM_output = batch
        answers = extract_answers_batch(LLM_output, task_name)
        # answers = parse_batched_answers(LLM_output, task_name)
        if len(answers) == len(batch[0]):
            for index, answer in zip(indices, answers):
                index_to_pred[index] = answer
        else:
            raise ValueError("Either the parsing fails or the LLM prompt fails to return desired output.")
    return index_to_pred

def get_ordered_lists(index_to_pred: dict, index_to_ground_truth: dict) -> (list, list):
    # Initialize empty lists for predictions and ground truth values.
    pred = []
    ground_truth = []
    
    # Ensure both dictionaries have the same keys, otherwise raise an exception.
    if set(index_to_pred.keys()) != set(index_to_ground_truth.keys()):
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

def get_pred_ground_truth(batched_model_outputs, answers_dict, task_name):
    index_to_pred = get_index_to_pred(batched_model_outputs, task_name)
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
    questions_config, examples_config, task_description, question_format, answer_format = configs
    config = BatchPromptingExperimentConfig(
    questions_dataset_config=questions_config,
    examples_dataset_config=examples_config,
    task_description=task_description,
    k_shot=7,
    example_selection=ExampleSelectionType.RANDOM,
    example_question_format=question_format,
    example_answer_format=answer_format,
    batch_size=4,
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
    batched_model_outputs, answers_dict = experiment.execute()
    if task_name == "MBPP":
        evaluator = CodeEvaluator()
        raise NotImplementedError()
        # mbpp_code_example = mbpp['train']['code'][index]
        # mbpp_test_cases_example = mbpp['train']['test_list'][index]
        # result = evaluator.run_code_and_tests(mbpp_code_example, mbpp_test_cases_example)
    else:
        pred, ground_truth = get_pred_ground_truth(batched_model_outputs, answers_dict, task_name)
        stat = evaluator.get_stats(y_pred=pred, y_true=ground_truth, answer_type = config_to_answer_type[task_name])
        evaluator = Evaluation()
