from src.utils.dataset_loader import extract_function_name

def mbpp_question_format(example, i):
    function_name = extract_function_name(example['test_list'][0])
    example_question_format = f"Q[{i}]: {example['question']}. The function name is {function_name}"
    return example_question_format

def mbpp_answer_format(example, i):
    return f"Answer[{i}]: {example['code']}"

def gsm8k_question_format(example, i):
    example_question_format = f"Q[{i}]: {example['question']}"
    return example_question_format

def gsm8k_answer_format(example, i):
    extract_numbers = lambda x: x.split("####")[-1].strip()
    extracted_values = extract_numbers(example['answer'])
    return f"A[{i}]: {extracted_values}"