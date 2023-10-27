from src.utils.dataset_loader import extract_function_name
from data.generated.mbpp_CoT import step_by_step_thinking

def mbpp_question_format(example, i):
    function_name = extract_function_name(example['test_list'][0])
    example_question_format = f"Q[{i}]: {example['question']}. The function name is {function_name}"
    return example_question_format

def mbpp_answer_format(example, i):
    return f"Answer[{i}]: {example['code']}"

def mbpp_CoT_answer_format(example, i):
    output_with_CoT = []
    # Note: We don't have CoT thinking for all examples, just the first 15 from the train set.
    raise NotImplementedError
    # for question, test_list, step_by_step, answer in zip(questions, test_list, step_by_step_thinking_without_dissalowed_indices, answers):
        # function_name = extract_function_name(test_list[0])
        # questions_with_function_names.append(f"Python Function Request: {question} The functions name should be {function_name}.")
    output_with_CoT += [f"Step by Step thinking:\n{step_by_step}\n\nFinal Output:\n{answer}"]
# TODO: add mbpp CoT

def gsm8k_question_format(example, i):
    example_question_format = f"Q[{i}]: {example['question']}"
    return example_question_format

def gsm8k_answer_format(example, i):
    extract_numbers = lambda x: x.split("####")[-1].strip()
    extracted_values = extract_numbers(example['answer'])
    return f"A[{i}]: {extracted_values}"

def gsm8k_CoT_question_format(example, i):
    return  gsm8k_question_format(example, i)

def gsm8k_CoT_answer_format(example, i):
    def transform_to_prompt(example):
        extract_numbers = lambda x: x.split("####")[-1].strip()
        extracted_value = extract_numbers(example)
        # Construct the new prompt
        new_prompt = f"Step by Step thinking:\n{example}\nFinal Output:\n{extracted_value}"
        return new_prompt
    prompt = transform_to_prompt(example)
    answer = f"A[{i}]: {prompt}"
    return  answer

def mnli_question_format(example, i):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    return f"Q[{i}]: Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"

def mnli_answer_format(example, i):
    label = example["label"]
    label_str = "entailment" if label == 0 else "neutral" if label == 1 else "contradiction" 
    return f"A[{i}]: {label_str}"


#TODO: Add mnli CoT
def rte_question_format(example, i):
    return f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"

def rte_answer_format(example, i):
    return f"Answer[{i}]: {example['label']}"
# TODO: add rte CoT


def commonsense_question_format(example, i):
    question_with_answer_choices = example["example"]
    rationale = example["choices"]
    def build_question_string(question, choices, i):
        question_str = f"Question[{i}]: {question}\nAnswer Choices: \n"
        choice_labels = choices['label']
        choice_texts = choices['text']
        
        for label, text in zip(choice_labels, choice_texts):
            question_str += f"{label}: {text}\n"
        
        return question_str
    question_str = build_question_string(question_with_answer_choices, rationale)
    return question_str

def commonsense_answer_format(example, i):
    answer = example["answerKey"]
    return f"Answer[{i}]: {answer}"

# TODO: add commonsense CoT

def gsm8k_hard_question_format(example, i):
    question = example["input"]
    return f"Q[{i}]: {question}"

def gsm8k_hard_answer_format(example, i):
    return f"Answer[{i}]: {str(int(example['target']))}"

# TODO: add gsm8k hard CoT