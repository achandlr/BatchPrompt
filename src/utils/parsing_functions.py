from src.utils.dataset_loader import extract_function_name
from data.generated.mbpp_CoT import step_by_step_thinking
import re
from typing import List


def gsm8k_question_format(example, i):
    example_question_format = f"Q[{i}]: {example['question']}"
    return example_question_format

def gsm8k_answer_format(example, i):
    extract_numbers = lambda x: x.split("####")[-1].strip()
    extracted_values = extract_numbers(example['answer'])
    return f"A[{i}]: {extracted_values}"

# TODO: Need verify
def gsm8k_CoT_question_format(example, i):
    return  gsm8k_question_format(example, i)

# TODO: Need verify
def gsm8k_CoT_answer_format(example, i):
    def transform_to_prompt(example):
        extract_numbers = lambda x: x.split("####")[-1].strip()
        extracted_value = extract_numbers(example)
        # Construct the new prompt
        new_prompt = f"Step by Step thinking:\n{example}\nFinal Output:\n{extracted_value}"
        return new_prompt
    prompt = transform_to_prompt(example["answer"])
    answer = f"A[{i}]: \n{prompt}"
    return  answer

# TODO: Need verify
def gsm8k_hard_question_format(example, i):
    question = example['input']
    return f"Q[{i}]: {question}"
    # def transform_to_prompts(examples_with_CoT):
    #     transformed_prompts = []
    #     extract_numbers = lambda x: x.split("####")[-1].strip()

    #     for example in examples_with_CoT:
    #         extracted_value = extract_numbers(example)

    #         # Construct the new prompt
    #         new_prompt = f"Step by Step thinking:\n{example}\n\nFinal Output:\n{extracted_value}"

    #         transformed_prompts.append(new_prompt)
    #     return transformed_prompts

# TODO: Need verify
def gsm8k_hard_answer_format(example, i):
    answer = str(int(example['target']) + 1)
    return f"A[{i}]: {answer}"

def gsm8k_CoT_hard_question_format(example, i):
    return gsm8k_CoT_question_format(example, i)

def gsm8k_CoT_hard_answer_format(example, i):
    return gsm8k_CoT_answer_format(example, i)
    # def transform_to_prompt(example):
    #     extract_numbers = lambda x: x.split("####")[-1].strip()
    #     extracted_value = extract_numbers(example)
    #     # Construct the new prompt
    #     new_prompt = f"Step by Step thinking:\n{example}\nFinal Output:\n{extracted_value}"
    #     return new_prompt
    # prompt = transform_to_prompt(example["target"])
    # answer = f"A[{i}]: \n{prompt}"



def mbpp_question_format(example, i):
    function_name = extract_function_name(example['test_list'][0])
    example_question_format = f"Q[{i}]: {example['question']}. The function name is {function_name}"
    return example_question_format

def mbpp_answer_format(example, i):
    return f"A[{i}]: {example['code']}"

def mbpp_CoT_question_format(example, i):
    raise NotImplementedError()

def mbpp_CoT_answer_format(example, i):
    output_with_CoT = []
    # Note: We don't have CoT thinking for all examples, just the first 15 from the train set.
    raise NotImplementedError


def mnli_question_format(example, i):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    return f"Q[{i}]: Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"

def mnli_answer_format(example, i):
    label = example["label"]
    label_str = "entailment" if label == 0 else "neutral" if label == 1 else "contradiction" 
    return f"A[{i}]: {label_str}"

def mnli_CoT_question_format(example, i):
    raise NotImplementedError()

def mnli_CoT_answer_format(example, i):
    raise NotImplementedError()

#TODO: Add mnli CoT
def rte_question_format(example, i):
    return f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"

def rte_answer_format(example, i):
    return f"A[{i}]: {example['label']}"
# TODO: add rte CoT

#TODO: Add mnli CoT
def rte_CoT_question_format(example, i):
    raise NotImplementedError()
    return f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"

def rte_CoT_answer_format(example, i):
    raise NotImplementedError()
    # return f"Answer[{i}]: {example['label']}

def commonsense_question_format(example, i):
    question_with_answer_choices = example["source"]
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

def commonsense_CoT_question_format(example, i):
    raise None

def commonsense_CoT_answer_format(example, i):
    raise None

