from src.utils.dataset_loader import extract_function_name
from data.generated.mbpp_CoT import step_by_step_thinking
import re
from typing import List


def gsm8k_question_format_baseline(example, i):
    example_question_format = f"{example['question']}"
    return example_question_format

def gsm8k_answer_format_baseline(example, i):
    extract_numbers = lambda x: x.split("####")[-1].strip()
    extracted_values = extract_numbers(example['answer'])
    return f"{extracted_values}"

def gsm8k_question_format(example, i):
    example_question_format = f"Q[{i}]: {example['question']}"
    return example_question_format

def gsm8k_answer_format(example, i):
    def transform_to_prompt(example):
        intermediate_reasoning = example.split("####")[0]
        intermediate_reasoning = intermediate_reasoning.rstrip("\n")
        extract_numbers = lambda x: x.split("####")[-1].strip()
        extracted_value = extract_numbers(example)
        # Construct the new prompt
        new_prompt = f"Intermediate Reasoning: {intermediate_reasoning}. The answer is {extracted_value}"
        return new_prompt
    prompt = transform_to_prompt(example["answer"])
    answer = f"A[{i}]: {prompt}"
    return  answer

def gsm8k_example_question_format(example, i):
    return gsm8k_question_format(example, i)
# def gsm8k_answer_format(example, i):
#     extract_numbers = lambda x: x.split("####")[-1].strip()
#     extracted_values = extract_numbers(example['answer'])
#     return f"A[{i}]: {extracted_values}"


# TODO: Need verify
def gsm8k_CoT_question_format(example, i):
    return  gsm8k_question_format(example, i)

# # TODO: Need verify
# def gsm8k_example_question_format(example, i):
#     def transform_to_prompt(example):
#         extract_numbers = lambda x: x.split("####")[-1].strip()
#         extracted_value = extract_numbers(example)
#         # Construct the new prompt
#         new_prompt = f"Intermediate Reasoning: {example}. The answer is {extracted_value}"
#         return new_prompt
#     prompt = transform_to_prompt(example["answer"])
#     answer = f"A[{i}]: \n{prompt}"
#     return  answer

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

def mnli_example_question_format(example, i):
    def extract_premise_hypothesis(text):
        # Regular expressions to match the premise and hypothesis
        premise_pattern = re.compile(r'Premise:\s*(.*?)\s*\n', re.DOTALL)
        hypothesis_pattern = re.compile(r'Hypothesis:\s*(.*?)\s*\n', re.DOTALL)

        # Search for premise and hypothesis using the patterns
        premise_match = premise_pattern.search(text)
        hypothesis_match = hypothesis_pattern.search(text)

        # Extracting premise and hypothesis from the matches
        premise = premise_match.group(1) if premise_match else None
        hypothesis = hypothesis_match.group(1) if hypothesis_match else None

        return premise, hypothesis
    premise, hypothesis = extract_premise_hypothesis(example['source'])
    return f"Premise[{i}]: {premise}\nHypothesis[{i}]: {hypothesis}"
    
def mnli_question_format(example, i):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    return f"Premise[{i}]: {premise}\nHypothesis[{i}]: {hypothesis}"

# def mnli_answer_format(example, i):
#     label = example["label"]
#     label_str = "entailment" if label == 0 else "neutral" if label == 1 else "contradiction" 
#     return f"A[{i}]: {label_str}"

def mnli_answer_format(example, i):
    rationale = example['rationale']
    label = example['target']
    label_as_int = 0 if label == 'yes' else 2 if label == 'contradiction' else 1 
    return f"A[{i}]: Intermediate Reasoning: {rationale}. The answer is {label_as_int}"

def mnli_CoT_question_format(example, i):
    raise NotImplementedError()

def mnli_CoT_answer_format(example, i):
    raise NotImplementedError()

#TODO: Add mnli CoT
def rte_question_format(example, i):
    return f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"

def rte_example_question_format(example, i):
    def extract_context_hypothesis(text, want_options_in_hypothesis = False):        
        # First regular expression pattern, using re.DOTALL to make '.' match newlines
        context_pattern_1 = r'Context:\s*([\s\S]+?)(?:\n{2,}|$)'
        if want_options_in_hypothesis:
            hypothesis_pattern_1 = r'Context:\s*([\s\S]+?)\n\nHypothesis:'
        else:
            hypothesis_pattern_1 = r'Hypothesis:\s*([\s\S]+?)(?:\n{2,}|\nOPTIONS|A:|$)'

        hypothesis_pattern_1 = r'Hypothesis:\s*([\s\S]+?)(?:\n{2,}|\nOPTIONS|A:|$)'
        
        # Backup regular expression pattern
        context_pattern_2 = r'context[:\s]+(.+?)(?:\s+hypothesis[:\s]+|$)'
        hypothesis_pattern_2 = r'hypothesis[:\s]+(.+?)(?:\s+context[:\s]+|$)'
  
        context_match = re.search(context_pattern_1, text)
        hypothesis_match = re.search(hypothesis_pattern_1, text)
        
        # If first regex fails, use backup regex
        if context_match is None:
            context_match = re.search(context_pattern_2, text)
        if hypothesis_match is None:
            hypothesis_match = re.search(hypothesis_pattern_2, text)
        
        # Extract context and hypothesis
        context = context_match.group(1) if context_match else None
        hypothesis = hypothesis_match.group(1) if hypothesis_match else None
        
        return context, hypothesis
    context, hypothesis = extract_context_hypothesis(example['source'])
    return f"Premise[{i}]: {context}\nHypothesis[{i}]: {hypothesis}"


def commonsense_example_question_format(example, i):
    question = example['source']
    assert "ve one answer for each question.\n\n" in question
    question_without_instruction = question.split("ve one answer for each question.\n\n")[1]
    return f"Q[{i}]: {question_without_instruction}"

def rte_answer_format(example, i):
    target = example["target"]
    label = 0 if target == "yes" else 1 if target == "no" else 2
    intermediate_logic = example["rationale"]
    assert label != 2
    return f"A[{i}]: Intermediate Reasoning: {intermediate_logic}. The answer is {label}"
# TODO: add rte CoT

#TODO: Add mnli CoT
def rte_CoT_question_format(example, i):
    raise NotImplementedError()
    return f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"

def rte_CoT_answer_format(example, i):
    raise NotImplementedError()
    # return f"Answer[{i}]: {example['label']}

def commonsense_question_format(example, i):
    question = example["question"]

    question_str = f"Question[{i}]: {question}\nAnswer Choices: \n"
    choice_labels = example['choices']['label']
    choice_texts = example['choices']['text']
    
    for label, text in zip(choice_labels, choice_texts):
        question_str += f"{label}: {text} "
    
    return question_str



# def commonsense_question_format(example, i):
#     question_with_answer_choices = example["source"]
#     rationale = example["choices"]
#     def build_question_string(question, choices, i):
#         question_str = f"Question[{i}]: {question}\nAnswer Choices: \n"
#         choice_labels = choices['label']
#         choice_texts = choices['text']
        
#         for label, text in zip(choice_labels, choice_texts):
#             question_str += f"{label}: {text}\n"
        
#         return question_str
#     question_str = build_question_string(question_with_answer_choices, rationale)
#     return question_str


def commonsense_answer_format(example, i):
    answer = example["target"]
    rationale = example['rationale']
    return f"A[{i}]: Intermediate Reasoning: {rationale}. The answer is {answer}"

# def commonsense_answer_format(example, i):
#     answer = example["answerKey"]
#     return f"Answer[{i}]: {answer}"

def commonsense_CoT_question_format(example, i):
    raise None

def commonsense_CoT_answer_format(example, i):
    raise None

