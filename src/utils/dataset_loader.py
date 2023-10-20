
from datasets import load_dataset
import pickle

def pickle_dataset(dataset, dataset_name):
    """
    Pickles the dataset
    """
    with open("data\\imported\\datasets\\pickled\\"+dataset_name, 'wb') as f:
        pickle.dump(dataset, f)
    return 

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_allowed_indices(disallowed_data_indices, total_length):
    """
    Given a list of disallowed indices, return a list of indices that are allowed.

    Parameters:
    - disallowed_data_indices (list): List of disallowed indices.
    - total_length (int): The total length of the list for which we are finding allowed indices.

    Returns:
    - list: List of allowed indices.
    """
    if disallowed_data_indices == None:
        return list(range(total_length))
    else:
        return [i for i in range(total_length) if i not in disallowed_data_indices]

def get_few_shot_examples(dataset, dataset_name, example_type, disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None):
    
    if dataset_name == "gsm8k":
        allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]['answer']))
        questions = [dataset[data_location]['input'][i] for i in allowed_indices]
        examples_with_CoT = [dataset[data_location]['answer'][i] for i in allowed_indices]
        if example_type == "CoT":
            return questions, examples_with_CoT
        else:
            extract_numbers = lambda x: x.split("####")[-1].strip()
            extracted_values = [extract_numbers(example) for example in examples_with_CoT]
            return questions, extracted_values
    elif dataset_name =="gsm-hard":
        if example_type == "CoT":
            # TODO: For now using gsm8k CoT examples because gsm-hard rather gives us code to solve but not step by step answers for code (ex: eggs_sold = eggs_per_day - eggs_eaten - eggs_baked instead of eggs_sold = 100 - 10 - 5)
            assert needed_extra_dataset != None
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(needed_extra_dataset[data_location]))
            questions = [needed_extra_dataset[data_location]['input'][i] for i in allowed_indices]
            examples_with_CoT = [needed_extra_dataset[data_location]['answer'][i] for i in allowed_indices]
            return questions, examples_with_CoT
        else:
            assert disallowed_data_indices == None
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]['target'])) 
            questions =  [dataset[data_location]['input'][i] for i in allowed_indices]
            answers =  [dataset[data_location]['target'][i] for i in allowed_indices]
            return questions, answers
    elif dataset_name == "commonsense_qa":
        allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]['answerKey']))
        if example_type == "CoT":
            assert needed_extra_dataset != None
            def filter_commensenseqa(example):
                return example['task'] == 'commonsenseqa'
            commonsense_qa_CoT = needed_extra_dataset['train'].filter(filter_commensenseqa)
            questions_with_answer_choices = commonsense_qa_CoT["source"]
            rationales = commonsense_qa_CoT["rationale"]
            answers = commonsense_qa_CoT["target"]
            def combine_rationale_and_answer(rationale, answer):
                # Remove any newline characters from the end of the rationale
                rationale_cleaned = rationale.rstrip("\n\\n")
                
                # Prepend "Step by step thinking" and append "Final Answer"
                combined_str = f"Step by step thinking:\n{rationale_cleaned}\n\nFinal Answer:\n{answer}"
                
                return combined_str
            answers_with_rationales = [combine_rationale_and_answer(rationale, answer) for rationale, answer in zip(rationales, answers)]
            return questions_with_answer_choices, answers_with_rationales
        else:
            question = [dataset[data_location]['question'][i] for i in allowed_indices]
            choices = [dataset[data_location]['choices'][i] for i in allowed_indices]
            answers = [dataset[data_location]['answerKey'][i] for i in allowed_indices]
            def build_question_string(question, choices):
                question_str = f"Question:\n{question}\nAnswer Choices:\n"
                choice_labels = choices['label']
                choice_texts = choices['text']
                
                for label, text in zip(choice_labels, choice_texts):
                    question_str += f"{label}: {text}\n"
                
                return question_str
            question_with_answer_choices = [build_question_string(question, choices) for question, choices in zip(question, choices)]
            return question_with_answer_choices, answers
    elif dataset_name == "mbpp":
        if example_type == "CoT":
            # TODO: Create our own CoT examples for MBPP
            raise NotImplementedError()
        else:
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]))
            example_questions = [dataset[data_location][i]['text'] for i in allowed_indices]
            example_answers = [dataset[data_location][i]['code'] for i in allowed_indices]
            return example_questions, example_answers
    elif dataset_name == "rte":
        if example_type == "CoT":
            # use CoT-Collection 
            assert needed_extra_dataset != None
            raise None
        else:
            raise None
    elif dataset_name == "mnli":
        if example_type == "CoT":
            # use CoT-Collection 
            assert needed_extra_dataset != None
            mnli_CoT =  needed_extra_dataset['train'].filter(lambda example: example['task'] == 'mnli')
            questions = mnli_CoT['source']
            rationales = mnli_CoT['rationale']
            answers = mnli_CoT['target']
            def build_CoT_answer(rationale, answer):
                # Remove any newline characters from the end of the rationale
                rationale_cleaned = rationale.rstrip("\n")
                # Construct the CoT example
                CoT_example = f"Step by Step Thinking:\n{rationale_cleaned}\n\nFinal Answer:\n{answer}"
                return CoT_example
            CoT_answers = [build_CoT_answer(rationale, answer) for rationale, answer in zip(rationales, answers)]
            return questions, CoT_answers
        else:
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]))
            premises = [dataset[data_location]['premise'][i] for i in allowed_indices]
            hypothesises = [dataset[data_location]['hypothesis'][i] for i in allowed_indices]
            labels =  [dataset[data_location]['label'][i] for i in allowed_indices]
            def build_mnli_question(premise, hypothesis):
                return f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"
            premise_with_hypothesis_list = [build_mnli_question(premise, hypothesis) for premise, hypothesis in zip(premises, hypothesises)]
            return premise_with_hypothesis_list, labels
    else:
        raise ValueError("Invalid dataset name")
# Load the gsm8k dataset
# TODO: Confirm all below
gsm8k = load_dataset("gsm8k", "main")  # well organized, easy to use
gsm_hard = load_dataset("reasoning-machines/gsm-hard") # well organized, easy to use
commensense_qa = load_dataset("commonsense_qa") # train and validation hold gold truth labels, test does not
human_eval = load_dataset("openai_humaneval") # test cases in human_eval["test"]["test"], but hard to parse and would be fair bit of work
mbpp = load_dataset("mbpp") # test cases well organized as list for each code generation in mbpp["train"]['test_list']
rte = load_dataset("glue", "rte") # well organized, easy to use, ignore test because no ground truth
mnli = load_dataset("glue", "mnli")
CoT_examples = load_dataset("kaist-ai/CoT-Collection")

# Pickle the datasets
pickle_dataset(gsm8k, "gsm8k")
pickle_dataset(gsm_hard, "gsm-hard")
pickle_dataset(commensense_qa, "commonsense_qa")
pickle_dataset(human_eval, "openai_humaneval")
pickle_dataset(mbpp, "mbpp")
pickle_dataset(rte, "rte")
pickle_dataset(mnli, "mnli")
pickle_dataset(CoT_examples, "CoT-Collection")


gsm8k_examples = get_few_shot_examples(dataset, dataset_name, example_type, disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)
gsm8k_few_shot_examples = get_few_shot_examples(dataset, dataset_name, example_type, disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)


# mnli = rte = load_dataset("glue", "mnli_matched")
# mnli = rte = load_dataset("glue", "mnli_mismatched")