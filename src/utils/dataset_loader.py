
from datasets import load_dataset
import pickle
from typing import List, Tuple
import re

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
    Given a list of disallowed indices, return a set of indices that are allowed.

    Parameters:
    - disallowed_data_indices (list): List of disallowed indices.
    - total_length (int): The total length of the list for which we are finding allowed indices.

    Returns:
    - set: Set of allowed indices.
    """
    if disallowed_data_indices is None:
        return set(range(total_length))
    else:
        return {i for i in range(total_length) if i not in disallowed_data_indices}
import numpy as np

def get_few_shot_examples(dataset, dataset_name, example_type = "End-to-end", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None):
    
    if dataset_name == "gsm8k":
        allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]['answer']))
        questions_np = np.array(dataset[data_location]['question'])
        questions = questions_np[list(allowed_indices)]
        examples_with_CoT_np = np.array(dataset[data_location]['answer'])
        examples_with_CoT = examples_with_CoT_np[list(allowed_indices)]
        if example_type == "CoT":
            def transform_to_prompts(examples_with_CoT):
                transformed_prompts = []
                extract_numbers = lambda x: x.split("####")[-1].strip()

                for example in examples_with_CoT:
                    extracted_value = extract_numbers(example)

                    # Construct the new prompt
                    new_prompt = f"Step by Step thinking:\n{example}\n\nFinal Output:\n{extracted_value}"

                    transformed_prompts.append(new_prompt)
                return transformed_prompts
            examples_with_CoT_formatted = transform_to_prompts(examples_with_CoT)
            return questions, examples_with_CoT_formatted
        else:
            extract_numbers = lambda x: x.split("####")[-1].strip()
            extracted_values = [extract_numbers(example) for example in examples_with_CoT]
            return questions, extracted_values
    elif dataset_name =="gsm_hard":
        if example_type == "CoT":
            # TODO: For now using gsm8k CoT examples because gsm-hard rather gives us code to solve but not step by step answers for code (ex: eggs_sold = eggs_per_day - eggs_eaten - eggs_baked instead of eggs_sold = 100 - 10 - 5)
            assert needed_extra_dataset != None
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(needed_extra_dataset[data_location]))
            questions_np = np.array(needed_extra_dataset[data_location]['question'])
            questions = questions_np[list(allowed_indices)]
            # questions = [needed_extra_dataset[data_location]['input'][i] for i in allowed_indices]
            examples_with_CoT_np = np.array(needed_extra_dataset[data_location]['answer'])
            examples_with_CoT = examples_with_CoT_np[list(allowed_indices)]
            def transform_to_prompts(examples_with_CoT):
                transformed_prompts = []
                extract_numbers = lambda x: x.split("####")[-1].strip()

                for example in examples_with_CoT:
                    extracted_value = extract_numbers(example)

                    # Construct the new prompt
                    new_prompt = f"Step by Step thinking:\n{example}\n\nFinal Output:\n{extracted_value}"

                    transformed_prompts.append(new_prompt)
                return transformed_prompts
            examples_with_CoT_formatted = transform_to_prompts(examples_with_CoT)
            # examples_with_CoT = [needed_extra_dataset[data_location]['answer'][i] for i in allowed_indices]
            return questions, examples_with_CoT_formatted
        else:
            assert disallowed_data_indices == None
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]['target'])) 
            questions_np = np.array(dataset[data_location]['input'])
            questions = questions_np[list(allowed_indices)]
            answers_np = np.array(dataset[data_location]['target'])
            answers = answers_np[list(allowed_indices)]
            # questions =  [dataset[data_location]['input'][i] for i in allowed_indices]
            # answers =  [dataset[data_location]['target'][i] for i in allowed_indices]
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
            questions_np = np.array(dataset[data_location]['question'])
            choices_np = np.array(dataset[data_location]['choices'])
            answers_np = np.array(dataset[data_location]['answerKey'])
            questions = questions_np[list(allowed_indices)]
            choices = choices_np[list(allowed_indices)]
            answers = answers_np[list(allowed_indices)]
            # question = [dataset[data_location]['question'][i] for i in allowed_indices]
            # choices = [dataset[data_location]['choices'][i] for i in allowed_indices]
            # answers = [dataset[data_location]['answerKey'][i] for i in allowed_indices]
            def build_question_string(question, choices):
                question_str = f"Question:\n\n{question}\nAnswer Choices:\n\n"
                choice_labels = choices['label']
                choice_texts = choices['text']
                
                for label, text in zip(choice_labels, choice_texts):
                    question_str += f"{label}: {text}\n"
                
                return question_str
            question_with_answer_choices = [build_question_string(question, choices) for question, choices in zip(questions, choices)]
            return question_with_answer_choices, answers
    elif dataset_name == "mbpp":
        if example_type == "CoT":
            # TODO: Create our own CoT examples for MBPP
            raise NotImplementedError()
        else:
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]))
            questions_np = np.array(dataset[data_location]['text'])
            answers_np = np.array(dataset[data_location]['code'])
            questions = questions_np[list(allowed_indices)]
            answers = answers_np[list(allowed_indices)]
            # example_questions = [dataset[data_location][i]['text'] for i in allowed_indices]
            # example_answers = [dataset[data_location][i]['code'] for i in allowed_indices]
            return questions, answers
    # TODO: implement
    elif dataset_name == "rte":
        
        if example_type == "CoT":
            # use CoT-Collection 
            assert needed_extra_dataset != None
            rte_wit_CoT =  needed_extra_dataset['train'].filter(lambda example: example['task'] == 'rte') 
            rte_wit_CoT_source_np = np.array(rte_wit_CoT['source'])
            rte_wit_CoT_format_1_indices = [i for i, x in enumerate(rte_wit_CoT_source_np) if "Question with options: can we draw the following hypothesis from the context?" in x]
            rte_wit_CoT_source_desired_format = rte_wit_CoT_source_np[rte_wit_CoT_format_1_indices]
            rte_wit_CoT_labels_np = np.array(rte_wit_CoT['target'])
            rte_wit_CoT_labels_desired_format = rte_wit_CoT_labels_np[rte_wit_CoT_format_1_indices]
            rationales_np = np.array(rte_wit_CoT['rationale'])
            rationales_np_desired_format = rationales_np[rte_wit_CoT_format_1_indices]

            def extract_contexts_hypothesises(texts, want_options_in_hypothesis = False):
                debug_dict = {'missing_context': 0, 'missing_hypothesis': 0}
                
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
                
                contexts = []
                hypothesises = []
                for text in texts:
                    try:
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
                        
                        # Update debug_dict for missing fields
                        if context is None:
                            debug_dict['missing_context'] += 1
                        if hypothesis is None:
                            debug_dict['missing_hypothesis'] += 1
                        contexts.append(context)
                        hypothesises.append(hypothesis)
                    
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        # TODO: might want to keep track of indices so could remove later so not mismatch between labels
                        continue
                return contexts, hypothesises
            # TODO: Might want to change how we format. But would want to keep consistent with non CoT examples
            contexts, hypothesises = extract_contexts_hypothesises(rte_wit_CoT_source_desired_format)
            # TODO: Might want to specify what is hypothesis and what is context
            def combine_rte_sentences(sentence_1, sentence_2):
                return f"Sentence 1:\n{sentence_1}\n\nSentence 2:\n{sentence_2}"
            combined_sentences = [combine_rte_sentences(sentence_1, sentence_2) for sentence_1, sentence_2 in zip(contexts, hypothesises)]
            # TODO: Write function that does this
            def combine_rationale_and_answer(rationales_np_desired_format, rte_wit_CoT_labels_desired_format):
                strings = []
                for rationale_np_desired_format, rte_wit_CoT_label_desired_format in zip(rationales_np_desired_format, rte_wit_CoT_labels_desired_format):
                    string = "Step by Step thinking:\n" + rationale_np_desired_format + "\n\nFinal Answer:\n" + rte_wit_CoT_label_desired_format
                    strings.append(string)
                return string
            rationales_with_labels = combine_rationale_and_answer(rationales_np_desired_format, rte_wit_CoT_labels_desired_format)
            return combined_sentences, rationales_with_labels

        else:
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]))
            sentence_1_np = np.array(dataset[data_location]['sentence1'])
            sentence_2_np = np.array(dataset[data_location]['sentence2'])
            labels_np = np.array(dataset[data_location]['label'])
            sentence_1 = sentence_1_np[list(allowed_indices)]
            sentence_2 = sentence_2_np[list(allowed_indices)]
            labels = labels_np[list(allowed_indices)]   
            # TODO: Might want to specify what is hypothesis and what is context
            def combine_rte_sentences(sentence_1, sentence_2):
                return f"Sentence 1:\n{sentence_1}\n\nSentence 2:\n{sentence_2}"
            combined_sentences = [combine_rte_sentences(sentence_1, sentence_2) for sentence_1, sentence_2 in zip(sentence_1, sentence_2)]
            labels_as_yes_no = ["yes" if label == 1 else "no" for label in labels]
            # TODO: Might want to convert labels to string (yes/no)
            return combined_sentences, labels_as_yes_no
    elif dataset_name == "mnli":
        if example_type == "CoT":
            # use CoT-Collection 
            assert needed_extra_dataset != None
            mnli_CoT =  needed_extra_dataset['train'].filter(lambda example: example['task'] == 'mnli')
            questions = mnli_CoT['source']
            rationales = mnli_CoT['rationale']
            labels = mnli_CoT['target']
            labels_as_string = ["entailment" if label == 'yes' else "neutral" if label == 'it is not possible to tell' else "contradiction" for label in labels]
            def build_CoT_answer(rationale, labels):
                # Remove any newline characters from the end of the rationale
                rationale_cleaned = rationale.rstrip("\n")
                # Construct the CoT example
                CoT_example = f"Step by Step Thinking:\n{rationale_cleaned}\n\nFinal Answer:\n{labels}"
                return CoT_example
            CoT_answers = [build_CoT_answer(rationale, answer) for rationale, answer in zip(rationales, labels_as_string)]
            # TODO: decide if want to return string representation or int representation
            return questions, CoT_answers
        else:
            allowed_indices = get_allowed_indices(disallowed_data_indices, len(dataset[data_location]))
            premises_np = np.array(dataset[data_location]['premise'])
            premises = premises_np[list(allowed_indices)]
            hypothesises_np = np.array(dataset[data_location]['hypothesis'])
            hypothesises = hypothesises_np[list(allowed_indices)]
            labels_np = np.array(dataset[data_location]['label'])
            labels = labels_np[list(allowed_indices)]
            # premises = [dataset[data_location]['premise'][i] for i in allowed_indices]
            # hypothesises = [dataset[data_location]['hypothesis'][i] for i in allowed_indices]
            # labels =  [dataset[data_location]['label'][i] for i in allowed_indices]
            def build_mnli_question(premise, hypothesis):
                return f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"
            premise_with_hypothesis_list = [build_mnli_question(premise, hypothesis) for premise, hypothesis in zip(premises, hypothesises)]
            # labels are entailment (0), neutral (1), contradiction (2)
            labels_as_string = ["entailment" if label == 0 else "neutral" if label == 1 else "contradiction" for label in labels]
            # TODO: decide if want to return string representation or int representation
            return premise_with_hypothesis_list, labels_as_string
    else:
        raise ValueError("Invalid dataset name")
    
def filter_dataset_by_task(dataset, acceptable_tasks={"commonsenseqa", "mnli", "rte"}):
    """
    Filters a dataset to include only rows with 'task' field in acceptable_tasks.

    Parameters:
    - dataset (datasets.Dataset): The dataset to filter.
    - acceptable_tasks (set): The set of acceptable tasks.

    Returns:
    - datasets.Dataset: The filtered dataset.
    """
    # Perform the filter operation
    filtered_dataset = dataset.filter(lambda example: example['task'] in acceptable_tasks)

    return filtered_dataset
# Load the gsm8k dataset
# TODO: Confirm all below
LOAD_PICKLE = False
LOAD_WITH_HUGGINGFACE = False
LOAD_EXAMPLES = False

if LOAD_PICKLE:
    gsm8k = load_pickle("data//imported//datasets//pickled//gsm8k")
    gsm_hard = load_pickle("data//imported//datasets//pickled//gsm-hard")
    commonsense_qa = load_pickle("data//imported//datasets//pickled//commonsense_qa")
    # human_eval = load_pickle("data//imported//datasets//pickled//human_eval")
    mbpp = load_pickle("data//imported//datasets//pickled//mbpp")
    rte = load_pickle("data//imported//datasets//pickled//rte")
    mnli = load_pickle("data//imported//datasets//pickled//mnli")
    # TODO: Comment out
    # CoT_examples = load_pickle("data//imported//datasets//pickled//CoT-Collection")
    # TODO: Delete these two lines after run once
    # CoT_examples_desired_tasks_only = filter_dataset_by_task(CoT_examples, acceptable_tasks={"commonsenseqa", "mnli", "rte"})
    # pickle_dataset(CoT_examples_desired_tasks_only, "CoT-Collection-desired-tasks-only")
    CoT_examples_desired_tasks_only = load_pickle("data//imported//datasets//pickled//CoT-Collection-desired-tasks-only")
elif LOAD_WITH_HUGGINGFACE: 
    gsm8k = load_dataset("gsm8k", "main")  # well organized, easy to use
    gsm_hard = load_dataset("reasoning-machines/gsm-hard") # well organized, easy to use
    commonsense_qa = load_dataset("commonsense_qa") # train and validation hold gold truth labels, test does not
    # human_eval = load_dataset("openai_humaneval") # test cases in human_eval["test"]["test"], but hard to parse and would be fair bit of work
    mbpp = load_dataset("mbpp") # test cases well organized as list for each code generation in mbpp["train"]['test_list']
    rte = load_dataset("glue", "rte") # well organized, easy to use, ignore test because no ground truth
    mnli = load_dataset("glue", "mnli") # load_dataset("glue", "mnli_matched"), load_dataset("glue", "mnli_mismatched")
    CoT_examples = load_dataset("kaist-ai/CoT-Collection")
    CoT_examples_desired_tasks_only = filter_dataset_by_task(CoT_examples, acceptable_tasks={"commonsenseqa", "mnli", "rte"})
    # Pickle the datasets
    pickle_dataset(gsm8k, "gsm8k")
    pickle_dataset(gsm_hard, "gsm-hard")
    pickle_dataset(commonsense_qa, "commonsense_qa")
    # pickle_dataset(human_eval, "openai_humaneval")
    pickle_dataset(mbpp, "mbpp")
    pickle_dataset(rte, "rte")
    pickle_dataset(mnli, "mnli")
    pickle_dataset(CoT_examples, "CoT-Collection")
    pickle_dataset(CoT_examples_desired_tasks_only, "CoT-Collection-desired-tasks-only")

if LOAD_EXAMPLES:
    gsm8k_examples = get_few_shot_examples(gsm8k, "gsm8k", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)
    gsm8k_examples_CoT = get_few_shot_examples(gsm8k, "gsm8k", example_type = "CoT", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)

    gsm_hard_examples = get_few_shot_examples(gsm_hard, "gsm_hard", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)
    gsm_hard_examples_CoT = get_few_shot_examples(gsm_hard, "gsm_hard", example_type = "CoT", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = gsm8k)

    commonsense_qa_examples = get_few_shot_examples(commonsense_qa, "commonsense_qa", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)
    commonsense_qa_examples_CoT = get_few_shot_examples(commonsense_qa, "commonsense_qa", example_type = "CoT", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = CoT_examples_desired_tasks_only)

    mbpp_examples = get_few_shot_examples(mbpp, "mbpp", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)

    rte_examples = get_few_shot_examples(rte, "rte", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)
    rte_examples_CoT = get_few_shot_examples(rte, "rte", example_type = "CoT", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = CoT_examples_desired_tasks_only)

    mnli_examples = get_few_shot_examples(mnli, "mnli", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = None)
    mnli_examples_CoT = get_few_shot_examples(mnli, "mnli", example_type = "CoT", disallowed_data_indices = None, data_location = "train", needed_extra_dataset = CoT_examples_desired_tasks_only)
