from src.utils.evaluation import CodeEvaluator, Evaluation
import re
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
import os
from litellm import batch_completion
from nltk.tokenize import word_tokenize
import together
import openai
import backoff
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional 
from pathlib import Path
import time
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import random
import pickle
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
from enum import Enum, auto
from tqdm import tqdm
from pathlib import Path

from langchain.prompts.example_selector.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


DEBUG_NUM_QUESTIONS_WANT_ANSWER_PER_EXPERIMENT = 8


class TogetherAIGenerationParameters(TypedDict):
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    logprobs: int

class OpenAIGenerationParameters(TypedDict):
    model_name: str
    temperature: float
    max_tokens: int
    frequency_penalty: float

class DebugGenerationParameters(TypedDict):
    pass

def read_api_token(token_path : str) -> str:
    # Read API token from a dedicated file
    with open(token_path, "r") as f:
        API_TOKEN = f.read().strip()
    return API_TOKEN

# TYPES AND ENUMS
ID_TYPE = Union[str, int]
EXAMPLE_FORMAT_FUNCTION_TYPE = Callable[[Dict[str, Any], Optional[int]], str]
GENERATION_PARAMETERS_TYPE = Union[
    TogetherAIGenerationParameters, 
    OpenAIGenerationParameters, 
    DebugGenerationParameters,
]

class DatasetType(Enum):
    GSM8K_HARD = auto()
    GSM8K_HARD_CoT = auto()
    COMMON_SENSE = "COMMON_SENSE"
    COMMON_SENSE_CoT = auto()
    GSM8K = "GSM8K"
    MBPP = "MBPP"
    RTE = "RTE"
    MNLI = "MNLI"

class ModelAPIType(Enum):
    TOGETHER_AI = auto()
    OPEN_AI = auto()
    DEBUG = auto()

# DICTS
DATASET_ID_KEYS = {
    # TODO: Verify what these do and fi they are correct
    DatasetType.GSM8K_HARD_CoT : ['idx'],
    DatasetType.GSM8K_HARD : ['idx'],
    DatasetType.COMMON_SENSE : ['idx'],
    DatasetType.COMMON_SENSE_CoT : ['idx'],
    DatasetType.GSM8K : ['idx'],
    DatasetType.MBPP : ['task_id'],
    DatasetType.RTE : ['idx'],
    DatasetType.MNLI : ['idx'],
}

DATASET_INPUT_KEYS = {
    # TODO: Verify
    DatasetType.GSM8K_HARD_CoT : ['question'],
    DatasetType.GSM8K_HARD : ['input'],
    DatasetType.COMMON_SENSE : ['question','choices'],
    DatasetType.COMMON_SENSE_CoT : ['source'],
    DatasetType.GSM8K : ['question'],
    DatasetType.MBPP : ['text','test_list'],
    DatasetType.RTE : ['sentence1', 'sentence2'],
    DatasetType.MNLI : ['premise', 'hypothesis'],
}

DATASET_LABEL_KEYS = {
    # TODO: Verify
    DatasetType.GSM8K_HARD_CoT : ['answer'],
    DatasetType.GSM8K_HARD : ['target'],
    DatasetType.COMMON_SENSE : ['answerKey'],
    DatasetType.COMMON_SENSE_CoT : ['rationale', 'target'],
    DatasetType.GSM8K : ['answer'],
    DatasetType.MBPP : ['code', 'test_list', 'test_setup_code', 'challenge_test_list'],
    DatasetType.RTE : ['label'],
    DatasetType.MNLI : ['label'],
}

# these are the texts that go before the Q[i] in batch prompts
# currently unused
DATASET_BATCH_INDEX_Q = {
    DatasetType.GSM8K_HARD_CoT : ['Q'],
    DatasetType.GSM8K_HARD : ['Q'],
    DatasetType.COMMON_SENSE : ['Q'],
    DatasetType.COMMON_SENSE_CoT : ['Q'],
    DatasetType.GSM8K : ['Q'],
    DatasetType.MBPP : ['Q'],
    DatasetType.RTE : ['Premise', 'Hypothesis'],
    DatasetType.MNLI : ['Premise', 'Hypothesis'],
}

# these are the texts that go before the Q[i] in batch prompts
# currently unused
DATASET_BATCH_INDEX_A = {
    DatasetType.GSM8K_HARD_CoT : ['A'],
    DatasetType.GSM8K_HARD : ['A'],
    DatasetType.COMMON_SENSE : ['A'],
    DatasetType.COMMON_SENSE_CoT : ['A'],
    DatasetType.GSM8K : ['A'],
    DatasetType.MBPP : ['A'],
    DatasetType.RTE : ['A'],
    DatasetType.MNLI : ['A'],
}

class ExampleSelectionType(Enum):
    RANDOM = auto()
    SEMANTIC = auto()
    LEXICAL = auto()
    MAX_MARGINAL_RELEVANCE = auto()

@dataclass
class DatasetConfig:
    dataset : DatasetType
    # could be the name of a dataset, or a list of strings that specify the path to a dataset 
    # e.g. 'mbpp' vs ['mbpp', 'sanitized']
    hf_dataset_path: Optional[Union[str, List[str]]] = None
    task_name_for_CoT_filter: Optional[str] = None
    split_name: Optional[str] = None
    # can also choose to load a dataset from a json file
    local_path: Optional[Path] = None

    def __post_init__(self):
        # validate the config
        self.validate()

    def validate(self):
        match (self.hf_dataset_path is not None, self.split_name is not None, self.local_path is not None):
            case (True, False, _) | (False, True, _):
                raise ValueError("Must either both or neither specify a huggingface dataset path and a split name")
            case (True, True , True):
                raise ValueError("Cannot specify both a local path and a huggingface dataset path")
            case(False, False, False):
                raise ValueError("Must specify either a local path or a huggingface dataset path")
            case _: pass
        
        if self.local_path is not None:
            if not self.local_path.exists():
                raise ValueError(f"Local path {self.local_path} does not exist")
            if not self.local_path.is_file():
                raise ValueError(f"Local path {self.local_path} is not a file")
            if not self.local_path.suffix == '.json':
                raise ValueError(f"Local path {self.local_path} is not a json file")

@dataclass
class MultiTaskBatchPromptingDebugConfig:
    truncate_examples : bool = False,
    truncate_batch_queries : bool = False
    save_batched_model_inputs : Optional[Path] = None
    save_batched_model_outputs : Optional[Path] = None

@dataclass
class MultiTaskBatchPromptingExperimentConfig:
    # can either load a dataset from huggingface or from a local json file
    questions_dataset_config : Dict[DatasetType, DatasetConfig]
    task_descriptions: Dict[DatasetType, str]
    objective_instructions: str
    io_instructions: str
    k_shot: int
    batch_size: int
    question_format: Dict[DatasetType, EXAMPLE_FORMAT_FUNCTION_TYPE]
    model_api: ModelAPIType
    generation_params: GENERATION_PARAMETERS_TYPE
    debug : Optional[MultiTaskBatchPromptingDebugConfig] = None



# a list of examples (dicts with featues) of different types:

MT_EXAMPLE_TYPE = Tuple[DatasetType, Dict[str, Any]]
MT_ID_TYPE = Tuple[DatasetType, ID_TYPE]


class MultiTaskBatchPromptExperiment:
    def __init__(
            self,
            config: MultiTaskBatchPromptingExperimentConfig,
    ):
        self.config = config

        self.questions = {
            dataset_key : self.load_dataset(dataset_config)
            for dataset_key, dataset_config in self.config.questions_dataset_config.items()
        }

        # must add an index column to gsm8k
        self.debug = self.config.debug
        self.batch_prompt_template = MultiTaskBatchPromptTemplate(
            datasets=list(self.questions.keys()),
            objective_instructions=self.config.objective_instructions,
            task_descriptions=self.config.task_descriptions,
            io_instructions=self.config.io_instructions,
            num_questions=self.config.batch_size,
            question_format=self.config.question_format,
            debug=self.config.debug,
        )

    def load_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        if dataset_config.local_path is not None:
            # load locally
            dataset = load_dataset(
                'json', 
                data_files=dataset_config.local_path,
                split='train', # loading from file makes a dataset with only train split
            )
        else:
            # load from huggingface
            dataset = load_dataset(
                *dataset_config.hf_dataset_path,
                split=dataset_config.split_name,
            )
            if dataset_config.task_name_for_CoT_filter != None:
                dataset = dataset.filter(lambda example: example['task'] == dataset_config.task_name_for_CoT_filter)
                if dataset_config.task_name_for_CoT_filter =="rte":
                    dataset = dataset.filter(lambda example: "Question with options: can we draw the following hypothesis from the context?" in example['source'])
                if dataset_config.task_name_for_CoT_filter =="mnli":
                    dataset = dataset.filter(lambda example: "Premise: " in example['source'] and "Hypothesis: " in example['source'])
        # add an index column to gsm8k
        match dataset_config.dataset:
            case DatasetType.COMMON_SENSE | DatasetType.GSM8K:
                dataset = dataset.add_column('idx', list(range(len(dataset))))
        return dataset
    
    def batch_query_model(
        self, 
        model_inputs: List[Tuple[List[MT_ID_TYPE], str]]
    ) -> List[Tuple[List[MT_ID_TYPE], str]]:

        messages = [[{"role": "user", "content": i[1]}] for i in model_inputs]
        attempt_cnt = 0
        max_attempts = 10 
        model_query_batch_size = 10
        tokens_per_message = [len(word_tokenize(model_input[1])) for model_input in model_inputs]
        # Note: *2 is because output tokens will count towards our minutely token limit
        tokens_per_batch_completion = [sum(tokens_per_message[i:i+model_query_batch_size])*2 for i in range(0, len(messages), model_query_batch_size)]
        token_rate_limit = 160000 # This is for GPT 3.5 Turbo token limit per minute
        results = []
        
        model_name = self.config.generation_params["model_name"]
        generation_params = {
            k : v for k, v in self.config.generation_params.items() if k != "model_name"
        }

        message_sublists = [messages[i:i+model_query_batch_size] for i in range(0, len(messages), model_query_batch_size)]
        for batched_messages, tokens_in_batch in zip(message_sublists, tokens_per_batch_completion):
            while attempt_cnt < max_attempts:
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(batch_completion, model=model_name, messages = batched_messages, **generation_params,)
                    try:
                        batch_exectution_start_time = time.time()
                        curr_results = future.result(timeout=60)
                        batch_exectution_end_time = time.time()
                        batch_execution_time = batch_exectution_end_time - batch_exectution_start_time
                        results.extend(curr_results)
                        

                        # Calculate the time needed to wait based on tokens processed and token rate limit
                        # TODO: Not entirely sure about this logic
                        tokens_per_second = token_rate_limit / 60
                        expected_time_for_tokens = tokens_in_batch / tokens_per_second
                        sleep_time = max(0, expected_time_for_tokens - batch_execution_time)
                        print(f"batch_execution_time: {batch_execution_time}, tokens_per_second: {tokens_per_second}, expected_time_for_tokens: {expected_time_for_tokens}, sleep_time: {sleep_time}")
                        time.sleep(sleep_time)

                        break
                    except TimeoutError:
                        attempt_cnt += 1
                        print(f"Timeout error occurred. Retrying attempt {attempt_cnt}...")
                        time.sleep(20*attempt_cnt)  # Add a short delay before retrying
                    except Exception as e:
                        attempt_cnt += 1
                        print(f"Error {str(e)} occurred. Retrying attempt {attempt_cnt}...")
                        time.sleep(20*attempt_cnt)  # Add a short delay before retrying
                # try:
                #     responses = batch_completion(
                #     model=model_name,
                #     messages = batched_messages,
                #     **generation_params,
                #     )
                #     curr_results = [response["choices"][0]["message"]["content"] for response in responses]
                #     results.extend(curr_results)
                #     break
                # except Exception as e:
                #     attempt_cnt += 1
                #     print(f"Error {str(e)} occurred. Retrying...")
                #     time.sleep(120* attempt_cnt)
            if attempt_cnt == max_attempts:
                curr_results = ["" for i in range(len(batched_messages))]
                results.extend(curr_results)
        ids = [i[0] for i in model_inputs]
        return list(zip(ids, results))
    
    '''
    # TODO: Use this for GPT-3.5 Prompts Only
def query_batch(batched_messages):
    try:
        responses = batch_completion(
            model="gpt-3.5-turbo",
            messages=batched_messages,
            temperature=0,
        )
        return [response["choices"][0]["message"]["content"] for response in responses]
    except Exception as e:
        raise e

def batch_query_model(model_inputs):
    messages = [[{"role": "user", "content": i}] for i in model_inputs]
    attempt_cnt = 0
    max_attempts = 10 
    model_query_batch_size = 10
    results = []
    message_sublists = [messages[i:i+model_query_batch_size] for i in range(0, len(messages), model_query_batch_size)]
    tokens_per_message = [len(word_tokenize(model_input)) for model_input in model_inputs]
    # Note: *2 is because output tokens will count towards our minutely token limit
    tokens_per_batch_completion = [sum(tokens_per_message[i:i+model_query_batch_size])*2 for i in range(0, len(messages), model_query_batch_size)]
    token_rate_limit = 160000 # This is for GPT 3.5 Turbo token limit per minute

    for batched_messages, tokens_in_batch in zip(message_sublists, tokens_per_batch_completion):
        attempt_cnt = 0
        while attempt_cnt < max_attempts:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(query_batch, batched_messages)
                try:
                    batch_exectution_start_time = time.time()
                    curr_results = future.result(timeout=60)
                    batch_exectution_end_time = time.time()
                    batch_execution_time = batch_exectution_end_time - batch_exectution_start_time
                    results.extend(curr_results)
                    

                    # Calculate the time needed to wait based on tokens processed and token rate limit
                    # TODO: Not entirely sure about this logic
                    tokens_per_second = token_rate_limit / 60
                    expected_time_for_tokens = tokens_in_batch / tokens_per_second
                    sleep_time = max(0, expected_time_for_tokens - batch_execution_time)
                    print(f"batch_execution_time: {batch_execution_time}, tokens_per_second: {tokens_per_second}, expected_time_for_tokens: {expected_time_for_tokens}, sleep_time: {sleep_time}")
                    time.sleep(sleep_time)

                    break
                except TimeoutError:
                    attempt_cnt += 1
                    print(f"Timeout error occurred. Retrying attempt {attempt_cnt}...")
                    time.sleep(20*attempt_cnt)  # Add a short delay before retrying
                except Exception as e:
                    attempt_cnt += 1
                    print(f"Error {str(e)} occurred. Retrying attempt {attempt_cnt}...")
                    time.sleep(20*attempt_cnt)  # Add a short delay before retrying

        if attempt_cnt == max_attempts:
            curr_results = ["" for _ in range(len(batched_messages))]
            results.extend(curr_results)

    return results

    '''

    # returns a list of batches of questions and their associated ids
    def batch_questions(self) -> List[Tuple[List[MT_ID_TYPE], List[MT_EXAMPLE_TYPE]]]:
        #TODO: Should we truncate the longer dataset to have an equal number of examples per dataset?
        all_questions : Dict[DatasetType, MT_EXAMPLE_TYPE] = {}
        ids : Dict[DatasetType, MT_ID_TYPE] = {}
        for dataset_key, dataset in self.questions.items():
            all_questions[dataset_key] = []
            ids[dataset_key] = []
            for example in dataset:
                all_questions[dataset_key].append((dataset_key, example))
                example_id = example[DATASET_ID_KEYS[dataset_key][0]]
                ids[dataset_key].append((dataset_key, example_id))
        
        # batch the questions
        batched_questions  = []
        
        # the largest number divisible by both 6 and 8 that is less than 277, the number of examples in RTE
        truncation = 240
        
        randomized_indices : Dict[DatasetType, List[Tuple[DatasetType, int]]] = {}
        for dataset_key, dataset in self.questions.items():
            indices : List[Tuple[DatasetType, int]] = [(dataset_key, i) for i in range(len(dataset))]
            random.shuffle(indices)
            randomized_indices[dataset_key] = indices[:truncation]

        # by cycling one from each dataset at a time, we ensure that the batches are balanced 
        # (i.e. have at least one from each dataset)
        one_from_each = list(zip(*randomized_indices.values()))
        all_indices = list(itertools.chain(*one_from_each))

        # will contain batches of (dataset, index) tuples
        batched_indices : List[List[Tuple[DatasetType, int]]] = []
        for i in range(0, len(all_indices), self.config.batch_size):
            batch = all_indices[i:i+self.config.batch_size]
            random.shuffle(batch)
            batched_indices.append(batch)
        
        # now we have a list of batches of (dataset, index) tuples
        batched_ids : List[List[MT_ID_TYPE]] = []
        batched_examples : List[List[MT_EXAMPLE_TYPE]] = []
        for batch in batched_indices:
            batched_ids.append([ids[ds][i] for (ds, i) in batch])
            batched_examples.append([all_questions[ds][i] for (ds, i) in batch])
        
        batched_questions : List[
            Tuple[List[MT_ID_TYPE],
            List[MT_EXAMPLE_TYPE]]
        ] = list(zip(batched_ids, batched_examples))

        return batched_questions
    
    # gives a dict from (dataset_type, question_id) to the datapoint dict (which has the answers)
    def answers_from_batched_questions(
        self, 
        batched_questions: List[Tuple[List[MT_ID_TYPE], List[MT_EXAMPLE_TYPE]]]
    ) -> Dict[MT_ID_TYPE, Dict[str, Any]]:
        answers :  Dict[ID_TYPE, Dict[str, Any]] = {
            (dataset_type, question_id) : question
            for (ids, questions) in batched_questions
            for ((dataset_type, question_id), (_, question)) in zip(ids, questions)
        }
        return answers


    def execute(self):
        """
        TODO:
        X Load Dataset
        X Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
        X query model for each input (save raw outputs to a file somewhere)
        """
        # splits self.questions into batches that are lists of individual dictionaries along with their ids
        batched_questions: List[Tuple[List[MT_ID_TYPE], List[MT_EXAMPLE_TYPE]]] = self.batch_questions()
        answers_dict : Dict[MT_ID_TYPE, Dict[str, Any]] = self.answers_from_batched_questions(batched_questions)

        # generate prompts for each batch
        batched_model_inputs : List[Tuple[List[ID_TYPE], str]] = [
            (ids, self.batch_prompt_template.generate_prompt(batch))
            for (ids, batch) in batched_questions
        ]
        # for debug purposes
        if self.debug:
            total_number_examples_wanted = DEBUG_NUM_QUESTIONS_WANT_ANSWER_PER_EXPERIMENT
            needed_llm_calls = total_number_examples_wanted/self.config.batch_size
            if needed_llm_calls.is_integer():
                needed_llm_calls = int(needed_llm_calls)
            else:
                needed_llm_calls = math.ceil(needed_llm_calls)
            if needed_llm_calls == 0:
                needed_llm_calls = 1
            needed_llm_calls = min(needed_llm_calls, len(batched_model_inputs))
            batched_model_inputs = batched_model_inputs[:needed_llm_calls]

        batched_model_outputs = self.batch_query_model(batched_model_inputs)
        if batched_model_inputs:
            if self.debug.save_batched_model_inputs:
                pickle.dump((batched_model_inputs), open(self.debug.save_batched_model_inputs, 'wb'))
            if self.debug.save_batched_model_outputs:
                pickle.dump((batched_model_outputs), open(self.debug.save_batched_model_outputs, 'wb'))

        return (batched_model_inputs, batched_model_outputs, answers_dict)

class MultiTaskBatchPromptTemplate:
    """
    MultiTaskBatchPromptTemplate is a class that generates prompts for a batch of examples that can have a mix of tasks.
    It is used to generate prompts for the multi-task k-shot experiment.

    prompts have the following:

    Objective:

    Task and Token(?) Description(s):

    Unified Input/Output Format Instructions:

    Examples:

    Batched Questions:
    """
    def __init__(
            self,
            datasets: List[DatasetType],
            objective_instructions: str,
            task_descriptions: Dict[DatasetType, str],
            io_instructions: str,
            num_questions: int,
            question_format: Dict[DatasetType, Callable[[Dict[str, Any], Optional[int]], str]],
            debug: Optional[MultiTaskBatchPromptingDebugConfig] = None,
    ):
        self.datasets = datasets
        self.objective_instructions = objective_instructions
        self.task_descriptions = task_descriptions
        self.io_instructions = io_instructions
        self.num_questions = num_questions
        self.question_format = question_format
        self.debug = debug

    def generate_prompt(self, batch: List[MT_EXAMPLE_TYPE]) -> str:
        """
        Generates a prompt for a batch of examples
        """
        objective_instructions = self.objective_instructions

        task_descriptions = "Task Descriptions:\n" + "\n".join([
            self.task_descriptions[task]
            for task in self.datasets
        ])
        
        io_instructions = self.io_instructions

        batched_questions = "\n".join([
            f"{self.question_format[dataset](example, i)}"
            for i, (dataset, example) in enumerate(batch)
        ])

        prompt = '''\
{objective_instructions}

{task_and_token_descriptions}

{io_instructions}

{batched_questions}'''.format(
            objective_instructions=objective_instructions.format(batch_size=self.num_questions),
            task_and_token_descriptions=task_descriptions,
            io_instructions=io_instructions,
            batched_questions=batched_questions,
        )
        return prompt

DATASET_QUESTIONS_CONFIGS = {
    DatasetType.RTE : DatasetConfig(
        dataset=DatasetType.RTE,
        hf_dataset_path=['glue', 'rte'],
        split_name='validation',
    ),
    DatasetType.COMMON_SENSE : DatasetConfig(
        dataset=DatasetType.COMMON_SENSE,
        hf_dataset_path=['commonsense_qa'],
        split_name='validation',
    ),
    DatasetType.GSM8K : DatasetConfig(
        dataset=DatasetType.GSM8K,
        hf_dataset_path=['gsm8k', 'main'],
        split_name='test',
    ),
    DatasetType.MNLI : DatasetConfig(
        dataset=DatasetType.MNLI,
        hf_dataset_path=['glue', 'mnli'],
        split_name='validation_matched',
    ),
}

#TODO: Alex fill in/verify task descriptions, some aren't completely consistent (i.e. mention batch_size or not)
DATASET_TASK_DESCRIPTIONS = {
    DatasetType.COMMON_SENSE : '''\
COMMON_SENSE: Instruction - our task is to solve a set of multiple-choice questions from the CommonsenseQA dataset in a batch. CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . You will be given {{batch_size}} questions each time, as input. These questions are designed to test your ability to answer queries that often require contextual understanding and general world knowledge. Your goal is to select the letter corresponding to the most appropriate answer among five options labeled 'a', 'b', 'c', 'd', and 'e' for each question in the batch.
COMMON_SENSE: Method - Use your expertise in NLP and contextual understanding to perform a sequence of logical evaluations for solving these questions.
COMMON_SENSE: Intermediate Reasoning - Include all the steps you took to arrive at your answer. This could include identifying key phrases, contradictions, or logical connections that led you to choose a particular option.
COMMON_SENSE: Output Meaning - Select the most appropriate answer and output the letter after "The answer is" with the corresponding letter.''',
    DatasetType.GSM8K : '''\
GSM8K: Instruction - Your task is to solve a set of math questions in a batch.
GSM8K: Method - Use basic arithmetic operations to perform a sequence of calculations for solving these questions.
GSM8K: Intermediate Reasoning -  Each question in the batch will require you to perform between 2 and 8 steps to arrive at the final answer.
GSM8K: Output Meaning - Each answer is an integer that is the answer to the question.''',
    DatasetType.RTE : "", 
    DatasetType.MNLI : '''\
MNLI: Instruction - Your task is to solve a set of MultiNLI (MNLI) questions in a batch.  You will be given premise-hypothesis pairs from the MNLI dataset as input. Your goal is to classify each pair into one of three classes: entailment, neutral, or contradiction.
MNLI: Method - Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations relationship between each Premise and Hypothesis pair. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).
MNLI: Intermediate Reasoning - Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.
MNLI: Output Meaning - An answer of 0 means the premise entails the hypothesis, indicating that if the premise is true, the hypothesis must also be true. In this case, the information in the hypothesis is a logical subset of the information in the premise.
An answer of 1 means the relationship between the premise and the hypothesis is neutral, suggesting that the truth of the premise neither guarantees nor contradicts the truth of the hypothesis. The hypothesis could be either true or false regardless of the premise's truth value.
An answer of 2 means the premise contradicts the hypothesis, implying that both cannot be true at the same time. If the premise is true, the hypothesis must necessarily be false, and vice versa.'''
}

QUESTION_FORMAT_FUNCTIONS = {
    DatasetType.COMMON_SENSE : lambda example, i: f"",
    DatasetType.GSM8K : lambda example, i: f"",
    DatasetType.RTE : lambda example, i: f"",
    DatasetType.MNLI : lambda example, i: f"",
}

IO_INSTRUCTIONS = ""
OBJECTIVE_INSTRUCTIONS = ""

oai_gen_params = OpenAIGenerationParameters(
    model_name='gpt-3.5-turbo',
    temperature=0.6,
    max_tokens=256,
    frequency_penalty=1.0,
)

def run_experiments():
    batch_sizes = []
    dataset_types = [
        DatasetType.COMMON_SENSE,
        DatasetType.GSM8K,
        DatasetType.RTE,
        DatasetType.MNLI,
    ]
    
    dataset_combinations = list(itertools.chain(*list(itertools.combinations(dataset_types, i) for i in [1,2,3,4])))

    dataset_combination_to_output = {}
    for dataset_combination in dataset_combinations:
        config = MultiTaskBatchPromptingExperimentConfig(
            questions_dataset_config={
                dataset_type: DATASET_QUESTIONS_CONFIGS[dataset_type]
                for dataset_type in dataset_combination
            },
            task_descriptions={
                dataset_type: DATASET_TASK_DESCRIPTIONS[dataset_type]
                for dataset_type in dataset_combination
            },
            objective_instructions=OBJECTIVE_INSTRUCTIONS,
            io_instructions=IO_INSTRUCTIONS,
            k_shot=0,
            batch_size=4,
            question_format={
                dataset_type: QUESTION_FORMAT_FUNCTIONS[dataset_type]
                for dataset_type in dataset_combination
            },
            model_api=ModelAPIType.OPEN_AI,
            generation_params=oai_gen_params
        )
        experiment = MultiTaskBatchPromptExperiment(config)
        output = experiment.execute()
        dataset_combination_to_output[dataset_combination] = output
        (batched_model_inputs, batched_model_outputs, answers_dict) = experiment.execute()


if __name__ == "__main__":
    # run_batched_tests(config_param_list, config_to_answer_type)
    print("Hello World")

    questions_config_rte = DatasetConfig(
        dataset=DatasetType.RTE,
        hf_dataset_path=['glue', 'rte'],
        split_name='validation',
    )

    questions_config_COMMON_SENSE = DatasetConfig(
        dataset=DatasetType.COMMON_SENSE,
        hf_dataset_path=['commonsense_qa'],
        split_name='validation',
    )

    task_description_rte = '''\
RTE: Instruction - Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes.
RTE: Method - Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.
RTE: Intermediate Reasoning - Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections. If the answer is ambiguous, pick the answer that is most likely to be true.
RTE: Input Format - Each question will  be prefixed with its index and followed by a sentence pair labeled as "Premise" and "Hypothesis" starting from 0, like so:
Q[0][RTE]
P: {{Premise_0_Text}}
H: {{Hypothesis_0_Text}}
...
Q[{{batch_size - 1}}][RTE]
P: {{Premise_{{batch_size - 1}}_Text}}
H: {{Hypothesis_{{batch_size - 1}}_Text}}
RTE: Output Meaning - An answer of 0 means that the given Hypothesis and Premise logically entail each other.  An answer of 1 means the given Hypothesis and Premise do NOT entail each other.
'''

    task_description_COMMON_SENSE = "COMMON_SENSE_QA: Answer questions using common sense and implicit knowledge."

    objective_instructions = "Objective: Your task is to solve a variety of questions across multiple domains in a single batch operation. You will be given a number of questions, each associated with a specific task domain, and your goal is to answer each question according to its domain while adhering to the desired output format. The total number of questions in the batch to answer is defined as batch_size = {batch_size}."

    io_instructions = """\
#### Input Format:
- Questions will be presented in a batch. Each question will be prefixed with its index, starting from 0, like so:
Q[0]: {{Question_0_Text}}
Q[1]: {{Question_1_Text}}
...
Q[{{batch_size - 1}}]: {{Question_{{batch_size - 1}}_Text}}

#### Output Format:
- You must adhere to the following format rigorously for each answer:
A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
- `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
- `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
- `{{Answer_Integer}}`: This is the final integer answer to each question.

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Ensure you output A[index] for each question before outputting {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that exactly {batch_size} answers are provided in our desired format. Do not include ANY reasoning after "The answer is", just the designated answer symbol.
"""

    def commonsense_question_format(example, i):
        prompt = f"Q[{i}][COMMON_SENSE] {example['question']}\n"
        prompt += "\n".join(
            [f"{label}: {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])]
        )
        return prompt

    rte_question_format = lambda example, i: f"Q[{i}][RTE]\nP: {example['sentence1']}\nH: {example['sentence2']}"
    # commonsense_question_format = lambda example, i: f"Q[{i}][COMMON_SENSE]\n{example['question']}\n{example['choices']}"


    oai_gen_params = OpenAIGenerationParameters(
        model_name='gpt-3.5-turbo',
        temperature=0.6,
        max_tokens=256,
        frequency_penalty=1.0,
    )

    os.environ['OPENAI_API_KEY'] = read_api_token(Path("data/imported/open_ai_token.txt"))

    config = MultiTaskBatchPromptingExperimentConfig(
        questions_dataset_config={
            DatasetType.RTE: questions_config_rte,
            DatasetType.COMMON_SENSE: questions_config_COMMON_SENSE,
        },
        task_descriptions={
            DatasetType.RTE: task_description_rte,
            DatasetType.COMMON_SENSE: task_description_COMMON_SENSE,
        },
        objective_instructions=objective_instructions,
        io_instructions=io_instructions,
        k_shot=0,
        batch_size=4,
        question_format={
            DatasetType.RTE: rte_question_format,
            DatasetType.COMMON_SENSE: commonsense_question_format,
        },
        model_api=ModelAPIType.OPEN_AI,
        generation_params=oai_gen_params,
        debug=MultiTaskBatchPromptingDebugConfig(
            truncate_examples=False,
            truncate_batch_queries=False,
            save_batched_model_inputs=Path("mt_batched_model_inputs.pkl"),
            save_batched_model_outputs=Path("mt_batched_model_outputs.pkl"),
        ),
    )

    experiment = MultiTaskBatchPromptExperiment(config)
    experiment.execute()
