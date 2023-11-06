# from src.experiments.k_shot_experiment_configs import * # config_param_list, oai_gen_params, config_to_answer_type
# from src.experiments.k_shot_experiment import * # BatchPromptExperiment,BatchPromptingExperimentConfig
from src.utils.evaluation import CodeEvaluator, Evaluation
import re
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# import litellm
import os
from litellm import batch_completion

import together
import openai
import backoff
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional 
from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import math

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

class LanguageModel:
    def __init__(self):
        raise NotImplementedError

    def query(self, prompt : str) -> str:
        raise NotImplementedError

class TogetherAIModel(LanguageModel):
    """ 
    A wrapper class for the Together AI API.
    
    NOTE: do NOT specify any default values for model/generation parameters.
    I'd like to keep the configurations separate from the code as much as possible
    so we have good documentation/reproducibility of the experiments without having
    to dig into the code/commit history.

    Attributes:
        api_token (str): The API token for the Together AI API.
        model_name (str): The name of the model to use.
            Llama2-70B is togethercomputer/llama-2-70b, for example.
        generation_params (dict): The parameters to use for generation.
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The temperature to use for generation.
            - top_p (float): The top_p to use for generation.
            - top_k (int): The top_k to use for generation.
            - repetition_penalty (float): The repetition penalty to use for generation.
            - logprobs (int): The number of logprobs to return.
    """
    def __init__(
        self,
        api_token : str,
        model_name : str,
        generation_params : dict,
    ):
        together.api_key = api_token
        self.api_token = api_token
        self.model_name = model_name
        self.generation_params = generation_params

    def __repr__(self):
        return f"TogetherAIModel(model_name={self.model_name}, generation_params={self.generation_params})"

    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def query(self, prompt : str) -> str:
        if "model_name" in self.generation_params:
            model_name = self.generation_params["model_name"]
        for attempt in range(10):  # Try up to 10 times
            try:
                response = together.Complete.create(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=0,
                    max_tokens=2048,
                    # frequency_penalty=1.0
                    # **self.generation_params,
                )
                return response["output"]["choices"][0]["text"]
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
                time.sleep(1)  # Sleep for 1 second before retrying
        z = ""
        return z
    

# with ThreadPoolExecutor() as executor:
#     future = executor.submit(openai.ChatCompletion.create,
#                             model=self.model_name,
#                             messages=message,
#                             **self.generation_params,)
#     response = future.result(timeout=timeout)
#     text_response = response["choices"][0]["message"]["content"]

class OpenAIModel(LanguageModel):
    def __init__(
        self,
        api_token,
        model_name,
        generation_params,
    ):
        openai.api_key = api_token
        self.api_token = api_token
        self.model_name = model_name
        self.generation_params = {key : value for key, value in generation_params.items() if key != 'model_name'}
        


    def __repr__(self):
        return f"OpenAIModel(model_name={self.model_name}, generation_params={self.generation_params})"
        
    # @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=10)
    def query(self, prompt : str):
        message = [{"role": "user", "content": prompt}]
        # Estimate the number of tokens used
        estimated_tokens = len(prompt.split()) * 3
        # Set the rate limits for different models
        rate_limit = 10_000 if "gpt-4" in self.model_name else 90_000
        
        try_cnt = 0
        max_try_cnt = 10
        timeout = 25
        
        while try_cnt < max_try_cnt:
            try:
                time.sleep(3)
                response = openai.ChatCompletion.create(
                                    model=self.model_name,
                                    messages=message,
                                    **self.generation_params)
                text_response = response["choices"][0]["message"]["content"]
                return text_response
            except Exception as e:
                wait_time = (estimated_tokens / rate_limit) * 60 * (1 + try_cnt**2 / 4)
                print(f"Error {str(e)} occurred. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                try_cnt += 1
        if try_cnt ==10:
            error_message = "Errors occurred too many times. Aborting..."
            print(error_message)
            return(error_message)
        
    
class DebugModel(LanguageModel):
    def __init__(self):
        pass

    def __repr__(self):
        return f"DebugModel(model_name={self.model_name}, generation_params={self.generation_params})"
    
    def query(self, prompt : str) -> str:
        print(f"Model Recieved: {prompt}")
    

# def query_model(model, prompt, model_temperature=.2, timeout=10):
#     message = [{"role": "user", "content": prompt}]
#     # Estimate the number of tokens used
#     estimated_tokens = len(prompt.split()) * 3
#     # Set the rate limits for different models
#     rate_limit = 10_000 if "gpt-4" in model else 90_000  
#     try_cnt = 0
#     max_try_cnt = 10  
#     while try_cnt < max_try_cnt:
#         with ThreadPoolExecutor() as executor:
#             future = executor.submit(openai.ChatCompletion.create,
#                                      model=model,
#                                      messages=message,
#                                      temperature=model_temperature,
#                                      frequency_penalty=0.0)
#             try:
#                 response = future.result(timeout=timeout)
#                 text_response = response["choices"][0]["message"]["content"]
#                 return text_response
#             except (TimeoutError, Exception) as e:
#                 wait_time = (estimated_tokens / rate_limit) * 60 * (1 + try_cnt / 4)
#                 print(f"Error {str(e)} occurred. Waiting for {wait_time} seconds...")
#                 time.sleep(wait_time)
#                 try_cnt += 1

#     return ""


import random
import pickle
# from src.models.model_api import (
#     read_api_token, 
#     LanguageModel, 
#     TogetherAIModel, 
#     OpenAIModel, 
#     DebugModel,
#     TogetherAIGenerationParameters, 
#     OpenAIGenerationParameters, 
#     DebugGenerationParameters,
# )
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
    COMMON_SENSE = auto()
    COMMON_SENSE_CoT = auto()
    GSM8K = auto()
    MBPP = auto()
    RTE = auto()
    MNLI = auto()

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
class BatchPromptingDebugConfig:
    truncate_examples : bool = False,
    truncate_batch_queries : bool = False
    save_batched_model_inputs : Optional[Path] = None
    save_batched_model_outputs : Optional[Path] = None
@dataclass
class BatchPromptingExperimentConfig:
    # can either load a dataset from huggingface or from a local json file
    questions_dataset_config : DatasetConfig
    examples_dataset_config : DatasetConfig
    # can also choose to load a dataset from a file for questions or examples
    task_description: str
    # we specify baseline as a boolean as we might have a batch of size 1 at the end, but it will still
    # use the batched prompt template rather than the baseline prompt template
    k_shot: int
    batch_size: int
    example_selection: ExampleSelectionType 
    example_question_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    example_answer_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    # Note that this will include the api model name whereas model_name will be less formal like GPT-3.5 vs LLama-2-70B, etc.
    model_api: ModelAPIType
    generation_params: GENERATION_PARAMETERS_TYPE
    debug : Optional[BatchPromptingDebugConfig] = None
    pre_question_instructions: Optional[str] = None
    prompt_format: Optional[Callable[[Dict[str, str]], str]] = None
    is_baseline: bool = False
    random_seed: int = 0

# datasets = ['GSM8K', 'MBPP', 'glue-RTE', 'glue-MNLI']

class BatchPromptExperiment:
    def __init__(
            self,
            config: BatchPromptingExperimentConfig,
    ):
        self.config = config
        self.questions = self.load_dataset(self.config.questions_dataset_config)
        self.examples = self.load_dataset(self.config.examples_dataset_config)
        # self.examples = load_dataset(
        #     *self.config.hf_dataset_path,
        #     split=self.config.examples_split_name,
        # )
        # self.questions = load_dataset(
        #     *self.config.hf_dataset_path,
        #     split=self.config.questions_split_name,
        # )
        # must add an index column to gsm8k
        self.debug = self.config.debug
        self.batch_prompt_template = BatchPromptTemplate(
            examples=self.examples,
            dataset=self.config.questions_dataset_config.dataset,
            task_description=self.config.task_description,
            pre_question_instructions=self.config.pre_question_instructions,
            num_questions=self.config.batch_size,
            num_examples=self.config.k_shot,
            example_question_format=self.config.example_question_format,
            example_answer_format=self.config.example_answer_format,
            prompt_format=self.config.prompt_format,
            example_selection=self.config.example_selection,
            debug=self.debug,
            is_baseline=self.config.is_baseline,
        )
        self.model = self.load_language_model(
            model_api=self.config.model_api, 
            generation_params=self.config.generation_params
        )
        random.seed(self.config.random_seed)

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
        # add an index column to gsm8k
        if dataset_config.dataset == DatasetType.GSM8K:
            dataset = dataset.add_column('idx', list(range(len(dataset))))
        return dataset
    
    def load_language_model(self, model_api, generation_params) -> LanguageModel:
        match model_api:
            case ModelAPIType.OPEN_AI:
                token = read_api_token(Path("data/imported/open_ai_token.txt"))
                model = OpenAIModel(
                    api_token=token,
                    model_name=generation_params['model_name'],
                    generation_params=generation_params
                )
            case ModelAPIType.TOGETHER_AI:
                # together.ai
                token = read_api_token(Path("data/imported/together_ai_token.txt"))
                model = TogetherAIModel(
                    api_token=token,
                    model_name=generation_params['model_name'],
                    generation_params=generation_params
                )
            case ModelAPIType.DEBUG:
                model = DebugModel()
            # otherwise
            case _: 
                raise NotImplementedError("Only OpenAI and TogetherAI APIs are currently supported")
        # cover all bases
        return model

    # def batch_query_model(
    #     self,
    #     model_inputs: List[Tuple[List[ID_TYPE], str]], 
    # ) -> List[Tuple[List[ID_TYPE], str]]:

    #     # create the model API object if it doesn't exist
    #     if self.model is None:
    #         self.model = self.load_language_model()

    #     batched_results = [
    #         (ids, self.model.query(prompt))
    #         for (ids, prompt) in tqdm(model_inputs)
    #     ]
    #     return batched_results
    
    def batch_query_model(
        self, 
        model_inputs: List[Tuple[List[int], str]]
    ) -> List[Tuple[List[int], str]]:
        
        if self.model is None:
            self.model = self.load_language_model()

        messages = [[{"role": "user", "content": i[1]}] for i in model_inputs]
        attempt_cnt = 0
        max_attempts = 10 
        model_query_batch_size = 10
        results = []
        message_sublists = [messages[i:i+model_query_batch_size] for i in range(0, len(messages), model_query_batch_size)]
        for batched_messages in message_sublists:
            while attempt_cnt < max_attempts:
                try:
                    responses = batch_completion(
                    model=self.model.model_name,
                    messages = batched_messages,
                    **self.model.generation_params,
                    # temperature=0,
                    # max_tokens=None,
                    )
                    curr_results = [response["choices"][0]["message"]["content"] for response in responses]
                    results.extend(curr_results)
                except Exception as e:
                    attempt_cnt += 1
                    print(f"Error {str(e)} occurred. Retrying...")
                    time.sleep(120* attempt_cnt)
            curr_results = ["" for i in range(len(batched_messages))]
            results.extend(curr_results)
        return results


    def batch_questions(self) -> List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]]:

        batched_dataset : List[Dict[str, List[Any]]] = [   
            self.questions[i:i+self.config.batch_size]
            for i in range(0, len(self.questions), self.config.batch_size)
        ]
        if "CommonsenseQA" in self.config.task_description:
            idx = 0
            for batch in batched_dataset:
                batch['idx'] = [idx + i for i in range(len(batch['question']))]
                idx += len(batch['question'])
        batched_questions : List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]] = []
        for batch in batched_dataset:
            ids = batch[DATASET_ID_KEYS[self.config.questions_dataset_config.dataset][0]]
            questions = [
                {key : batch[key][i] for key in batch.keys()}
                for i in range(len(ids))
            ]
            batched_questions.append((ids, questions))

        return batched_questions
    
    def answers_from_batched_questions(
        self, 
        batched_questions: List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]]
    ) -> Dict[ID_TYPE, Dict[str, Any]]:
        answers :  Dict[ID_TYPE, Dict[str, Any]] = {
            question_id : question
            for (ids, questions) in batched_questions
            for (question_id, question) in zip(ids, questions)
        }
        return answers


    def execute(self) -> Tuple[List[Tuple[List[ID_TYPE], str]], Dict[ID_TYPE, Dict[str, Any]]]:
        """
        TODO:
        X Load Dataset
        X Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
        X query model for each input (save raw outputs to a file somewhere)
        """
        # splits self.questions into batches that are lists of individual dictionaries along with their ids
        batched_questions: List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]] = self.batch_questions()
        answers_dict = self.answers_from_batched_questions(batched_questions)

        # generate prompts for each batch
        batched_model_inputs : List[Tuple[List[ID_TYPE], str]] = [
            (ids, self.batch_prompt_template.generate_prompt(batch))
            for (ids, batch) in batched_questions
        ]
        # for debug purposes
        if self.debug:
            total_number_examples_wanted = 500
            needed_llm_calls = total_number_examples_wanted/self.config.batch_size
            if needed_llm_calls.is_integer():
                needed_llm_calls = int(needed_llm_calls)
            else:
                needed_llm_calls = math.ceil(needed_llm_calls)
            if needed_llm_calls == 0:
                needed_llm_calls = 1
            needed_llm_calls = min(needed_llm_calls, len(batched_model_inputs))
            batched_model_inputs = batched_model_inputs[:needed_llm_calls]
        # TODO: igure out how to also save the config
        # which includes a lambda/function that might be tricky to pickle
        # query model
        batched_model_outputs = self.batch_query_model(batched_model_inputs)
        if batched_model_inputs:
            if self.debug.save_batched_model_inputs:
                pickle.dump((batched_model_inputs), open(self.debug.save_batched_model_inputs, 'wb'))
            if self.debug.save_batched_model_outputs:
                pickle.dump((batched_model_outputs), open(self.debug.save_batched_model_outputs, 'wb'))


        return (batched_model_inputs, batched_model_outputs, answers_dict)
        # TODO: Alex, move this logic to a separate file
        # pred = []
        # for batched_output in batched_model_outputs:
        #     batched_output_parsed = extract_answers_batch(batched_output)
        #     assert len(batched_output_parsed) == len(batched_model_inputs)
        #     pred.extend(batched_output_parsed)
        # evaluator = Evaluation()
        # stats = evaluator.get_stats(y_pred=pred, y_true=ground_truth_answers, answer_type = answer_types[i])
        # # save the pickled batched model outputs to file
        # print("Dumping batched model outputs to file...")
        # pickle.dump((batched_model_outputs), open('batched_model_outputs.pkl', 'wb'))


def parse_answers(model_outputs: List[Tuple[List[ID_TYPE], str]]) -> Dict[List[ID_TYPE], str]:
    raise NotImplementedError()

class BatchPromptTemplate:
    # write the docs for this
    """
    BatchPromptTemplate is a class that generates prompts for a batch of examples.
    It is used to generate prompts for the k-shot experiment.
    Args:
    - Examples: a huggingface dataset of examples
    - task_description: a string describing the task - this goes at the top level of the prompt
    - num_questions: the number of questions to ask per prompt
    - num_examples: the number of examples to include in each prompt, > 1 means batched
    - example_template: a PromptTemplate that takes in a dataset's features dictionary and returns a string,
        it can also optionally take in an index [i] for batched prompts. This function will be used 
        both for building/retrieving from the example database and for generating prompts.
    - example_selection: an enum that specifies how to select examples for the prompt

    """
    def __init__(
            self,
            examples: Dataset,
            dataset: DatasetType,
            task_description: str,
            pre_question_instructions: str,
            num_questions: int,
            num_examples: int,
            is_baseline: bool,
            # the optional int is the index of the example in the batch, none if baseline
            example_question_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_answer_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_selection: ExampleSelectionType,
            debug: Optional[BatchPromptingDebugConfig] = None,
            prompt_format: Optional[Callable[[Dict[str, str]], str]] = None,
    ):
        self.examples = examples
        self.dataset = dataset
        self.task_description = task_description
        self.pre_question_instructions = pre_question_instructions
        self.num_questions = num_questions
        self.num_examples = num_examples
        self.example_question_format = example_question_format
        self.example_answer_format = example_answer_format
        self.prompt_format = prompt_format
        self.example_selection = example_selection
        self.is_baseline = is_baseline

        if self.is_baseline:
            assert(self.num_questions == 1)

        match self.example_selection:
            case ExampleSelectionType.RANDOM:
                pass
            case ExampleSelectionType.LEXICAL:
                raise NotImplementedError("Lexical example selection is not yet implemented")
            case ExampleSelectionType.SEMANTIC | ExampleSelectionType.MAX_MARGINAL_RELEVANCE:
                selector_class = {
                    ExampleSelectionType.SEMANTIC: SemanticSimilarityExampleSelector,
                    ExampleSelectionType.MAX_MARGINAL_RELEVANCE: MaxMarginalRelevanceExampleSelector,
                }[self.example_selection]
                print("Initializing Semantic Example Selector...")

                examples = list(self.examples)
                if debug:
                    if debug.truncate_batch_queries:
                        examples = examples[:150]

                self.example_selector = selector_class.from_examples(
                    # Need to turn hf dataset into a list of dicts
                    examples=examples,
                    # TODO: do we want embeddings to be configurable? probably not... it has sensible defaults
                    # and it is certainly not a menaingful variable in our experiments
                    embeddings=HuggingFaceEmbeddings(),
                    vectorstore_cls=Chroma,
                    k=self.num_examples,
                    input_keys=DATASET_INPUT_KEYS[self.dataset],
                )
                print("Done initializing Semantic Example Selector...")

    def get_examples(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # batch is a list of dataset instances that will have the same keys as the dataset
        match self.example_selection:
            case ExampleSelectionType.RANDOM:
                return [self.examples[i] for i in random.sample(range(len(self.examples)), self.num_examples)] 
            case ExampleSelectionType.LEXICAL:
                raise NotImplementedError("Lexical example selection is not yet implemented")
            case ExampleSelectionType.SEMANTIC | ExampleSelectionType.MAX_MARGINAL_RELEVANCE:
                top_k_examples_per_question = [
                    self.example_selector.select_examples(question)
                    for question in batch
                ]
                # num questions
                # note that we don't use self.num_questions because the batch could 
                # be smaller than that if it's the last batch
                b = len(top_k_examples_per_question)
                # take the first k examples, one from each question looping if necessary
                batch_examples = [
                    top_k_examples_per_question[i % b][i // b]
                    for i in range(self.num_examples)
                ]
                assert self.num_examples == len(batch_examples)
                return batch_examples

    # TODO: How to distinguish between baseline and batch size of 1 (baseline shouldn't have [i] in the prompt)
    # Resolved - ignore i in format function if baseline
    def generate_prompt(self, batch: List[Dict[str, Any]]) -> str:
        example_questions = []
        example_answers = []
        questions = []

        examples = self.get_examples(batch)
        for i, example in enumerate(examples):
            # the format functions take care of the Q[i] notation
            example_questions.append(self.example_question_format(example, i))
            example_answers.append(self.example_answer_format(example, i))
        
        for i, question in enumerate(batch):
            questions.append(self.example_question_format(question, i))
        
        '''TODO Rohan:  add an extra sentence so that the LLM knows the following are examplesdefault_shot_types = {
        "Zero-Shot": "",
        "Few-Shot": "Consider the following examples and maintain their formatting.\n",
        "One-Shot": "Consider the following example and maintain its formatting."
        TODO Rohan:  add an extra sentence so that the LLM knows the answers to the example questions are answers to the example questions ex: Response to examples in Batch for Few-Shot
        # TODO: Rohan add an extra sentence so that the LLM knows the following are the actual questions to answer: #Questions in Batch to answer
        ''' 
        

        if self.prompt_format is not None:
            fields = {
                'task_description' : self.task_description,
                'example_questions' : example_questions,
                'example_answers' : example_answers,
                'pre_questions_instructions' : self.pre_question_instructions,
                'questions' : questions,
            }
            prompt = self.prompt_format(fields)
        else:
            if self.is_baseline:
                examples = [
                    item 
                    for pair in zip(example_questions, example_answers) 
                    for item in pair
                ]
            else:
                examples = [
                    *example_questions,
                    *example_answers,
                ]
            # will be empty and provide nothing to the prompt if pre_questions_instructions is None
            # pre_questions_instructions = [self.pre_question_instructions] if self.pre_question_instructions is not None else []
            # example_str = ''.join([str(example) + '\n' for example in examples])
            # prompt = (
            #     "Task Description: " + self.task_description + "\n" +
            #     "Examples of batched questions and answers (Note: These are not the actual questions we want you to answer, but rather an example of what to expect as input, and examples of the desired format of answer):\n" +
            #     example_str +
            #     ''.join(pre_questions_instructions) + "\n" +
            #     "Actual Questions to answer:\n" + '\n'.join(questions)
            # )
            # prompt = (
                # for earlier template
                    # self.task_description.format(batch_size = self.num_questions) + '\n'.join(questions))
            # if examples == None or len(examples) == 0:
            prompt = (
                self.task_description.format(batch_size = self.num_questions) + '\n'.join(questions))
            # else:
            #     raise NotImplementedError()


            # prompt = "\n".join(
            #     [
            #         self.task_description,
            #         *examples,
            #         *pre_questions_instructions,
            #         *questions,
            #     ]
            # )
        return prompt



# if __name__ == "__main__":
    
#     # from data.parsing_functions import *
#     example_question_format = lambda example, i: f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"
#     example_answer_format = lambda example, i: f"Answer[{i}]: {example['label']}"
#     example_question_format_baseline = lambda example, i: f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}"
#     example_answer_format_baseline = lambda example, i: f"Answer: {example['label']}"

#     oai_gen_params = OpenAIGenerationParameters(
#             model_name='gpt-3.5-turbo',
#             temperature=0.6,
#             max_tokens=64,
#             frequency_penalty=1.0,
#         )




#     questions_config_rte = DatasetConfig(
#         dataset=DatasetType.RTE,
#         hf_dataset_path=['glue', 'rte'],
#         split_name='validation',
#     )
#     examples_config_rte = DatasetConfig(
#         dataset=DatasetType.RTE,
#         hf_dataset_path=['glue', 'rte'],
#         split_name='train',
#     )
#     task_description_rte = 'Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.'
#     # example_question_format_rte = lambda example, i: f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"
#     # example_answer_format_rte = lambda example, i: f"A[{i}]: {example['label']}"

#     # TODO: Alex, move these configs to a separate file
#     # questions_config_GSM8K = DatasetConfig(
#     #     dataset=DatasetType.GSM8K,
#     #     hf_dataset_path=['gsm8k', 'main'],
#     #     split_name='test',
#     # )
#     # examples_config_GSM8K = DatasetConfig(
#     #     dataset=DatasetType.GSM8K,
#     #     hf_dataset_path=['gsm8k', 'main'],
#     #     split_name='train',
#     # )
#     # task_description_GSM8K = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''

#     # '''
#     # # TODO: Rohan: Can you split reasoning-machines/gsm-hard[train] into a train test split?  
#     # We only have train in gsm-hard so we need to split both. The following below is commented out because sampling is done from the same place.
#     # '''
#     # questions_config_GSM8K_HARD = DatasetConfig(
#     #     dataset=DatasetType.GSM8K_HARD,
#     #     hf_dataset_path=["reasoning-machines/gsm-hard"],
#     #     split_name='train',
#     # )
#     # examples_config_GSM8K_HARD = DatasetConfig(
#     #     dataset=DatasetType.GSM8K_HARD,
#     #     hf_dataset_path=["reasoning-machines/gsm-hard"],
#     #     split_name='train',
#     # )
#     # task_description_GSM8K_HARD = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''



#     # questions_config_MBPP = DatasetConfig(
#     #     dataset=DatasetType.MBPP,
#     #     hf_dataset_path=['mbpp'],
#     #     split_name='validation',
#     # )
#     # examples_config_MBPP = DatasetConfig(
#     #     dataset=DatasetType.MBPP,
#     #     hf_dataset_path=['mbpp'],
#     #     split_name='train',
#     # )
#     # task_description_MBPP = '''You are tasked with solving Python programming problems that are designed to be solvable by entry-level programmers. Each problem will consist of a task description, and your job is to output a string that when parsed is an executable Python code function that fulfills the requirements of the task. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format.'''



#     # questions_config_MNLI = DatasetConfig(
#     #     dataset=DatasetType.MNLI,
#     #     hf_dataset_path=['glue', 'mnli'],
#     #     split_name='validation_matched',
#     # )
#     # examples_config_MNLI = DatasetConfig(
#     #     dataset=DatasetType.MNLI,
#     #     hf_dataset_path=['glue', 'mnli'],
#     #     split_name='train',
#     # )
#     # task_description_MNLI = '''You are tasked with the job of Multi-Genre Natural Language Inference (MNLI). For each task, you will be given a premise sentence and a hypothesis sentence. Your job is to predict the relationship between the premise and the hypothesis, classifying each pair as either 'entailment', 'contradiction', or 'neutral'. Instruction: For each question in the batch, provide a single answer, following the format A[idx]: answer. Output only the answers with the associated index in "A[idx]: answer" format. Each answer should be only one of the following: 'entailment', 'contradiction', or 'neutral'. So in other words, for each question, you should output one of the following: A[idx]: entailment, A[idx]: contradiction, or A[idx]: neutral.'''



#     # questions_config_COMMON_SENSE = DatasetConfig(
#     #     dataset=DatasetType.COMMON_SENSE,
#     #     hf_dataset_path=['commonsense_qa'],
#     #     split_name='validation',
#     # )
#     # examples_config_COMMON_SENSE = DatasetConfig(
#     #     dataset=DatasetType.COMMON_SENSE,
#     #     hf_dataset_path=['commonsense_qa'],
#     #     split_name='train',
#     # )
#     # task_description_COMMON_SENSE = '''You are tasked with answering multiple-choice questions that require both contextual understanding and general world knowledge. Each question will have five options labeled 'a', 'b', 'c', 'd', and 'e'. Your job is to select the most appropriate answer by outputting the letter corresponding to that option. " These questions are part of the CommonsenseQA dataset, designed to test your ability to answer questions that often require prior knowledge. Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format. '''






#     # config_param_list = [
#     #     [questions_config_rte, examples_config_rte, task_description_rte, rte_question_format, rte_answer_format],
#     #     [questions_config_GSM8K, examples_config_GSM8K, task_description_GSM8K, gsm8k_question_format, gsm8k_answer_format],
#     #     # [questions_config_MBPP, examples_config_MBPP, task_description_MBPP, mbpp_question_format, mbpp_answer_format],
#     #     [questions_config_MNLI, examples_config_MNLI, task_description_MNLI, mnli_question_format, mnli_answer_format],
#     #     # [questions_config_GSM8K_HARD, examples_config_GSM8K_HARD, task_description_GSM8K_HARD, gsm8k_question_format, gsm8k_answer_format],
#     #     # [questions_config_COMMON_SENSE, examples_config_COMMON_SENSE, task_description_COMMON_SENSE, commonsense_question_format, commonsense_answer_format] 
#     #     ]

#     # stats = []
#     # for questions_config, examples_config, task_description in config_param_list:
#     config = BatchPromptingExperimentConfig(
#         questions_dataset_config=questions_config_rte,
#         examples_dataset_config=examples_config_rte,
#         task_description='Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.\n',
#         pre_question_instructions="Consider the following examples and maintain their formatting.\n",
#         k_shot=7,
#         example_selection=ExampleSelectionType.RANDOM,
#         example_question_format=example_question_format_baseline,
#         example_answer_format=example_answer_format_baseline,
#         batch_size=1,
#         model_api=ModelAPIType.OPEN_AI,
#         generation_params=oai_gen_params,
#         random_seed=0,
#         is_baseline=True,
#         debug=BatchPromptingDebugConfig(
#             truncate_examples=True,
#             truncate_batch_queries=True,
#             save_batched_model_inputs=Path('batched_model_inputs.pkl'),
#             save_batched_model_outputs=Path('batched_model_outputs.pkl'),
#         ),
#     )
#     experiment = BatchPromptExperiment(config)
#     experiment.execute()
# # from src.utils.parsing_functions import extract_answers_batch
















# from src.experiments.k_shot_experiment import *
from src.utils.parsing_functions import * 

oai_gen_params = OpenAIGenerationParameters(
            model_name='gpt-3.5-turbo',
            temperature=0.6,
            max_tokens=64,
            frequency_penalty=1.0,
        )

questions_config_rte = DatasetConfig(
    dataset=DatasetType.RTE,
    hf_dataset_path=['glue', 'rte'],
    split_name='validation',
)
examples_config_rte = DatasetConfig(
    dataset=DatasetType.RTE,
    hf_dataset_path=['glue', 'rte'],
    split_name='train',
)

# THIS prompt does not work for llama2 but is modified to try to work for it
# task_description_rte = """\
# **Objective**: Your task is to solve a batch of Recognizing Textual Entailment (RTE) questions. You will be given {batch_size} sentence pairs from the Textual Entailment Recognition dataset as input. Classify each sentence pair into one of two classes: 0 for entailment and 1 for non-entailment.

# **Method**: Use NLP techniques to logically evaluate the relationship between each sentence pair.

# #### Instructions:

# 1. **Intermediate Reasoning**: Clearly state the logical steps you took to arrive at each answer.
# 2. **Batch Size**: Provide exactly {batch_size} answers, one for each question in the batch.

# #### Input Format:

# Questions will be batched and each will include a "Premise" and a "Hypothesis", prefixed with an index starting from 0:
# P[0]: {{{{Premise_0_Text}}}}
# H[0]: {{{{Hypothesis_0_Text}}}}
# ...
# P[{batch_size} - 1]: {{{{Premise_{batch_size}_1_Text}}}}
# H[{batch_size} - 1]: {{{{Hypothesis_{batch_size}_1_Text}}}}

# #### Output Format:

# For each question, your answer should strictly follow this format:
# A[index]: {{{{Intermediate_Reasoning}}}}; The answer is {{{{Answer_Integer}}}}

# - `index`: The index of the question, prefixed with 'A' and enclosed in square brackets.
# - {{{{Intermediate_Reasoning}}}}: The logical steps leading to the answer.
# - {{{{Answer_Integer}}}}: The final integer answer, representing the classification of the sentence pair.

# Do not include any additional information or questions in your answers. Stick strictly to answering the questions provided in the specified format.

# Batched Questions to Answer:

# """
# NOTE I BELIEVE THIS COULD BE THE GOOD ONE FOR GPT ETC
task_description_rte = '''**Objective**: Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes. You must answer all questions in the batch. The total number of questions in the batch is defined as batch_size = {batch_size}.

An answer of 0 means that the given Hypothesis and Premise logically entail each other. 
An answer of 1 means the given Hypothesis and Premise do NOT entail each other.

**Method**: Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.

#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

#### Input Format:
- Questions will be presented in a batch. Each question will include a sentence pair labeled as "Premise" and "Hypothesis" and will be prefixed with its index, starting from 0, like so:
P[0]: {{Premise_0_Text}}
H[0]: {{Hypothesis_0_Text}}
...
P[{{batch_size - 1}}]: {{Premise_{{batch_size - 1}}_Text}}
H[{{batch_size - 1}}]: {{Hypothesis_{{batch_size - 1}}_Text}}

#### Output Format:
- You must adhere to the following format rigorously for each answer:
A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
- `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
- `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
- `{{Answer_Integer}}`: This is the final integer answer to the question, representing the class into which the sentence pair falls.

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

Batched Questions to Answer:
'''

# task_description_COMMON_SENSE = '''You are tasked with answering multiple-choice questions that require both contextual understanding and general world knowledge. Each question will have five options labeled 'a', 'b', 'c', 'd', and 'e'. Your job is to select the most appropriate answer by outputting the letter corresponding to that option. " These questions are part of the CommonsenseQA dataset, designed to test your ability to answer questions that often require prior knowledge. Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format. '''
task_description_COMMON_SENSE = '''
### **Objective**: Your task is to solve a set of multiple-choice questions from the CommonsenseQA dataset in a batch. CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . You will be given `{{batch_size}}` questions each time, as input. These questions are designed to test your ability to answer queries that often require contextual understanding and general world knowledge. Your goal is to select the letter corresponding to the most appropriate answer among five options labeled 'a', 'b', 'c', 'd', and 'e' for each question in the batch. You must answer all questions in the batch. The total number of questions in the batch is defined as `batch_size = {{batch_size}}`.

An answer of 'a' means you believe option 'a' is the most appropriate answer.
An answer of 'b' means you believe option 'b' is the most appropriate answer.
An answer of 'c' means you believe option 'c' is the most appropriate answer.
An answer of 'd' means you believe option 'd' is the most appropriate answer.
An answer of 'e' means you believe option 'e' is the most appropriate answer.

**Method**: Use your expertise in NLP and contextual understanding to perform a sequence of logical evaluations for solving these questions.

#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to arrive at your answer. This could include identifying key phrases, contradictions, or logical connections that led you to choose a particular option.

2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{{batch_size}}`.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

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
- `{{Answer_Integer}}`: This is the final letter answer to each question.

The phrase 'The answer is' must directly precede each letter answer and come after the intermediate reasoning, separated by a semicolon. Ensure you output A[index] for each question before outputting {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

Batched Questions to Answer:
'''


task_description_MNLI = '''### **Objective**: Your task is to solve a set of MultiNLI (MNLI) questions in a batch. You will be given `{{batch_size}}` premise-hypothesis pairs from the MNLI dataset as input. Your goal is to classify each pair into one of three classes: entailment, neutral, or contradiction. You must answer all questions in the batch. The total number of questions in the batch is defined as `batch_size = {{batch_size}}`.

An answer of 0 means the premise entails the hypothesis, indicating that if the premise is true, the hypothesis must also be true. In this case, the information in the hypothesis is a logical subset of the information in the premise.
An answer of 1 means the relationship between the premise and the hypothesis is neutral, suggesting that the truth of the premise neither guarantees nor contradicts the truth of the hypothesis. The hypothesis could be either true or false regardless of the premise's truth value.
An answer of 2 means the premise contradicts the hypothesis, implying that both cannot be true at the same time. If the premise is true, the hypothesis must necessarily be false, and vice versa.

**Method**: Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations relationship between each Premise and Hypothesis pair. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).

#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

#### Input Format:
- Questions will be presented in a batch. Each question will include a sentence pair labeled as "Premise" and "Hypothesis" and will be prefixed with its index, starting from 0, like so:
P[0]: {{Premise_0_Text}}
H[0]: {{Hypothesis_0_Text}}
...
P[{{batch_size - 1}}]: {{Premise_{{batch_size - 1}}_Text}}
H[{{batch_size - 1}}]: {{Hypothesis_{{batch_size - 1}}_Text}}

#### Output Format:
- You must adhere to the following format rigorously for each answer:
A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
- `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
- `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
- `{{Answer_Integer}}`: This is the final integer answer to the question, representing the class into which the sentence pair falls.

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

Batched Questions to Answer:
'''

# task_description_mnli = '''**Objective**: Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes. You must answer all questions in the batch. The total number of questions in the batch is defined as batch_size = {batch_size}.

# An answer of 0 means that the given Hypothesis and Premise logically entail each other. 
# An answer of 1 means the given Hypothesis and Premise do NOT entail each other.

# **Method**: Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.

# #### Instructions:

# 1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

# 2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

# 3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

# #### Input Format:
# - Questions will be presented in a batch. Each question will include a sentence pair labeled as "Premise" and "Hypothesis" and will be prefixed with its index, starting from 0, like so:
# P[0]: {{Premise_0_Text}}
# H[0]: {{Hypothesis_0_Text}}
# ...
# P[{{batch_size - 1}}]: {{Premise_{{batch_size - 1}}_Text}}
# H[{{batch_size - 1}}]: {{Hypothesis_{{batch_size - 1}}_Text}}

# #### Output Format:
# - You must adhere to the following format rigorously for each answer:
# A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
# - `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
# - `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
# - `{{Answer_Integer}}`: This is the final integer answer to the question, representing the class into which the sentence pair falls.

# The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

# Batched Questions to Answer:
# '''

# task_description_rte = '''**Objective**: Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify the sentence pair into two classes. The total number of questions in the batch is defined as batch_size = {batch_size}.

# An answer of 0 means that the given Hypothesis and Premise are logical and following (entailment) to each other. 
# An answer of 1 the given Hypothesis and Premise are NOT following (entailment) to each other.

# **Method**: Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.

# #### Instructions:

# 1. **Intermediate Reasoning**: Include all the intermediate steps you took to arrive at each answer. This ensures that the answer is both well-reasoned and reliable.

# 2. **Batch Size**: The number of answers you provide must exactly match the specified `{batch_size}`.

# #### Input Format:

# - Questions will be presented in a batch. Each question will include a sentence pair labeled as "Premise" and "Hypothesis" and will be prefixed with its index, starting from 0, like so:
# P[0]: {{Premise_0_Text}}
# H[0]: {{Hypothesis_0_Text}}
# ...
# P[{{batch_size - 1}}]: {{Premise_{{batch_size - 1}}_Text}}
# H[{{batch_size - 1}}]: {{Hypothesis_{{batch_size - 1}}_Text}}

# #### Output Format:

# - You must adhere to the following format rigorously for each answer:
# A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
# - `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
# - `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
# - `{{Answer_Integer}}`: This is the final integer answer to the question, representing the class into which the sentence pair falls.

# The phrase 'The answer is' must directly precede the integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the output is in the desired format. Answer all question, ensuring that {batch_size} answers are provided in our desired format.

# Batched Questions to Answer:

# ''' 

questions_config_GSM8K = DatasetConfig(
    dataset=DatasetType.GSM8K,
    hf_dataset_path=['gsm8k', 'main'],
    split_name='test',
)
examples_config_GSM8K = DatasetConfig(
    dataset=DatasetType.GSM8K,
    hf_dataset_path=['gsm8k', 'main'],
    split_name='train',
)
# task_description_GSM8K = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Think algorithmically and try to break down each question into parts that can be combined to get your final answer. Output only the answers with the associated index in A[idx]: answer format.\n'''

# '''**Objective**: Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes. You must answer all questions in the batch. The total number of questions in the batch is defined as batch_size = {batch_size}.

# An answer of 0 means that the given Hypothesis and Premise logically entail each other. 
# An answer of 1 means the given Hypothesis and Premise do NOT entail each other.

# **Method**: Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.

# #### Instructions:

# 1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

# 2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

# 3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

# #### Input Format:
# - Questions will be presented in a batch. Each question will include a sentence pair labeled as "Premise" and "Hypothesis" and will be prefixed with its index, starting from 0, like so:
# P[0]: {{Premise_0_Text}}
# H[0]: {{Hypothesis_0_Text}}
# ...
# P[{{batch_size - 1}}]: {{Premise_{{batch_size - 1}}_Text}}
# H[{{batch_size - 1}}]: {{Hypothesis_{{batch_size - 1}}_Text}}

# #### Output Format:
# - You must adhere to the following format rigorously for each answer:
# A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
# - `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
# - `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
# - `{{Answer_Integer}}`: This is the final integer answer to the question, representing the class into which the sentence pair falls.

# The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

# Batched Questions to Answer:
# '''

task_description_GSM8K = '''**Objective**: Your task is to solve a set of math questions in a batch. You will be given {{batch_size}} questions from the GSM8K dataset as input. Your goal is to answer each question.  You must answer all questions in the batch. The total number of questions in the batch is defined as batch_size = {batch_size}.

**Complexity**: Each question in the batch will require you to perform between 2 and 8 steps to arrive at the final answer.

**Method**: Use basic arithmetic operations to perform a sequence of calculations for solving these questions.

#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

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

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Ensure you output A[index] for each question before outputting {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

Batched Questions to Answer:
'''



# '''**Objective**: Your task is to solve a set of math questions in a batch. The total number of questions in the batch is defined as `{batch_size}`.

# **Complexity**: Each question in the batch will require you to perform between 2 and 8 steps to arrive at the final answer.

# **Method**: Use basic arithmetic operations to perform a sequence of calculations for solving these questions.

# #### Instructions:

# 1. **Intermediate Reasoning**: Include all the intermediate steps you took to arrive at each answer. This ensures that the answer is both well-reasoned and reliable.

# 2. **Batch Size**: The number of answers you provide must exactly match the specified `{batch_size}`.

# #### Input Format:

# - Questions will be presented in a batch. Each question will be prefixed with its index, starting from 0, like so:
# Q[0]: {Question_0_Text}
# Q[1]: {Question_1_Text}
# ...
# Q[{batch_size - 1}]: {Question_{batch_size - 1}_Text}
# #### Output Format:

# - You must adhere to the following format rigorously for each answer:
# A[index]: {Intermediate_Reasoning}; The answer is {Answer_Integer}
# - `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
# - `{Intermediate_Reasoning}`: This is where you provide all the intermediate steps that led you to the final answer.
# - `{Answer_Integer}`: This is the final integer answer to the question.

# The phrase 'The answer is' must directly precede the integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the output is in the desired format.

# Batched Questions to Answer:
# '''
questions_config_GSM8K_HARD = DatasetConfig(
    dataset=DatasetType.GSM8K_HARD,
    hf_dataset_path=["reasoning-machines/gsm-hard"],
    split_name='train',
)
examples_config_GSM8K_HARD = DatasetConfig(
    dataset=DatasetType.GSM8K_HARD,
    hf_dataset_path=["reasoning-machines/gsm-hard"],
    split_name='train',
)
task_description_GSM8K_HARD = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''



questions_config_MBPP = DatasetConfig(
    dataset=DatasetType.MBPP,
    hf_dataset_path=['mbpp'],
    split_name='validation',
)
examples_config_MBPP = DatasetConfig(
    dataset=DatasetType.MBPP,
    hf_dataset_path=['mbpp'],
    split_name='train',
)
task_description_MBPP = '''You are tasked with solving Python programming problems that are designed to be solvable by entry-level programmers. Each problem will consist of a task description, and your job is to output a string that when parsed is an executable Python code function that fulfills the requirements of the task. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format.'''



questions_config_MNLI = DatasetConfig(
    dataset=DatasetType.MNLI,
    hf_dataset_path=['glue', 'mnli'],
    split_name='validation_matched',
)
examples_config_MNLI = DatasetConfig(
    dataset=DatasetType.MNLI,
    hf_dataset_path=['glue', 'mnli'],
    split_name='train',
)
# TODO: Either change the promtpt so that it outputs numbers, OR change the parse so that it extracts the three categorical labels.
# task_description_MNLI = '''You are tasked with the job of Multi-Genre Natural Language Inference (MNLI). For each task, you will be given a premise sentence and a hypothesis sentence. Your job is to predict the relationship between the premise and the hypothesis, classifying each pair as either 'entailment', 'contradiction', or 'neutral'. Instruction: For each question in the batch, provide a single answer, following the format A[idx]: answer. Output only the answers with the associated index in "A[idx]: answer" format. Each answer should be only one of the following: 'entailment', 'contradiction', or 'neutral'. So in other words, for each question, you should output one of the following: A[idx]: entailment, A[idx]: contradiction, or A[idx]: neutral.'''



questions_config_COMMON_SENSE = DatasetConfig(
    dataset=DatasetType.COMMON_SENSE,
    hf_dataset_path=['commonsense_qa'],
    split_name='validation',
)
examples_config_COMMON_SENSE = DatasetConfig(
    dataset=DatasetType.COMMON_SENSE,
    hf_dataset_path=['commonsense_qa'],
    split_name='train',
)




config_to_answer_type = {"GSM8K": "numerical", 
                "GSM8K_HARD": "numerical", 
                "COMMON_SENSE": "categorical", 
                "MBPP": "code",
                "MNLI": "binary",
                "RTE": "categorical"}
config_param_list = { 
    "rte": [questions_config_rte, examples_config_rte, task_description_rte, rte_question_format, rte_answer_format],
    "GSM8K": [questions_config_GSM8K, examples_config_GSM8K, task_description_GSM8K, gsm8k_question_format, gsm8k_answer_format],
    # # "MBPP": [questions_config_MBPP, examples_config_MBPP, task_description_MBPP, mbpp_question_format, mbpp_answer_format],
    "MNLI": [questions_config_MNLI, examples_config_MNLI, task_description_MNLI, mnli_question_format, mnli_answer_format],
    # #"GSM8K_HARD": [questions_config_GSM8K_HARD, examples_config_GSM8K_HARD, task_description_GSM8K_HARD, gsm8k_question_format, gsm8k_answer_format],
    "COMMON_SENSE": [questions_config_COMMON_SENSE, examples_config_COMMON_SENSE, task_description_COMMON_SENSE, commonsense_question_format, commonsense_answer_format] 
}

def extract_math_answers(text):
    # Initialize an empty list to store the extracted numbers along with their positions
    extracted_numbers = []
    
    # Regex pattern with a fallback capturing group (.+?) to capture any string that appears after "The answer is"
    # pattern = r"The answer is\s*(-?\$?|\$?-?)([\d,.-]+)|The answer is\s*(.+?)\."
    
    pattern = r'The answer is\s*(-?)\$?(\d{1,3}(?:,\d{3})*)'

    # Find all matches of the pattern in the text using finditer
    matches = re.finditer(pattern, text)
    
    # Extract numbers and their positions from the matches
    for match in matches:
        position = match.start()
        
        # Check if the match is a parsable number
        if match.group(2):
            try:
                # Determine if the number is negative based on the presence of the negative sign in the first capturing group
                is_negative = '-' in match.group(1)
                
                # Remove commas and convert to integer
                num = int(match.group(2).replace(',', ''))
                
                # Apply the negative sign if needed
                if is_negative:
                    num = -num
                
                extracted_numbers.append((position, num))
            except ValueError:
                extracted_numbers.append((position, None))
        else:
            # If the match is not a parsable number, insert None
            extracted_numbers.append((position, None))
    
    # Sort the extracted numbers based on their positions in the text
    extracted_numbers.sort(key=lambda x: x[0])
    
    # Return only the numbers or None placeholders, in the order they appeared in the text
    return [num for position, num in extracted_numbers]


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
    elif answer_type in ["GSM8K_HARD", "GSM8K"]:
        answers = extract_math_answers(output_str)
        return answers
        # # Step 1: Split the input string into a list of substrings based on "A[idx]: " pattern
        # split_string = re.split(r'A\[\d+\]: ', output_str)[1:]  # Skip the first empty string if any
        
        # # Initialize the list to store the last integer from each substring
        # last_numbers = []
        
        # for substring in split_string:
        #     # Step 2: Find all integers in the substring
        #     # Note: We use four different patterns to capture integers:
        #     # 1. Regular integers (\d+)
        #     # 2. Integers with commas (\d{1,3}(,\d{3})*)
        #     # 3. Integers with a preceding dollar sign (\$\d+|\$\d{1,3}(,\d{3})*)
        #     # 4. Negative integers (-\d+)
        #     numbers = re.findall(r'\d+|\d{1,3}(,\d{3})*|\$\d+|\$\d{1,3}(,\d{3})*|-\d+', substring)
            
        #     # Convert found numbers into integers
        #     int_numbers = []
        #     for num in numbers:
        #         # Remove commas and dollar signs, if any
        #         cleaned_num = num.replace(',', '').replace('$', '')
        #         # Convert to integer
        #         int_num = int(cleaned_num)
        #         int_numbers.append(int_num)
            
        #     # Step 3: Append the last integer to the list
        #     if int_numbers:
        #         last_numbers.append(int_numbers[-1])
                
        # return last_numbers
    elif answer_type in ["RTE", "MNLI"]:
        answers = []
        
        # Split the string by newlines to process each line individually.
        lines = output_str.strip().split("\n")
        
        # General regex pattern to extract a potential answer.
        # general_pattern = r": (\d+)|(\d+)"
        
        # Loop through each line to extract the answer.
        for line in lines:
            if line =='':
                continue
            answer = extract_last_number(line)
            if answer == None and "Q[" in line:
                continue
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

def split_answers(text):
    # Initialize an empty list to store the individual answers
    answer_list = []
    
    idx = 0
    while True:
        # Dynamic regular expression pattern to match "A[idx]" or "[Aidx]"
        pattern = r"(?:A\[" + str(idx) + r"\]|\[A" + str(idx) + r"\])"
        
        # Find all matches of the pattern in the text using finditer
        matches = [m for m in re.finditer(pattern, text)]
        
        # If no matches are found, break the loop
        if not matches:
            break
        
        # Get the first match for the current index
        first_match = matches[0]
        
        # Dynamic regular expression pattern to match "A[idx+1]" or "[Aidx+1]"
        next_pattern = r"(?:A\[" + str(idx + 1) + r"\]|\[A" + str(idx + 1) + r"\])"
        
        # Find all matches of the next pattern in the text using finditer
        next_matches = [m for m in re.finditer(next_pattern, text)]
        
        # If matches for the next index are found, get the last match
        if next_matches:
            next_match = next_matches[0]
            # Slice the text from the first match's start position to the next match's start position
            answer_list.append(text[first_match.start():next_match.start()])
        else:
            # If no next match is found, slice from the first match's start position to the end of the text
            answer_list.append(text[first_match.start():])
        
        # Increment the index for the next iteration
        idx += 1
    
    return answer_list
# def split_answers(text):
#     # Initialize an empty list to store the individual answers
#     answer_list = []
    
#     # Regular expression pattern to match "A[idx]" where "idx" is any number
#     pattern = r"(?:A\[\d+\]|\[A\d+\])" # r"A\[\d+\]"
    
#     # Find all matches of the pattern in the text using finditer
#     matches = re.finditer(pattern, text)
    
#     # Initialize a variable to store the start position of the previous match
#     prev_start = None
    
#     for match in matches:
#         # If this is not the first match, slice the text from the previous start position to the current start position
#         if prev_start is not None:
#             answer_list.append(text[prev_start:match.start()])
        
#         # Update the previous start position to the current start position
#         prev_start = match.start()
    
#     # Append the last answer block from the last start position to the end of the text
#     if prev_start is not None:
#         answer_list.append(text[prev_start:])
    
#     return answer_list    
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
            index_to_ground_truth[answer_index] = answer['answerKey']
        elif task_name in ["RTE", "MNLI"]:
            index_to_ground_truth[answer_index] = int(answer["label"])
        else:
            raise ValueError("Task name not recognized.")
    return index_to_ground_truth

def extract_last_integer(text):
    # Define the regex pattern to match numbers with optional negative sign, commas, and dollar sign
    pattern = r'he answer is ([-]?\$?\d{1,3}(?:,\d{3})*)'
    
    # Use re.findall() to find all occurrences of the pattern
    matches = re.findall(pattern, text)
    
    if not matches:
        pattern = r'nswer is ([-]?\$?\d{1,3}(?:,\d{3})*)'
        # Use re.findall() to find all occurrences of the pattern
        matches = re.findall(pattern, text)
        if not matches:
            return None
    
    # Take the last occurrence
    last_match = matches[-1]
    
    # Remove commas and dollar signs, if any
    cleaned_match = last_match.replace(',', '').replace('$', '')
    
    # Convert to integer
    last_integer = int(cleaned_match)
    
    return last_integer

def extract_last_letter(text):
    # Define the regex pattern to match 'The answer is ' followed by a single letter (A-E or a-e)
    # The \b ensures that the letter is a word boundary, so nothing comes immediately after it.
    pattern = r'he answer is .*?([A-Ea-e])\b'
    
    # Use re.findall() to find all occurrences of the pattern
    matches = re.findall(pattern, text)
    
    if not matches:
        pattern = r'nswer is .*?([A-Ea-e])\b'
        # Use re.findall() to find all occurrences of the pattern
        matches = re.findall(pattern, text)
        if not matches:
            return None
    
    # Take the last occurrence
    last_match = matches[-1]
    
    # Convert to uppercase if it's not
    last_match = last_match.upper()
    
    return last_match
def get_index_to_pred(batched_model_inputs, batched_model_outputs, task_name):
    index_to_pred = {}
    for batch_input, batch in zip(batched_model_inputs, batched_model_outputs):
        indices = batch_input[0]
        LLM_output = batch
        
        text_split_by_batch = split_answers(LLM_output)
        if task_name == "COMMON_SENSE":
            answers = [extract_last_letter(text) for text in text_split_by_batch]
        else: 
            answers = [extract_last_integer(text) for text in text_split_by_batch]
        # answers = extract_answers_batch(LLM_output, task_name)
        # answers = parse_batched_answers(LLM_output, task_name)
        if len(answers) == len(indices):
            for index, answer in zip(indices, answers):
                index_to_pred[index] = answer
        elif len(answers) > len(indices):
            for index, answer in zip(indices, answers[0:len(indices)]):
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



def run_batched_tests(config, config_to_answer_type):
    task_to_stats ={}
    batch_sizes = [1, 2, 4, 8, 16]

    # batch_sizes = [1, 2, 4, 8]
    # batch_sizes = [4, 8, 16, 32]

    os.environ['OPENAI_API_KEY'] = read_api_token(Path("data/imported/open_ai_token.txt"))
    for task_name, configs in config_param_list.items():
        # if task_name.upper() == "RTE":
        #     continue
        # if task_name != "MNLI":
        #     continue
        for batch_size in batch_sizes:
            # llama_2_70B_gen_params = TogetherAIGenerationParameters(
            # model_name='togethercomputer/llama-2-70b-chat',
            # temperature = 0,
            # # temperature=0.6,
            # # max_tokens=64,
            # max_tokens=None,
            # frequency_penalty=1.0,
            # )
            oai_gen_params = OpenAIGenerationParameters(
                    model_name='gpt-3.5-turbo-16k',
                    # model_name='gpt-4',
                    temperature=0,
                    max_tokens=None,
                    frequency_penalty=1.0,
                )
            questions_config, examples_config, task_description, question_format, answer_format = configs

            config = BatchPromptingExperimentConfig(
            questions_dataset_config=questions_config,
            examples_dataset_config=examples_config,
            task_description=task_description,
            pre_question_instructions="Consider the following examples and maintain their formatting.\n",
            k_shot=1,
            example_selection=ExampleSelectionType.RANDOM,
            # example_selection=None,

            example_question_format=question_format,
            example_answer_format=answer_format,
            batch_size=batch_size,
            model_api=ModelAPIType.OPEN_AI,
            # model_api=ModelAPIType.TOGETHER_AI,
            generation_params=oai_gen_params,

            # generation_params=llama_2_70B_gen_params,
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
            if task_name not in task_to_stats:
                task_to_stats[task_name] = {}
            if batch_size not in task_to_stats[task_name]:
                task_to_stats[task_name][batch_size] = {}
            task_to_stats[task_name][batch_size]["batched_model_outputs"] = batched_model_outputs
            task_to_stats[task_name][batch_size]["stat"] = stat
            task_to_stats[task_name][batch_size]["pred"] = pred
            task_to_stats[task_name][batch_size]["ground_truth"] = ground_truth
    with open("task_to_stats_rte_gsm8k_mnli_commonsense", 'wb') as f:
        pickle.dump(task_to_stats, f)
    return task_to_stats

if __name__ == "__main__":
    run_batched_tests(config_param_list, config_to_answer_type)


multi_task_prompt = '''

Objective: Your task is to solve a variety of questions across multiple domains in a batch. You will be given {{batch_size}} questions, each associated with a specific task domain: Your goal is to answer each question according to its domain in the desired output format. The total number of questions in the batch to answer is defined as batch_size = {batch_size}.

Task Tokens and Descriptions:

RTE: Instruction - Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes.
RTE: Method - Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.
RTE: Intermediate Reasoning - Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.
RTE: Input Format - Each question will  be prefixed with its index and followed by a sentence pair labeled as "Premise" and "Hypothesis" starting from 0, like so:
Q[0][RTE]
P: {{Premise_0_Text}}
H: {{Hypothesis_0_Text}}
...
Q[{{batch_size - 1}}][RTE]
P: {{Premise_{{batch_size - 1}}_Text}}
H: {{Hypothesis_{{batch_size - 1}}_Text}}
RTE: Output Meaning - An answer of 0 means that the given Hypothesis and Premise logically entail each other.  An answer of 1 means the given Hypothesis and Premise do NOT entail each other.


GSM8K: Instruction - Your task is to solve a set of math questions in a batch.
GSM8K: Method - Use basic arithmetic operations to perform a sequence of calculations for solving these questions.
GSM8K: Intermediate Reasoning -  Each question in the batch will require you to perform between 2 and 8 steps to arrive at the final answer.
GSM8K: Input Format -  Each question will be prefixed with its index, starting from 0, like so:
Q[0][GSM8K]: {{Question_0_Text}}
Q[1][GSM8K]: {{Question_1_Text}}
...
Q[{{batch_size - 1}}][GSM8K]: {{Question_{{batch_size - 1}}_Text}}
GSM8K: Output Meaning - Each answer is an integer that is the answer to the question.

MNLI: Judge the relationship between two sentences as entailment, contradiction, or neutral. 

COMMON_SENSE_QA: Answer questions using common sense and implicit knowledge.

#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

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

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Ensure you output A[index] for each question before outputting {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.

Batched Questions to Answer:
'''