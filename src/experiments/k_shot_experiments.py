import random
import pickle
from src.models.model_api import (
    read_api_token, 
    LanguageModel, 
    TogetherAIModel, 
    OpenAIModel, 
    DebugModel,
    TogetherAIGenerationParameters, 
    OpenAIGenerationParameters, 
    DebugGenerationParameters,
)
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

from src.utils.evaluation import extract_answers_batch, Evaluation

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
class BatchPromptingExperimentConfig:
    # can either load a dataset from huggingface or from a local json file
    questions_dataset_config : DatasetConfig
    examples_dataset_config : DatasetConfig
    # can also choose to load a dataset from a file for questions or examples
    task_description: str
    # we specify baseline as a boolean as we might have a batch of size 1 at the end, but it will still
    # use the batched prompt template rather than the baseline prompt template
    k_shot: int
    example_selection: ExampleSelectionType 
    example_question_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    example_answer_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    batch_size: int
    # Note that this will include the api model name whereas model_name will be less formal like GPT-3.5 vs LLama-2-70B, etc.
    model_api: ModelAPIType
    generation_params: GENERATION_PARAMETERS_TYPE
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

        self.batch_prompt_template = BatchPromptTemplate(
            examples=self.examples,
            dataset=self.config.questions_dataset_config.dataset,
            task_description=self.config.task_description,
            num_questions=self.config.batch_size,
            num_examples=self.config.k_shot,
            example_question_format=self.config.example_question_format,
            example_answer_format=self.config.example_answer_format,
            example_selection=self.config.example_selection,
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
            dataset = dataset.add_column('idx', list(range(len(self.questions))))
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
                    model_name=generation_params.model_name,
                    generation_params=generation_params
                )
            case ModelAPIType.DEBUG:
                model = DebugModel()
            # otherwise
            case _: 
                raise NotImplementedError("Only OpenAI and TogetherAI APIs are currently supported")
        # cover all bases
        return model

    def batch_query_model(
        self,
        model_inputs: List[Tuple[List[ID_TYPE], str]], 
    ) -> List[Tuple[List[ID_TYPE], str]]:

        # create the model API object if it doesn't exist
        if self.model is None:
            self.model = self.load_language_model()

        batched_results = [
            (ids, self.model.query(prompt))
            for (ids, prompt) in tqdm(model_inputs)
        ]
        return batched_results

    def batch_questions(self) -> List[Tuple[List[ID_TYPE], Dict[str, Any]]]:

        batched_dataset : List[Dict[str, List[Any]]] = [   
            self.questions[i:i+self.config.batch_size]
            for i in range(0, len(self.questions), self.config.batch_size)
        ]
        batched_questions : List[Tuple[List[ID_TYPE], Dict[str, Any]]] = []
        for batch in batched_dataset:
            ids = batch[DATASET_ID_KEYS[self.config.questions_dataset_config.dataset][0]]
            questions = [
                {key : batch[key][i] for key in batch.keys()}
                for i in range(len(ids))
            ]
            batched_questions.append((ids, questions))

        return batched_questions

    def execute(self) -> Dict[int, str]:
        """
        TODO:
        X Load Dataset
        X Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
        X query model for each input (save raw outputs to a file somewhere)
        - parse out answer from model response
        """
        # splits self.questions into batches that are lists of individual dictionaries along with their ids
        batched_questions: List[Dict[str, List[Any]]] = self.batch_questions()

        batched_questions = batched_questions

        # generate prompts for each batch
        batched_model_inputs : List[Tuple[List[ID_TYPE], str]] = [
            (ids, self.batch_prompt_template.generate_prompt(batch))
            for (ids, batch) in batched_questions
        ]
        # for debug purposes
        batched_model_inputs = batched_model_inputs
        # TODO: igure out how to also save the config
        # which includes a lambda/function that might be tricky to pickle
        # print("Dumping batched model inputs to file...")
        # pickle.dump((batched_model_inputs), open('batched_model_inputs.pkl', 'wb'))
        # query model
        batched_model_outputs = self.batch_query_model(batched_model_inputs)

        pred = []
        for batched_output in batched_model_outputs:
            batched_output_parsed = extract_answers_batch(batched_output)
            assert len(batched_output_parsed) == len(batched_model_inputs)
            pred.extend(batched_output_parsed)
        evaluator = Evaluation()
        stats = evaluator.get_stats(y_pred=pred, y_true=ground_truth_answers, answer_type = answer_types[i])
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
            num_questions: int,
            num_examples: int,
            # the optional int is the index of the example in the batch, none if baseline
            example_question_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_answer_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_selection: ExampleSelectionType,
    ):
        self.examples = examples
        self.dataset = dataset
        self.task_description = task_description
        self.num_questions = num_questions
        self.num_examples = num_examples
        self.example_question_format = example_question_format
        self.example_answer_format = example_answer_format
        self.example_selection = example_selection

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
                self.example_selector = selector_class.from_examples(
                    # Need to turn hf dataset into a list of dicts
                    examples=list(self.examples),
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
        # TODO     
        prompt = "\n".join(
            [
                # TODO: Should the spacing between description and examples and questions be user-defined/programmable?
                self.task_description,
                "",
                *example_questions,
                *example_answers,
                "",
                *questions,
            ]
        )
        return prompt



if __name__ == "__main__":
    
    from data.parsing_functions import *
    example_question_format = lambda example, i: f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"
    example_answer_format=lambda example, i: f"Answer[{i}]: {example['label']}"

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
    task_description_rte = 'Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.'
    # example_question_format_rte = lambda example, i: f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"
    # example_answer_format_rte = lambda example, i: f"A[{i}]: {example['label']}"


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
    task_description_GSM8K = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''

    '''
    # TODO: Rohan: Can you split reasoning-machines/gsm-hard[train] into a train test split?  
    We only have train in gsm-hard so we need to split both. The following below is commented out because sampling is done from the same place.
    '''
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
    task_description_MNLI = '''You are tasked with the job of Multi-Genre Natural Language Inference (MNLI). For each task, you will be given a premise sentence and a hypothesis sentence. Your job is to predict the relationship between the premise and the hypothesis, classifying each pair as either 'entailment', 'contradiction', or 'neutral'. Instruction: For each question in the batch, provide a single answer, following the format A[idx]: answer. Output only the answers with the associated index in "A[idx]: answer" format. Each answer should be only one of the following: 'entailment', 'contradiction', or 'neutral'. So in other words, for each question, you should output one of the following: A[idx]: entailment, A[idx]: contradiction, or A[idx]: neutral.'''



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
    task_description_COMMON_SENSE = '''You are tasked with answering multiple-choice questions that require both contextual understanding and general world knowledge. Each question will have five options labeled 'a', 'b', 'c', 'd', and 'e'. Your job is to select the most appropriate answer by outputting the letter corresponding to that option. " These questions are part of the CommonsenseQA dataset, designed to test your ability to answer questions that often require prior knowledge. Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format. '''






    config_param_list = [
        [questions_config_rte, examples_config_rte, task_description_rte, rte_question_format, rte_answer_format],
        [questions_config_GSM8K, examples_config_GSM8K, task_description_GSM8K, gsm8k_question_format, gsm8k_answer_format],
        # [questions_config_MBPP, examples_config_MBPP, task_description_MBPP, mbpp_question_format, mbpp_answer_format],
        [questions_config_MNLI, examples_config_MNLI, task_description_MNLI, mnli_question_format, mnli_answer_format],
        # [questions_config_GSM8K_HARD, examples_config_GSM8K_HARD, task_description_GSM8K_HARD, gsm8k_question_format, gsm8k_answer_format],
        # [questions_config_COMMON_SENSE, examples_config_COMMON_SENSE, task_description_COMMON_SENSE, commonsense_question_format, commonsense_answer_format] 
        ]

    stats = []
    for questions_config, examples_config, task_description in config_param_list:
        config = BatchPromptingExperimentConfig(
            questions_dataset_config=questions_config,
            examples_dataset_config=examples_config,
            task_description='Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.',
            k_shot=7,
            example_selection=ExampleSelectionType.MAX_MARGINAL_RELEVANCE,
            example_question_format=example_question_format,
            example_answer_format=example_answer_format,
            batch_size=4,
            model_api=ModelAPIType.OPEN_AI,
            generation_params=oai_gen_params,
            random_seed=0,
        )
        experiment = BatchPromptExperiment(config)
        stat = experiment.execute()
        stats.append(stat)



# Other testing graveyard

# config = BatchPromptingExperimentConfig(
#     dataset=DatasetType.RTE,
#     hf_dataset_path=['glue', 'rte'],
#     examples_split_name='train',
#     questions_split_name='validation',
#     task_description='Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.',
#     k_shot=3,
#     example_selection=ExampleSelectionType.RANDOM,
#     example_question_format=example_question_format,
#     example_answer_format=example_answer_format,
#     batch_size=4,
#     generation_params=TogetherAIGenerationParameters(
#         model_name='togethercomputer/llama-2-7b',
#         max_tokens=64,
#         temperature=0.7,
#         top_p=0.9,
#         top_k=0,
#         repetition_penalty=1.0,
#         logprobs=0,
#     ),
#     random_seed=0,
# )