import random
import pickle
from src.models.model_api import LanguageModel, TogetherAIModel, OpenAIModel, read_api_token
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
from enum import Enum, auto
from tqdm import tqdm

from langchain.prompts.example_selector.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# TYPES AND ENUMS
ID_TYPE = Union[str, int]
EXAMPLE_FORMAT_FUNCTION_TYPE = Callable[[Dict[str, Any], Optional[int]], str]

class DatasetType(Enum):
    GSM8K = auto()
    MBPP = auto()
    RTE = auto()
    MNLI = auto()

# DICTS

DATASET_ID_KEYS = {
    DatasetType.GSM8K : ['idx'],
    DatasetType.MBPP : ['task_id'],
    DatasetType.RTE : ['idx'],
    DatasetType.MNLI : ['idx'],
}

DATASET_INPUT_KEYS = {
    DatasetType.GSM8K : ['question'],
    DatasetType.MBPP : ['text'],
    DatasetType.RTE : ['sentence1', 'sentence2'],
    DatasetType.MNLI : ['premise', 'hypothesis'],
}

DATASET_LABEL_KEYS = {
    DatasetType.GSM8K : ['answer'],
    DatasetType.MBPP : ['code', 'test_list', 'test_setup_code', 'challenge_test_list'],
    DatasetType.RTE : ['label'],
    DatasetType.MNLI : ['label'],
}

# these are the texts that go before the Q[i] in batch prompts
DATASET_BATCH_INDEX_Q = {
    DatasetType.GSM8K : ['Q'],
    DatasetType.MBPP : ['Q'],
    DatasetType.RTE : ['Premise', 'Hypothesis'],
    DatasetType.MNLI : ['Premise', 'Hypothesis'],
}

# these are the texts that go before the Q[i] in batch prompts
DATASET_BATCH_INDEX_A = {
    DatasetType.GSM8K : ['A'],
    DatasetType.MBPP : ['A'],
    DatasetType.RTE : ['Answer'],
    DatasetType.MNLI : ['Answer'],
}

class ExampleSelectionType(Enum):
    RANDOM = auto()
    SEMANTIC = auto()
    LEXICAL = auto()
    MAX_MARGINAL_RELEVANCE = auto()

@dataclass
class TogetherAIGenerationParameters:
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    logprobs: int

# TODO: 
@dataclass
class OpenAIGenerationParameters:
    model_name: str
    temperature: float
    max_tokens: int
    # frequency_penalty: float

@dataclass
class BatchPromptingExperimentConfig:
    # could be the name of a dataset, or a list of strings that specify the path to a dataset 
    # e.g. 'mbpp' vs ['mbpp', 'sanitized']
    dataset : DatasetType
    hf_dataset_path: Union[str, List[str]]
    examples_split_name: str
    evaluation_split_name: str
    task_description: str
    # we specify baseline as a boolean as we might have a batch of size 1 at the end, but it will still
    # use the batched prompt template rather than the baseline prompt template
    k_shot: int
    is_baseline: bool
    example_selection: ExampleSelectionType 
    example_question_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    example_answer_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    batch_size: int
    # Note that this will include the api model name whereas model_name will be less formal like GPT-3.5 vs LLama-2-70B, etc.
    generation_params: Union[TogetherAIGenerationParameters, OpenAIGenerationParameters]
    random_seed: int = 0

# datasets = ['GSM8K', 'MBPP', 'glue-RTE', 'glue-MNLI']

class BatchPromptExperiment:
    def __init__(
            self,
            config: BatchPromptingExperimentConfig,
    ):
        self.config = config
        self.examples = load_dataset(
            *self.config.hf_dataset_path,
            split=self.config.examples_split_name,
        )
        self.questions = load_dataset(
            *self.config.hf_dataset_path,
            split=self.config.evaluation_split_name,
        )
        # must add an index column to gsm8k
        if self.config.dataset == DatasetType.GSM8K:
            self.questions = self.questions.add_column('idx', list(range(len(self.questions))))

        self.batch_prompt_template = BatchPromptTemplate(
            examples=self.examples,
            dataset=self.config.dataset,
            task_description=self.config.task_description,
            num_questions=self.config.batch_size,
            num_examples=self.config.k_shot,
            is_baseline=self.config.is_baseline,
            example_question_format=self.config.example_question_format,
            example_answer_format=self.config.example_answer_format,
            example_selection=self.config.example_selection,
        )
        self.model = None
        self.model : LanguageModel = self.load_language_model()
        random.seed(self.config.random_seed)
    
    def load_language_model(self) -> LanguageModel:
        if isinstance(self.config.generation_params, OpenAIGenerationParameters):
            token = read_api_token(r'data/imported/api_token.txt')
            model = OpenAIModel(
                api_token=token,
                model_name=self.config.generation_params.model_name,
                generation_params=self.config.generation_params
            )
        elif isinstance(self.config.generation_params, TogetherAIGenerationParameters):
            # together.ai
            token = read_api_token(r'data\imported\datasets\together_ai_token.txt')
            model = TogetherAIModel(
                api_token=token,
                model_name=self.config.generation_params.model_name,
                generation_params=self.config.generation_params
            )
        else: 
            raise NotImplementedError("Only OpenAI and TogetherAI APIs are currently supported")
        # cover all bases
        self.model = model
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
            ids = batch[DATASET_ID_KEYS[self.config.dataset][0]]
            questions = [
                {key : batch[key][i] for key in batch.keys()}
                for i in range(len(ids))
            ]
            batched_questions.append((ids, questions))

        return batched_questions

    def execute(self) -> Dict[int, str]:
        """
        TODO:
        - Load Dataset
        - Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
        - query model for each input (save raw outputs to a file somewhere)
        - parse out answer from model response
        """
        # splits self.questions into batches that are lists of individual dictionaries along with their ids
        batched_questions: List[Dict[str, List[Any]]] = self.batch_questions()

        # TODO Rohan: Why are we taking the first 10
        batched_questions = batched_questions[:10]

        # generate prompts for each batch
        batched_model_inputs : List[Tuple[List[ID_TYPE], str]] = [
            (ids, self.batch_prompt_template.generate_prompt(batch))
            for (ids, batch) in batched_questions
        ]
        # TODO: Need to store the batched model inputs in a reasonable way/ folder location
        # TODO: Need to run evaluation
        outputs = self.batch_query_model(batched_model_inputs[0:2])
        # TODO: Need to return evalautiono
        # TODO: igure out how to also save the config, which includes a lambda/function that might be tricky to pickle
        print("Dumping batched model inputs to file...")
        pickle.dump((batched_model_inputs), open('batched_model_inputs.pkl', 'wb'))
        # query model

        # batched_model_outputs = self.batch_query_model(batched_model_inputs)
        # # # save the pickled batched model outputs to file
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
            is_baseline: bool,
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
        self.is_baseline = is_baseline
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

                self.examples_selector = selector_class.from_examples(
                    # Need to turn hf dataset into a list of dicts
                    examples=list(self.examples),
                    # TODO: do we want embeddings to be configurable? probably not... it has sensible defaults
                    # and it is certainly not a menaingful variable in our experiments
                    embeddings=HuggingFaceEmbeddings(),
                    vectorstore_cls=Chroma,
                    k=self.num_examples,
                    input_keys=DATASET_INPUT_KEYS[self.dataset],
                )

    def get_examples(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        match self.example_selection:
            case ExampleSelectionType.RANDOM:
                return [self.examples[i] for i in random.sample(range(len(self.examples)), self.num_examples)] 
            case ExampleSelectionType.LEXICAL:
                raise NotImplementedError("Lexical example selection is not yet implemented")
            case ExampleSelectionType.SEMANTIC | ExampleSelectionType.MAX_MARGINAL_RELEVANCE:
                raise NotImplementedError("TODO: decide how to handle semantic example retrieval for batches of examples")

    # TODO: How to distinguish between baseline and batch size of 1 (baseline shouldn't have [i] in the prompt)
    # Resolved - pass is_baseline
    def generate_prompt(self, batch: List[Dict[str, Any]]) -> str:
        example_questions = []
        example_answers = []
        questions = []

        examples = self.get_examples(batch)
        for i, example in enumerate(examples):
            # the format functions take care of the Q[i] notation
            example_questions.append(self.example_question_format(example, None if self.is_baseline else i))
            example_answers.append(self.example_answer_format(example, None if self.is_baseline else i))
        
        for i, question in enumerate(batch):
            questions.append(self.example_question_format(question, None if self.is_baseline else i))
        
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
    
    # TODO: I don't think these are necessary and instead should be loaded as question or as answer
    example_question_format = lambda example, i: f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"
    example_answer_format=lambda example, i: f"Answer[{i}]: {example['label']}"

    config = BatchPromptingExperimentConfig(
        dataset=DatasetType.RTE,
        hf_dataset_path=['glue', 'rte'],
        examples_split_name='train',
        evaluation_split_name='validation',
        task_description='Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.',
        k_shot=3,
        is_baseline=False,
        example_selection=ExampleSelectionType.RANDOM,
        example_question_format=example_question_format,
        example_answer_format=example_answer_format,
        batch_size=4,
        generation_params=TogetherAIGenerationParameters(
            model_name='togethercomputer/llama-2-7b',
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            logprobs=0,
        ),
        # generation_params=OpenAIGenerationParameters(
        #     model_name = "gpt-3.5-turbo",
        #     temperature= .2,
        #     max_tokens = 4000
        # ),
        random_seed=0,
    )

    # config = BatchPromptingExperimentConfig(
    #     dataset=DatasetType.RTE,
    #     hf_dataset_path=['glue', 'rte'],
    #     examples_split_name='train',
    #     evaluation_split_name='validation',
    #     task_description='Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.',
    #     k_shot=3,
    #     is_baseline=False,
    #     example_selection=ExampleSelectionType.RANDOM,
    #     example_question_format=example_question_format,
    #     example_answer_format=example_answer_format,
    #     batch_size=4,
    #     generation_params=OpenAIGenerationParameters(
    #         model_name = "gpt-3.5-turbo",
    #         temperature= .2,
    #         max_tokens = 4000
    #     ),
    #     random_seed=0,
    # )
    # TODO: modify config to be for gsm8k
    # gsm8k_config = BatchPromptingExperimentConfig(
    #     dataset=DatasetType.RTE,
    #     hf_dataset_path=['glue', 'rte'],
    #     examples_split_name='train',
    #     evaluation_split_name='validation',
    #     task_description='Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.',
    #     k_shot=3,
    #     is_baseline=False,
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

    experiment = BatchPromptExperiment(config)
    experiment.execute()
    print("DONE")