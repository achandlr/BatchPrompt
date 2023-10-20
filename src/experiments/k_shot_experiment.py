from src.utils.prompts import FlexiblePromptTemplate
from src.models.GPT_API import set_gpt_api_key, read_api_token, query_model
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

@dataclass
class TogetherAIGenerationParameters:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    logprobs: int

@dataclass
class OpenAIGenerationParameters:
    model_temperature: float
    max_tokens: int
    frequency_penalty: float

@dataclass
class KShotBatchPromptingExperimentConfig:
    dataset_name: str
    model_name: str
    k_shot: int
    batch_size: int
    generation_params: dict

# GSM8K: https://huggingface.co/datasets/gsm8k
"""
Datsets:

GSM8K: Math Reasoning
https://huggingface.co/datasets/gsm8k
HF Dataset: load_dataset('gsm8k', 'main')
DatasetDict({
    train: Dataset({
        features: ['question', 'answer'],
        num_rows: 7473
    })
    test: Dataset({
        features: ['question', 'answer'],
        num_rows: 1319
    })
})

MBPP: Mostly Basic Python Problems
https://huggingface.co/datasets/mbpp
HF Dataset: load_dataset('mbpp', 'main')


"""
datasets = ['GSM8K', 'MBPP', 'RTE', 'MNLI']

def execute_k_shot_experiment(experiment_config: KShotBatchPromptingExperimentConfig) -> Dict[int, str]:
    """
    TODO:
    - Load Dataset
    - Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
    - query model for each input (save raw outputs to a file somewhere)
    - parse out answer from model response
    """
    
