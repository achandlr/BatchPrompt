from src.experiments.k_shot_experiment import *
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
task_description_rte = 'Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.'


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




config_param_list = { "rte": [questions_config_rte, examples_config_rte, task_description_rte, rte_question_format, rte_answer_format],
    "GSM8K": [questions_config_GSM8K, examples_config_GSM8K, task_description_GSM8K, gsm8k_question_format, gsm8k_answer_format],
    # "MBPP": [questions_config_MBPP, examples_config_MBPP, task_description_MBPP, mbpp_question_format, mbpp_answer_format],
    "MNLI": [questions_config_MNLI, examples_config_MNLI, task_description_MNLI, mnli_question_format, mnli_answer_format],
    #"GSM8K_HARD": [questions_config_GSM8K_HARD, examples_config_GSM8K_HARD, task_description_GSM8K_HARD, gsm8k_question_format, gsm8k_answer_format],
    #"COMMON_SENSE": [questions_config_COMMON_SENSE, examples_config_COMMON_SENSE, task_description_COMMON_SENSE, commonsense_question_format, commonsense_answer_format] 
}