task_description_rte = '''**Objective**: Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes. You must answer all questions in the batch. The total number of questions in the batch is defined as batch_size = {batch_size}.

An answer of 0 means that the given Hypothesis and Premise logically entail each other. 
An answer of 1 means the given Hypothesis and Premise do NOT entail each other.

**Method**: Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.

#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.

2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified `{batch_size}`.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer. Never answer with saying the answer is ambiguous or that you can't answer a question. Even if you are not sure, ensure that for every answer, you output the "The answer is 0" or "The answer is 1".

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
4. **Answer Formatting**: After each intermediate reasoning, you must conclude with a definitive statement in the form of "The answer is 0" or "The answer is 1" without any variation. This precise phrasing is crucial and must be used consistently for each response to ensure clarity and uniformity in the results. Deviations from this format will be considered incorrect, even if the classification is accurate.

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.
{few_shot_examples}
Batched Questions to Answer:
'''

# task_description_COMMON_SENSE = '''You are tasked with answering multiple-choice questions that require both contextual understanding and general world knowledge. Each question will have five options labeled 'a', 'b', 'c', 'd', and 'e'. Your job is to select the most appropriate answer by outputting the letter corresponding to that option. " These questions are part of the CommonsenseQA dataset, designed to test your ability to answer questions that often require prior knowledge. Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format. '''
task_description_COMMON_SENSE = '''
### **Objective**: Your task is to solve a set of multiple-choice questions from the CommonsenseQA dataset in a batch. CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . You will be given `{{batch_size}}` questions each time, as input. These questions are designed to test your ability to answer queries that often require contextual understanding and general world knowledge. Your goal is to select the letter corresponding to the most appropriate answer among five options labeled 'a', 'b', 'c', 'd', and 'e' for each question in the batch. You must answer all questions in the batch. The total number of questions in the batch is defined as `batch_size = {{batch_size}}`.

An answer of 'A' means you believe option 'A' is the most appropriate answer.
An answer of 'B' means you believe option 'B' is the most appropriate answer.
An answer of 'C' means you believe option 'C' is the most appropriate answer.
An answer of 'D' means you believe option 'D' is the most appropriate answer.
An answer of 'E' means you believe option 'E' is the most appropriate answer.

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
4. **Answer Formatting**: Remember to output A[idx] before every answer, where idx corresponds to the correct question number, starting with 0. After each intermediate reasoning, you must conclude with a definitive statement in the form of "The answer is A", "The answer is B", "The answer is C", "The answer is D", or "The answer is E" without any variation. This precise phrasing is crucial and must be used consistently for each response to ensure clarity and uniformity in the results. Deviations from this format will be considered incorrect, even if the classification is accurate.
The phrase 'The answer is' must directly precede each letter answer and come after the intermediate reasoning, separated by a semicolon. Ensure you output A[index] for each question before outputting {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.
{few_shot_examples}
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
3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer. Never answer with saying the answer is ambiguous or that you can't answer a question. Even if you are not sure, choose the answer that is most likely to be correct, and ensure that for every answer, you output the "The answer is 0", "The answer is 1", or "The answer is 2".


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
4. **Answer Formatting**: Rember to output A[idx] before every answer, where idx corresponds to the correct premise and hypothesis number, starting with 0. After each intermediate reasoning, you must conclude with a definitive statement in the form of "The answer is 0", "The answer is 1", or "The answer is 2" without any variation. This precise phrasing is crucial and must be used consistently for each response to ensure clarity and uniformity in the results. Deviations from this format will be considered incorrect, even if the classification is accurate.
The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format. Before outputting text, ensure that the format of each answer is "A[idx]: {{intermediate_reasoning}} The answer is {{result_integer}}." where idx is the index of the question, intermediate_reasoning is the intermediate reasoning, and result_integer is the final integer answer to the question. If you haven't said verbatim "The answer is 0", "The answer is 1", or "The answer is 2" for each answer, then your answer will be marked as incorrect. 
{few_shot_examples}
Batched Questions to Answer:
'''
task_description_GSM8K = '''
**Objective**: Your task is to solve a set of math questions in a batch. You will be given a series of questions from the GSM8K dataset as input, totaling {{batch_size}}. The goal is to accurately respond to each one, providing clear reasoning and the final integer answer.

**Complexity**: Expect to engage in a reasoning process that involves 2 to 8 steps to derive the answer for each question.

**Method**: Employ basic arithmetic operations to execute a series of calculations that lead to the solution of each question.

#### Instructions:

1. **Intermediate Reasoning**: For each question, detail all the evaluative steps you've taken to understand the premise and formulate the hypothesis. This may include parsing key phrases, identifying contradictions, or establishing logical links.

2. **Batch Size**: You are required to provide an answer for each question within the batch, with the total number of provided answers matching {{batch_size}}.

3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer. Never answer with saying the answer is ambiguous or that you can't answer a question. Even if you are not sure, ensure that for every answer, you output the "The answer is 0", "The answer is 1", or the answer is whatever integer that is your best guess to the problem.

#### Input Format:
- The batch of questions will be presented with each question prefixed by its index, starting from 0:
Q[0]: {{Question_0_Text}}
Q[1]: {{Question_1_Text}}
...
Q[{{batch_size - 1}}]: {{Question_{{batch_size - 1}}_Text}}

#### Output Format:
- Adhere to the following structure for each response without deviation:
A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
- `index`: The index of the question being answered, prefixed with 'A' and enclosed in square brackets.
- `{{Intermediate_Reasoning}}`: The detailed steps and logical progression that led to the conclusion.
- `{{Answer_Integer}}`: The final answer must be an integer, without any extraneous characters, punctuation, quotations, symbols, or units.
- `{{Answer_Integer}}`: This is the final integer answer to the question without any units.

4. **Answer Formatting**: Remember to output A[idx] before every answer, where idx corresponds to the correct question number, starting with 0. After each intermediate reasoning, you must conclude with a definitive statement in the form of "The answer is 0" or "The answer is 1" without any variation. This precise phrasing is crucial and must be used consistently for each response to ensure clarity and uniformity in the results. Deviations from this format will be considered incorrect, even if the classification is accurate.

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that {batch_size} answers are provided in our desired format.
{few_shot_examples}
Batched Questions to Answer:
'''

# from src.experiments.k_shot_experiment import *
# from src.utils.parsing_functions import * 

# oai_gen_params = OpenAIGenerationParameters(
#             model_name='gpt-3.5-turbo',
#             temperature=0.6,
#             max_tokens=64,
#             frequency_penalty=1.0,
#         )

# questions_config_rte = DatasetConfig(
#     dataset=DatasetType.RTE,
#     hf_dataset_path=['glue', 'rte'],
#     split_name='validation',
# )
# examples_config_rte = DatasetConfig(
#     dataset=DatasetType.RTE,
#     hf_dataset_path=['glue', 'rte'],
#     split_name='train',
# )
# task_description_rte = 'Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.'


# questions_config_GSM8K = DatasetConfig(
#     dataset=DatasetType.GSM8K,
#     hf_dataset_path=['gsm8k', 'main'],
#     split_name='test',
# )
# examples_config_GSM8K = DatasetConfig(
#     dataset=DatasetType.GSM8K,
#     hf_dataset_path=['gsm8k', 'main'],
#     split_name='train',
# )
# task_description_GSM8K = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''

# '''
# # TODO: Rohan: Can you split reasoning-machines/gsm-hard[train] into a train test split?  
# We only have train in gsm-hard so we need to split both. The following below is commented out because sampling is done from the same place.
# '''
# questions_config_GSM8K_HARD = DatasetConfig(
#     dataset=DatasetType.GSM8K_HARD,
#     hf_dataset_path=["reasoning-machines/gsm-hard"],
#     split_name='train',
# )
# examples_config_GSM8K_HARD = DatasetConfig(
#     dataset=DatasetType.GSM8K_HARD,
#     hf_dataset_path=["reasoning-machines/gsm-hard"],
#     split_name='train',
# )
# task_description_GSM8K_HARD = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''



# questions_config_MBPP = DatasetConfig(
#     dataset=DatasetType.MBPP,
#     hf_dataset_path=['mbpp'],
#     split_name='validation',
# )
# examples_config_MBPP = DatasetConfig(
#     dataset=DatasetType.MBPP,
#     hf_dataset_path=['mbpp'],
#     split_name='train',
# )
# task_description_MBPP = '''You are tasked with solving Python programming problems that are designed to be solvable by entry-level programmers. Each problem will consist of a task description, and your job is to output a string that when parsed is an executable Python code function that fulfills the requirements of the task. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format.'''



# questions_config_MNLI = DatasetConfig(
#     dataset=DatasetType.MNLI,
#     hf_dataset_path=['glue', 'mnli'],
#     split_name='validation_matched',
# )
# examples_config_MNLI = DatasetConfig(
#     dataset=DatasetType.MNLI,
#     hf_dataset_path=['glue', 'mnli'],
#     split_name='train',
# )
# task_description_MNLI = '''You are tasked with the job of Multi-Genre Natural Language Inference (MNLI). For each task, you will be given a premise sentence and a hypothesis sentence. Your job is to predict the relationship between the premise and the hypothesis, classifying each pair as either 'entailment', 'contradiction', or 'neutral'. Instruction: For each question in the batch, provide a single answer, following the format A[idx]: answer. Output only the answers with the associated index in "A[idx]: answer" format. Each answer should be only one of the following: 'entailment', 'contradiction', or 'neutral'. So in other words, for each question, you should output one of the following: A[idx]: entailment, A[idx]: contradiction, or A[idx]: neutral.'''



# questions_config_COMMON_SENSE = DatasetConfig(
#     dataset=DatasetType.COMMON_SENSE,
#     hf_dataset_path=['commonsense_qa'],
#     split_name='validation',
# )
# examples_config_COMMON_SENSE = DatasetConfig(
#     dataset=DatasetType.COMMON_SENSE,
#     hf_dataset_path=['commonsense_qa'],
#     split_name='train',
# )
# task_description_COMMON_SENSE = '''You are tasked with answering multiple-choice questions that require both contextual understanding and general world knowledge. Each question will have five options labeled 'a', 'b', 'c', 'd', and 'e'. Your job is to select the most appropriate answer by outputting the letter corresponding to that option. " These questions are part of the CommonsenseQA dataset, designed to test your ability to answer questions that often require prior knowledge. Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format. '''




# config_param_list = { "rte": [questions_config_rte, examples_config_rte, task_description_rte, rte_question_format, rte_answer_format],
#     "GSM8K": [questions_config_GSM8K, examples_config_GSM8K, task_description_GSM8K, gsm8k_question_format, gsm8k_answer_format],
#     # "MBPP": [questions_config_MBPP, examples_config_MBPP, task_description_MBPP, mbpp_question_format, mbpp_answer_format],
#     "MNLI": [questions_config_MNLI, examples_config_MNLI, task_description_MNLI, mnli_question_format, mnli_answer_format],
#     #"GSM8K_HARD": [questions_config_GSM8K_HARD, examples_config_GSM8K_HARD, task_description_GSM8K_HARD, gsm8k_question_format, gsm8k_answer_format],
#     #"COMMON_SENSE": [questions_config_COMMON_SENSE, examples_config_COMMON_SENSE, task_description_COMMON_SENSE, commonsense_question_format, commonsense_answer_format] 
# }