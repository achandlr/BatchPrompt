# Expanding Batch Prompting to Zero-Shot and Multi-Task Regimes
## Overview
Led by Alex Chandler and Rohan Jha from The University of Texas at Austin, this project brings significant advancements to Batch Prompting (BP) in Large Language Models (LLMs). Traditionally, LLMs required separate prompts for each task, which was computationally expensive. Our innovation lies in extending BP to zero-shot and multi-task contexts. We introduce new BatchPrompt templates that cater to both single-task and multi-task reasoning, enabling LLMs to process multiple queries simultaneously without needing few-shot examples. This approach maintains robustness even across diverse tasks in a batch.

## What is Batch Prompting?
Batch Prompting, as introduced by Cheng et al. (2023)[https://arxiv.org/abs/2301.08721], is a strategic inference technique enhancing the efficiency of LLMs in Natural Language Processing (NLP). It addresses the high computational and time costs associated with conventional NLP tasks—such as question answering, sentiment analysis, and named entity recognition—which required detailed prompts with in-context examples. Batch Prompting consolidates a group of b questions into a single prompt, allowing the LLM to generate responses for all questions in one LLM inference call. This method reduces the overall cost by sharing the instruction and example tokens across multiple samples. Initially limited to single-task formats with few-shot examples (typically k=b), our research expands Batch Prompting to include zero-shot scenarios and multi-task queries, overcoming previous constraints and increasing its application range.
 
## Technical Contributions and Findings
### Zero-Shot and Multi-Task Batch Prompting: 
We pioneer BP's extension to zero-shot learning, allowing LLMs to respond to queries without prior examples. Our research, spanning various models and datasets, shows that zero-shot BP is viable but has a higher rate of misformatted outputs. Additionally, BP proves resilient to task diversity in a batch, maintaining consistent performance across mixed tasks.

### BatchPrompt Templates: 
We introduce two innovative BatchPrompt templates. One is optimized for single-task reasoning, and the other for multi-task scenarios. These templates guide LLMs in generating structured and clear responses, adaptable in both zero-shot and few-shot settings.

### Token Efficiency and Limitations: 
We propose a refined token efficiency metric, taking into account the additional tokens required in batch prompts. Our study suggests that BP's efficiency peaks in tasks demanding minimal reasoning with concise outputs. Despite its efficiency, BP encounters challenges like parsing errors in zero-shot scenarios and limitations in tasks requiring complex reasoning.

### Future Directions: 
Our findings pave the way for future research, such as enhancing BP's efficiency in zero-shot contexts, reducing parsing errors, and exploring dynamic batch sizing techniques. Fine-tuning LLMs to reduce input tokens for effective BP is also an area worth exploring.

This project emphasizes Batch Prompting's versatility and efficiency in LLM applications, marking a substantial advancement in natural language processing.

## Installation
It's highly recommended to create a virtual environment for this project. In the root folder of your virtual environment, execute the following commands:

pip install .
For running the code in an editable mode through the debugger, use:

pip install -e .

## Running Code:
src/experiments/batched_tests.py runs all Single-Task Batched Experiments.
src/experiments/batched_tests_multitask.py runs all Multi-Task Batched Experiments.

## Understanding the Project:
For a comprehensive understanding, refer to our upcoming paper "Expanding Batch Prompting to Zero-Shot and Multi-Task Regimes," available at https://www.chandlerai.com/research/batch-prompting.
