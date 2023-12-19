# Expanding Batch Prompting to Zero-Shot and Multi-Task Regimes
## Overview
This project, led by Alex Chandler and Rohan Jha from The University of Texas at Austin, innovates in Batch Prompting (BP) for Large Language Models (LLMs). Traditionally, LLMs required individual prompts for each task, increasing computational costs. BP concatenates several questions into a single prompt, enabling LLMs to answer all in one go. Our work extends BP into zero-shot and multi-task contexts, introducing new BatchPrompt templates for both single-task and multi-task reasoning. These templates facilitate effective LLM performance without needing few-shot examples and retain robustness across diverse tasks in a batch.

## Technical Contributions and Findings
### Zero-Shot and Multi-Task Batch Prompting: We pioneer BP's extension to zero-shot learning, allowing LLMs to respond to queries without prior examples. Our research, spanning various models and datasets, shows that zero-shot BP is viable but has a higher rate of misformatted outputs. Additionally, BP proves resilient to task diversity in a batch, maintaining consistent performance across mixed tasks.

### BatchPrompt Templates: We introduce two innovative BatchPrompt templates. One is optimized for single-task reasoning, and the other for multi-task scenarios. These templates guide LLMs in generating structured and clear responses, adaptable in both zero-shot and few-shot settings.

### Token Efficiency and Limitations: We propose a refined token efficiency metric, taking into account the additional tokens required in batch prompts. Our study suggests that BP's efficiency peaks in tasks demanding minimal reasoning with concise outputs. Despite its efficiency, BP encounters challenges like parsing errors in zero-shot scenarios and limitations in tasks requiring complex reasoning.

### Future Directions: Our findings pave the way for future research, such as enhancing BP's efficiency in zero-shot contexts, reducing parsing errors, and exploring dynamic batch sizing techniques. Fine-tuning LLMs to reduce input tokens for effective BP is also an area worth exploring.

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
