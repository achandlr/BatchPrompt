from src.experiments.k_shot_experiment_configs import config_param_list
from src.experiments.k_shot_experiment import * # BatchPromptExperiment,BatchPromptingExperimentConfig

# TODO: For now just use OPEN_AI, but soon we will loop through API calls.

for questions_config, examples_config, task_description, question_format, answer_format in config_param_list:
    config = BatchPromptingExperimentConfig(
    questions_dataset_config=questions_config,
    examples_dataset_config=examples_config,
    task_description=task_description,
    k_shot=7,
    example_selection=ExampleSelectionType.RANDOM,
    example_question_format=question_format,
    example_answer_format=answer_format,
    batch_size=4,
    model_api=ModelAPIType.OPEN_AI,
    generation_params=oai_gen_params,
    random_seed=0,
)
    experiment = BatchPromptExperiment(config)
    results = experiment.execute()
    