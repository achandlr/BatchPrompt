from src.models.GPT_API import query_model
from src.models.GPT_API import set_api_key

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate as LangchainPromptTemplate
from langchain.prompts import FewShotPromptTemplate


class FlexiblePromptTemplate:
    # Class variables to store default configurations
    default_task_description = ""
    default_reasoning_types = {
        "End-to-end": "",
        "CoT": "Let's think step by step. Output sequential logic."
    }
    default_shot_types = {
        "Zero-Shot": "",
        "Few-Shot": "Consider the following examples and maintain their formatting.\n",
        "One-Shot": "Consider the following example and maintain its formatting."
    }
    # default_example_formats = {
    #     "End-to-end": "Question: {question}\Output: {output}",
    #     "CoT": "Question: {question}\Output: {output}"
    # }
    default_num_shots = {
        "Few-Shot": 3,
        "One-Shot": 1,
        "Zero-Shot": 0
    }

    def __init__(self, examples, task_description, num_shots_per_question=None, reasoning_type="End-to-end", shot_type="Zero-Shot"):

    # def __init__(self, examples, task_description, example_format, num_shots=None, reasoning_type="End-to-end", shot_type="Zero-Shot"):
        # Update the examples for Langchain components
        self.update_langchain_examples(examples)
        
        # Store the task description
        self.task_description = task_description
        
        # Set the reasoning type and retrieve its default description
        self.reasoning_type = reasoning_type
        self.reasoning_type_description = self.default_reasoning_types[reasoning_type]
        
        # Set the shot type and retrieve its default description
        self.shot_type = shot_type
        self.shot_type_description = self.default_shot_types[shot_type]
        
        # Create a full task description by combining task description, reasoning type, and shot type
        self.task_description_with_reasoning_type = f"{self.task_description}\n{self.reasoning_type_description}\n{self.shot_type_description}"
        
        # Choose the example format. If none is provided, select a default based on the reasoning type.
        # if example_format is None:
        #     self.example_format = self.default_example_formats[reasoning_type]
        # else:
        #     self.example_format = example_format'
        self.example_format = '{question}\n{output}'
        # Choose the number of shots. If none is provided, select a default based on the shot type.
        if num_shots_per_question is None:
            self.num_shots_per_question = self.default_num_shots[shot_type]
        else:
            self.num_shots_per_question = num_shots_per_question

        # Initialize Langchain components if Few-Shot is selected
        if self.shot_type == "Few-Shot" or self.shot_type == "One-Shot":
            self.init_langchain()

    # Method to update examples for Langchain components
    def update_langchain_examples(self, examples):
        self.examples = examples

    # Initialize Langchain components like example selectors and prompt templates
    def init_langchain(self):
        embeddings = HuggingFaceEmbeddings()
        to_vectorize = []
        for example in self.examples:
            question = example["question"]
            output = example["output"]
            example_in_format = self.example_format.format(question=question, output=output)
            to_vectorize.append(example_in_format)

        # Initialize example selector
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            HuggingFaceEmbeddings(),
            Chroma,
            k=self.num_shots_per_question
        )
        
        # Initialize prompt template
        example_prompt = LangchainPromptTemplate(
            input_variables=["question", "output"],
            template=self.example_format
        )

        # Initialize Few-Shot prompt
        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            # prefix=self.task_description_with_reasoning_type,
            example_prompt=example_prompt,
            suffix="Question {question}\nOutput: ",
            input_variables=["question"]
        )

    def fill_in(self, questions):
        filled_prompts = []

        # Add task description and reasoning type

        filled_prompts.append(self.task_description_with_reasoning_type)
        # Add few-shot examples if applicable
        if self.shot_type == "Few-Shot":
            filled_prompts.append('''#Examples in Batch for Few-Shot''')
            few_shot_questions = []
            few_shot_answers = []
            for idx in range(len(questions)):
                # for shot_num in range(self.num_shots_per_question):
                question_to_get_few_shot_from = questions[idx]
                few_shot_selected_examples= self.few_shot_prompt.format(question=question_to_get_few_shot_from)
                few_shot_selected_examples_split = few_shot_selected_examples.split("\n")
                for i in range(0, len(few_shot_selected_examples_split), 3):
                    # Check if there is a complete set of 3 elements
                    if i + 2 < len(few_shot_selected_examples_split):
                        # Append the question and answer to their respective lists
                        few_shot_questions.append(few_shot_selected_examples_split[i])
                        few_shot_answers.append(few_shot_selected_examples_split[i + 1])
                # questions = few_shot_selected_examples_split[0::2]
                # answers = few_shot_selected_examples_split[1::2]
                # few_shot_questions.append(questions)
                # few_shot_answers.append(answers)
            
            for idx in range(len(few_shot_questions)):
                filled_prompts.append(f"Q[{idx+1}]: {few_shot_questions[idx]}")
            filled_prompts.append("#Response to examples in Batch for Few-Shot")
            for idx in range(len(few_shot_answers)):
                filled_prompts.append(f"A[{idx+1}]: {few_shot_answers[idx]}")
            filled_prompts.append("#Questions in Batch to answer")

        if self.shot_type == "One-Shot":
            filled_prompts.append('''# Example in Batch for One-Shot''')
            few_shot_questions = []
            few_shot_answers = []
            # for shot_num in range(self.num_shots_per_question):
            question_to_get_few_shot_from = questions[0]
            few_shot_selected_examples= self.few_shot_prompt.format(question=question_to_get_few_shot_from)
            few_shot_selected_examples_split = few_shot_selected_examples.split("\n")

            # Append the question and answer to their respective lists
            few_shot_questions.append(few_shot_selected_examples_split[0])
            few_shot_answers.append(few_shot_selected_examples_split[1])
            # questions = few_shot_selected_examples_split[0::2]
            # answers = few_shot_selected_examples_split[1::2]
            # few_shot_questions.append(questions)
            # few_shot_answers.append(answers)
            
            for idx in range(len(few_shot_questions)):
                filled_prompts.append(f"Q[{idx+1}]: {few_shot_questions[idx]}")
            filled_prompts.append("# Response to Example in Batch for One-Shot")
            for idx in range(len(few_shot_answers)):
                filled_prompts.append(f"A[{idx+1}]: {few_shot_answers[idx]}")
            filled_prompts.append("## Questions in Batch to Answer")
            # example = self.few_shot_prompt.format(question=self.examples[shot_num]["question"])
            # filled_prompts.append(f"Q[{question_idx}]: {self.examples[shot_num]['question']}")
            # for idx in range(len(questions)):
            #     for shot_num in range(self.num_shots_per_question):
            #         example = self.few_shot_prompt.format(question=self.examples[shot_num]["question"])
            #         filled_prompts.append(f"A[{idx+1}]: {self.examples[shot_num]['output']}")
                    # filled_prompts.append(f"Q[{idx+1}-ex{shot_num+1}]: {self.examples[shot_num]['question']}\nA[{idx+1}-ex{shot_num+1}]: {self.examples[shot_num]['output']}")
        
        # Add the actual questions in the prompt
        for idx, question in enumerate(questions):
            filled_prompts.append(f"Q[{idx+1}]: {question}")
        # Return the complete prompt joined with line breaks
        batched_prompt = '\n'.join(filled_prompts)
        return batched_prompt
    
if __name__ == "__main__":

    flexible_prompt_template = FlexiblePromptTemplate(
        examples=[
            {'question': 'What is 2+2?', 'output': '4'},
            {'question': 'What is 3+3?', 'output': '6'},
            {'question': 'How many elbows does a man have?', 'output': '2'},
            {'question': 'Where are elbows?', 'output': 'Elbows are on the arm.'}
        ],
        task_description='You can solve arithmetic problems. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format',
        # example_format='Q: {question}\nA: {output}',
        # example_format='{question}\n{output}',
        # example_format='{question}\n{output}',
        num_shots_per_question=2,
        reasoning_type='End-to-end',
        shot_type='Few-Shot'
    )
    # Generate a batched prompt with multiple questions
    batched_prompt = flexible_prompt_template.fill_in(['What is 5+5?', 'How many elbows does a women have?', 'Why are elbows important?'])
    set_api_key(r"data/imported/datasets/api_token.txt")

    output = query_model("gpt-3.5-turbo", batched_prompt)

    # Display the batched prompt
    print(batched_prompt)
    print(output)

