
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate as LangchainPromptTemplate
from langchain.prompts import FewShotPromptTemplate

class FlexiblePromptTemplate:
    # TODO: Create high quality examples for few-shot
    # default_standard_examples = [
    #     {"claim": "The Earth is flat", "context": "Multiple scientific studies confirm that the Earth is an oblate spheroid.", "output": "0"},
    #     {"claim": "Water boils at 100 degrees Celsius at sea level", "context": "The boiling point of water is generally accepted to be 100 degrees Celsius at 1 atmosphere, which is sea level.", "output": "1"},
    #     {"claim": "Light travels faster than sound", "context": "The speed of light in a vacuum is approximately 299,792 kilometers per second, far exceeding the speed of sound in air, which is around 343 meters per second.", "output": "1"}
    # ]

    # default_CoT_examples = [
    #     {"claim": "The Earth is flat", "context": "Multiple scientific studies confirm that the Earth is an oblate spheroid.", "output": "Let's think step by step. The claim 'The Earth is flat' is directly contradicted by the context 'Multiple scientific studies confirm that the Earth is an oblate spheroid.' Therefore, the claim is completely not supported, so the score is 0."},
    #     {"claim": "Water boils at 100 degrees Celsius at sea level", "context": "The boiling point of water is generally accepted to be 100 degrees Celsius at 1 atmosphere, which is sea level.", "output": "Let's think step by step. The claim 'Water boils at 100 degrees Celsisus at sea level' is directly supported by the context 'The boiling point of water is generally accepted to be 100 degrees Celsius at 1 atmosphere, which is sea level.' Therefore, the claim is completely supported, so the score is 1."},
    #     {"claim": "Light travels faster than sound", "context": "The speed of light in a vacuum is approximately 299,792 kilometers per second, far exceeding the speed of sound in air, which is around 343 meters per second.", "output": "Let's think step by step. The claim states that 'Light travels faster than sound'. The context states that 'The speed of light in a vacuum is approximately 299,792 kilometers per second. The speed of sound in air is around 343 meters per second. Converting to the same units, the speed of air is .343 kilometers per second. Since, 299,792 kilometers per second is much faster than .343 kilometers per second, the claim is directly supported. Therefore, the score is 1."}
    # ]
    default_task_description = ""

    default_reasoning_types = {
        "End-to-end": "",
        "CoT": "Let's think step by step. Output sequential logic."
    }
        # Class variable for default shot types
    default_shot_types = {
        "Zero-Shot": "",
        "Few-Shot": "Consider the following examples and maintain their formatting.",
        "One-Shot": "Consider the following example and maintain its formatting."
    }
    default_example_formats = {
        "End-to-end" : "Question: {question}\Output: {output}",
        "CoT":  "Question: {question}\Output: {output}"
    }

    default_num_shots = {
        "Few-Shot" : 3,
        "One-Shot" : 1,
        "Zero-Shot" : 0
    }

    def __init__(self, examples, task_description, example_format, num_shots = None, reasoning_type="End-to-end", shot_type="Zero-Shot"):
        self.examples = examples
        self.task_description = task_description
        self.reasoning_type = reasoning_type
        self.reasoning_type_description = self.default_reasoning_types[reasoning_type]
        self.shot_type = shot_type
        self.shot_type_description = self.default_shot_types[shot_type]
        self.task_description_with_reasoning_type = f"{self.task_description}\n{self.reasoning_type_description}\n{self.shot_type_description}"
        if example_format is None:
            self.example_format = self.default_example_formats[reasoning_type]
        else:
            self.example_format = example_format
        if num_shots is None:
            self.num_shots = self.default_num_shots[shot_type]
        else:
            self.num_shots = num_shots
        if self.shot_type == "Few-Shot":
            self.init_langchain()

    def init_langchain(self):
        # Initialize Langchain components
        embeddings = HuggingFaceEmbeddings()
        to_vectorize = []
        for example in self.examples:
            question = example["question"]
            output = example["output"]
            example_in_format = self.example_format.format(question=question, output=output)
            to_vectorize.append(example_in_format) 

        # vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=self.examples)
        # example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=self.num_shots)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
        # The list of examples available to select from.
        self.examples, 
        # The embedding class used to produce embeddings which are used to measure semantic similarity.
        HuggingFaceEmbeddings(), 
        # The VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma, 
        # The number of examples to produce.
        k=self.num_shots
        )
        
        example_prompt = LangchainPromptTemplate(
            input_variables=["question", "output"],
            template=self.example_format)

        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            # example_prompt=example_prompt,
            prefix=self.task_description_with_reasoning_type,
            # suffix="", 
            # input_variables=["claim", "context"]  # These variables are used in the prefix and suffix
            # examples=self.examples, 
            example_prompt=example_prompt, 
            suffix="Question {question}\nOutput: ", 
            input_variables=["question"]
        )
        return
    
    def fill_in(self, question):

        if self.shot_type == "Few-Shot" or self.shot_type == "One-Shot":
            filled_prompt = self.few_shot_prompt.format(question = question)
        else:
            filled_prompt = f"{self.task_description_with_reasoning_type}\nQuestion: {question}\nOutput: "
        return filled_prompt



# class Prompt:
#     # Class variables for default templates
#     default_input_types = {
#         "Standard": "",
#         "Batch": "You can answer questions. I will give you a few batches of exemplars in format Q[idx]:question, A[idx]:answer."
#     }

#     default_reasoning_types = {
#         "End-to-end": "",
#         "CoT": "Let's think step by step. Output sequential logic.",
#         "Program": "Generate programs to answer the question."
#     }

#     default_shot_types = {
#         "Zero-Shot": "",
#         "Few-Shot": "Consider the following examples and maintain their formatting.",
#         "One-Shot": "Consider the following example and maintain its formatting."
#     }

#     def __init__(self, input_type="Standard", reasoning_type="End-to-end", shot_type="Zero-Shot"):
#         self.input_type = input_type
#         self.reasoning_type = reasoning_type
#         self.shot_type = shot_type
#         self.update_template()

#     def update_template(self):
#         self.template = f"{Prompt.default_input_types[self.input_type]}\n{Prompt.default_reasoning_types[self.reasoning_type]}\n{Prompt.default_shot_types[self.shot_type]}\n{{data_examples}}"

#     def set_template_part(self, category, value):
#         if category == "input_type":
#             Prompt.default_input_types[self.input_type] = value
#         elif category == "reasoning_type":
#             Prompt.default_reasoning_types[self.reasoning_type] = value
#         elif category == "shot_type":
#             Prompt.default_shot_types[self.shot_type] = value
#         self.update_template()

#     def fill_in(self, data_examples):
#         filled_prompt = self.template.format(data_examples=data_examples)
#         return filled_prompt

# # Example usage
# p = Prompt(input_type="Batch", reasoning_type="CoT", shot_type="Few-Shot")
# print("Initial Template:", p.template)

# # Modify a specific part of the template
# p.set_template_part("reasoning_type", "Think sequentially.")
# print("Modified Template:", p.template)

# # Create a new instance with the same parameters
# p2 = Prompt(input_type="Batch", reasoning_type="CoT", shot_type="Few-Shot")
# print("New Instance Template:", p2.template)

# data_examples = {
#     "question": "What is 2 + 2?",
#     "answer": "4"
# }

# filled_prompt = p.fill_in(str(data_examples))
# print("Filled Prompt:", filled_prompt)




gsm8k_BatchPrompt_Original_prompt = '''
1st round input:
You can answer questions. I will give you a few batches of exemplars in format Q[idx]:question, A[idx]:answer.
Say "continue" to get the next input. Finally, a batch of test samples with only contexts Q[idx]:question are input.
Answer the test samples in format A[idx]:answer. Say "okay" if you understand.
1st round output:
okay.
2nd round input:
Q[1]: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
there will be 21 trees. How many trees did the grove workers plant today?
Q[2]: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Q[3]: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left
in total?
Q[4]: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops
did Jason give to Denny?
A[1]: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have
been 21 - 15 = 6. The answer is 6.
A[2]: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
A[3]: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35,
they had 74 - 35 = 39. The answer is 39.
A[4]: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12
= 8. The answer is 8.
2nd round output:
continue.
3rd round input:
Q[1]: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he
have now?
Q[2]: There were nine computers in the server room. Five more computers were installed each day, from monday
to thursday. How many computers are now in the server room?
Q[3]: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf
balls did he have at the end of wednesday?
Q[4]: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A[1]: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.
The answer is 9.
A[2]: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20
computers were added. 9 + 20 is 29. The answer is 29.
A[3]: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had
35 - 2 = 33 golf balls. The answer is 33.
A[4]: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left.
23 - 15 is 8. The answer is 8.
3rd round output:
continue.
4th round input:
Q[1]: A garden produced 237 potatoes, 60 fewer cucumbers and twice as many peppers than the cucumbers. How
many vegetables did the garden produce?
Q[2]: John’s cow weighs 400 pounds. It increased its weight to 1.5 times its starting weight. He is able to sell the cow
for $3 per pound. How much more is it worth after gaining the weight?
Q[3]: John writes 20 pages a day. How long will it take him to write 3 books that are 400 pages each?
Q[4]: James has a rainwater collection barrel. For each inch of rain he collects 15 gallons. On Monday it rained 4 inches
and on Tuesday it rained 3 inches. He can sell water for $1.2 per gallon. How much money did he make from selling
all the water?
A[1]: The garden produced 237 - 60 = 177 cucumbers. The garden produced 177 * 2 = 354 peppers. The garden
produced 237 + 177 + 354 = 768 vegetables. The answer is 768.
A[2]: The cow initially weighs 400 * 1.5 = 600 pounds. So it gained 600 - 400 = 200 pounds. It is worth 200 * 3 = 600
dollars more. The answer is 600.
A[3]: He wants to write 3 * 400 = 1200 pages. So it will take him 1200 / 20= 60 days The answer is 60.
A[4]: It rained 3 + 4 = 7 inches So he collected 7 * 15 = 105 gallons So he makes 105 * 1.2 = 126 from selling the
water. The answer is 126.
4th round output:
continue.
Test round input:
{four test questions}
Test round output:
{four test answers.}
'''
mnli_BatchPrompt_Original_prompt = '''Premise[1]: Conceptually cream skimming has two basic dimensions - product and geography.
Hypothesis[1]: Product and geography are what make cream skimming work.
Premise[2]: One of our number will carry out your instructions minutely.
Hypothesis[2]: A member of my team will execute your orders with immense precision.
Premise[3]: Analyzing Postal Service accounts for depreciation, fuel, and maintenance for
city delivery carriers, we have estimated the average city delivery vehicle cost per route.
Hypotheis[3]: Driving cost estimates can be averaged with sufficient data.
Premise[4]: Consider the United States Postal Service.
Hypothesis[4]: Forget the United States Postal Service.
Answer[1]: The answer is Neutral.
Answer[2]: The answer is True.
Answer[3]: The answer is Neutral.
Answer[4]: The answer is False.
Premise[1]: Take a remarkable statistic that Shesol cites but lets pass relatively unexamined.
Hypothesis[1]: They had data that was very relevant but under used.
Premise[2]: The man on the ground thinks for a moment and yells back, You must work in management.
Hypothesis[2]: There was no one on the ground, man or woman.
Premise[3]: Hello, Ben.
Hypothesis[3]: I ignored Ben.
Premise[4]: How can you prove it?
Hypothesis[4]: Can you tell me how to prove it?
Answer[1]: The answer is True.
Answer[2]: The answer is False.
Answer[3]: The answer is False.
Answer[4]: The answer is True.
Premise[1]: In the midst of this amazing amalgam of cultures is a passion for continuity.
Hypothesis[1]: A passion for continuity is not the most important of these cultures.
Premise[2]: Poirot, I exclaimed, with relief, and seizing him by both hands, I dragged him into the room.
Hypothesis[2]: Poirot was now back and I was sorry that he would take over what I now considered
my own investigation.
Premise[3]: There’s a uh a couple called um oh i’m going to forgot his name now uh Dirkson.
Hypothesis[3]: I can’t remember their name.
Premise[4]: It’s not that the questions they asked weren’t interesting or legitimate (though most did fall
under the category of already asked and answered).
Hypothesis[4]: All of the questions were interesting according to a focus group consulted on the subject.
Answer[1]: The answer is Neutral.
Answer[2]: The answer is False.
Answer[3]: The answer is True.
Answer[4]: The answer is Neutral'''

commensense_qa_BatchPrompt_Original_prompt = '''Q[1]: What do people use to absorb extra ink from a fountain pen?
Answer Choices[1]: (a) shirt pocket (b) calligrapher’s hand (c) inkwell (d) desk drawer (e) blotter
Q[2]: What home entertainment equipment requires cable?
Answer Choices[2]: (a) radio shack (b) substation (c) television (d) cabinet
Q[3]: The fox walked from the city into the forest, what was it looking for?
Answer Choices[3]: (a) pretty flowers (b) hen house (c) natural habitat (d) storybook
Q[4]: Sammy wanted to go to where the people were. Where might he go?
Answer Choices[4]: (a) populated areas (b) race track (c) desert (d) apartment (e) roadblock
A[1]: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to
absorb ink. So the answer is (e).
A[2]: The answer must require cable. Of the above choices, only television requires cable. So the answer
is (c).
A[3]: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest.
So the answer is (b).
A[4]: The answer must be a place with a lot of people. Of the above choices, only populated areas have a
lot of people. So the answer is (a).
Q[1]: Where do you put your grapes just before checking out?
Answer Choices[1]: (a) mouth (b) grocery cart (c)supermarket (d) fruit basket (e) fruit market
Q[2]: Google Maps and other highway and street GPS services have replaced what?
Answer Choices[2]: (a) united states (b) mexico (c) countryside (d) atlas
Q[3]: Before getting a divorce, what did the wife feel who was doing all the work?
Answer Choices[3]: (a) harder (b) anguish (c) bitterness (d) tears (e) sadness
Q[4]: James went to the tennis court that was located in his home what?
Answer Choices[4]: (a) country club (b) park (c) michigan (d) sports (e) town
A[1]: The answer should be the place where grocery items are placed before checking out. Of the above
choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).
A[2]: The answer must be something that used to do what Google Maps and GPS services do, which is to
give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).
A[3]: The answer should be the feeling of someone getting divorced who was doing all the work. Of the
above choices, the closest feeling is bitterness. So the answer is (c).
A[4]: The answer must be a place where tennis courts are located. Of the above choices, only home town
has tennis courts. So the answer is (e).
Q[1]: What does you body do when you exercise?
Answer Choices[1]: (a) need for food (b) thirst (c) work out (d) sweating (e) injury
Q[2]: In order to see a story on the big screen what must you do?
Answer Choices[2]: (a) go to movies (b) visualize (c) reading (d) open book (e) sketching a picture
Q[3]: He followed the train tracks hoping to get home, he had gotten lost in the Yooperland where?
Answer Choices[3]: (a) ghetto (b) michigan (c) new york (d) canada (e) train station
Q[4]: What would you get if you want a painting but cannot afford the original?
Answer Choices[4]: (a) reproduction (b) derivative (c) reproduction (d) simile (e) remake
A[1]: The answer must be something that happens when you exercise. Of the above choices, only sweating
happens when you exercise. So the answer is (d).
A[2]: The answer must be something that you do to see a story on the big screen. Of the above choices,
only going to movies makes sense. So the answer is (a).
A[3]: The answer should be a place that relates to Yooperland. Of the above choices, only michigan is
related to Yooperland. So the answer is (b).
A[4]: The answer must be something that is similar to the original. Of the above choices, only
reproduction is similar to the original. So the answer is (a).'''