
from datasets import load_dataset
import pickle

# def load_dataset(dataset_name):
#     """
#     Loads the dataset from the HuggingFace Hub
#     """
#     return load_dataset(dataset_name)

def pickle_dataset(dataset, dataset_name):
    """
    Pickles the dataset
    """
    with open("data\\imported\\datasets\\pickled\\"+dataset_name, 'wb') as f:
        pickle.dump(dataset, f)
    return 

# Load the gsm8k dataset
# TODO: Confirm all below
gsm8k = load_dataset("gsm8k", "main")  # well organized, easy to use
gsm_hard = load_dataset("reasoning-machines/gsm-hard") # well organized, easy to use
commensense_qa = load_dataset("commonsense_qa") # train and validation hold gold truth labels, test does not
human_eval = load_dataset("openai_humaneval") # test cases in human_eval["test"]["test"], but hard to parse and would be fair bit of work
mbpp = load_dataset("mbpp") # test cases well organized as list for each code generation in mbpp["train"]['test_list']
rte = load_dataset("glue", "rte") # well organized, easy to use, ignore test because no ground truth
mnli = load_dataset("glue", "mnli")

pickle_dataset(gsm8k, "gsm8k")
pickle_dataset(gsm8k, "gsm-hard")
pickle_dataset(gsm8k, "commonsense_qa")
pickle_dataset(gsm8k, "openai_humaneval")
pickle_dataset(gsm8k, "mbpp")
pickle_dataset(gsm8k, "rte")
pickle_dataset(gsm8k, "mnli")
print("DONE")
# mnli = rte = load_dataset("glue", "mnli_matched")
# mnli = rte = load_dataset("glue", "mnli_mismatched")