
from datasets import load_dataset
import pickle

def load_dataset(dataset_name):
    """
    Loads the dataset from the HuggingFace Hub
    """
    return load_dataset(dataset_name)

def pickle_dataset(dataset, dataset_path):
    """
    Pickles the dataset
    """
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return 

# Load the gsm8k dataset
# TODO: Confirm all below
gsm8k = load_dataset("gsm8k")
gsm_hard = load_dataset("reasoning-machines/gsm-hard")
commensense_qa = load_dataset("commonsense_qa")
human_eval = load_dataset("openai_humaneval")
mbpp = load_dataset("mbpp")
rte = load_dataset("glue", "rte")
# TODO: Confirm
mnli = rte = load_dataset("glue", "mnli")
# mnli = rte = load_dataset("glue", "mnli_matched")
# mnli = rte = load_dataset("glue", "mnli_mismatched")