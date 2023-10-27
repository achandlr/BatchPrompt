from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import traceback

# This evaluation class should be able to handle everything except evaluation code for correctness.
class Evaluation:
    def __init__(self):
        pass

    def _validate_input(self, y_pred, y_true, answer_type):
        # Validate that the answer_type is one of the valid options
        valid_types = ['numerical', 'categorical', 'binary']
        assert answer_type in valid_types, f"Invalid answer_type: {answer_type}. Choose from {valid_types}."
        
        # Validate that the length of predictions and ground truths are the same
        assert len(y_pred) == len(y_true), "Prediction and ground truth lengths must match."

    def _calculate_accuracy(self, y_pred, y_true):
        # Calculate accuracy
        return accuracy_score(y_true, y_pred)

    def _calculate_f1(self, y_pred, y_true):
        # Calculate F1 score
        return f1_score(y_true, y_pred, average='weighted')

    def _calculate_sensitivity(self, y_pred, y_true):
        # Calculate sensitivity
        cm = confusion_matrix(y_true, y_pred)
        TP = cm[1, 1]
        FN = cm[1, 0]
        assert (TP + FN) != 0, "True Positive + False Negative cannot be zero."
        return TP / (TP + FN)

    def _calculate_confusion_elements(self, y_pred, y_true):
        # Calculate elements of the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        return {'TP': cm[1, 1], 'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0]}

    def get_stats(self, y_pred, y_true, answer_type):
        # Validate the input first
        self._validate_input(y_pred, y_true, answer_type)
        
        # Initialize the results dictionary
        results = {}
        
        # Convert input to numpy arrays for easier handling
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate and store accuracy
        results['Accuracy'] = self._calculate_accuracy(y_pred, y_true)
        
        # If the answer_type is binary or categorical, calculate the additional metrics
        if answer_type in ['binary', 'categorical']:
            results['F1'] = self._calculate_f1(y_pred, y_true)
            results['Sensitivity'] = self._calculate_sensitivity(y_pred, y_true)
            results.update(self._calculate_confusion_elements(y_pred, y_true))
        
        return results


class CodeEvaluator:
    def __init__(self):
        self.status = "UNTESTED"

    def run_code_and_tests(self, code_str, test_cases):
        # Split the code into lines to isolate import statements
        code_lines = code_str.strip().split('\n')
        import_lines = [line for line in code_lines if line.startswith('import') or line.startswith('from')]
        non_import_lines = [line for line in code_lines if not line.startswith('import') and not line.startswith('from')]

        # Execute any import lines and update globals
        for import_line in import_lines:
            try:
                exec(import_line, globals())
            except Exception as e:
                self.status = f"FAILURE (Import Error: {traceback.format_exc()})"
                return self.status

        # Reassemble the non-import lines and execute
        non_import_code_str = '\n'.join(non_import_lines)
        try:
            exec(non_import_code_str, globals())
            self.status = "RUNNING"
        except SyntaxError as se:
            self.status = f"FAILURE (Syntax Error: {traceback.format_exc()})"
            return self.status
        except Exception as e:
            self.status = f"FAILURE (Code Error: {traceback.format_exc()})"
            return self.status
        
        # Execute each test case individually
        for idx, test_case in enumerate(test_cases):
            try:
                exec(test_case, globals())
            except AssertionError:
                self.status = f"FAILURE (Test {idx + 1} failed)"
                return self.status
            except Exception as e:
                self.status = f"FAILURE (Test Error: {traceback.format_exc()})"
                return self.status

        # All test cases passed
        self.status = "PASS"
        return self.status
import re
from typing import List

def extract_answers_batch(output_str: str, answer_type = None) -> List[int]:
    # Initialize an empty list to store the extracted answers.
    answers = []
    
    # Step 1: Split the string by newlines to process each line individually.
    lines = output_str.strip().split("\n")
    
    # Primary regex pattern to extract number after ": ".
    primary_pattern = r": (\d+)"
    
    # Backup regex pattern to extract any number in the line.
    backup_pattern = r"(\d+)"
    
    if answer_type == "commonsense":
        raise None
    # Step 2: Loop through each line to extract the answer.
    for line in lines:
        # Try primary regex pattern first.
        match = re.search(primary_pattern, line)
        if match:
            answers.append(int(match.group(1)))
        else:
            # If primary fails, try the backup pattern.
            match = re.search(backup_pattern, line)
            if match:
                answers.append(int(match.group(1)))

    return answers



# Example Usage
if __name__ == "__main__":

    from src.utils.dataset_loader import load_pickle
    mbpp = load_pickle( "data\\imported\\datasets\\pickled\\mbpp")

    # mbpp_code_example = mbpp['train']['code'][1]
    # mbpp_test_cases_example = mbpp['train']['test_list'][1]

    evaluator = CodeEvaluator()
    # Example Code to run evaluator
    bad_indices = {326}
    for index in range(len(mbpp['train']['code'])):
        if index in bad_indices:
            continue
        # index = 326
        mbpp_code_example = mbpp['train']['code'][index]
        mbpp_test_cases_example = mbpp['train']['test_list'][index]
        result = evaluator.run_code_and_tests(mbpp_code_example, mbpp_test_cases_example)
        print(f"Code Evaluation Result: {result}")

    print("DONE")