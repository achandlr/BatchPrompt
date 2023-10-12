from sklearn.utils import shuffle

def balance_dataset(dataset, label_column, random_state=42):
    # Getting the minimum number of samples per class
    min_samples_per_class = dataset[label_column].value_counts().min()

    # Creating a balanced dataset by sampling min_samples_per_class from each label
    balanced_dataset = (dataset.groupby(label_column)
                               .apply(lambda x: x.sample(min_samples_per_class, random_state=random_state))
                               .reset_index(drop=True))

    # Shuffling the balanced dataset
    balanced_dataset = shuffle(balanced_dataset, random_state=random_state)
    
    return balanced_dataset