import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
import numpy as np
from google.colab import drive
from nltk.tokenize import word_tokenize
from datasets import load_dataset, load_metric, Dataset, load_from_disk
import nltk
drive.mount('/content/drive')
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Function to load and preprocess data
def load_and_preprocess_data(filepath = 'jigsaw-unintended-bias-in-toxicity-classification/train.csv'):
    """
    Load and preprocess the dataset.
    
    Parameters:
    - filepath: str, path to the dataset file.
    
    Returns:
    - preprocessed data ready for model input.
    """
    # Load data
    data = pd.read_csv(filepath, encoding='unicode_escape')
    # group by identity groups
    groups = data.columns[8:32]
    data_g = []
    max = 0
    for i in range(len(groups)):
        data_g.append(data[data[groups[i]] >0.1])
        if data[data[groups[i]] >0.1].shape[0] >max:
            max = data[data[groups[i]] >0.1].shape[0]
    # add no group (no hate) as an extra label
    condition = (data[groups[0]] == 0)
    for group in groups[1:]:
        condition &= (data[group] == 0)

    data_g.append(data[condition].sample(max))
    # find target label and add comment_texts to a list
    X = []
    Y = []
    for i, group in enumerate(data_g):
        X.append(group.comment_text)#.sample(min))
        Y.append(i*np.ones(group.shape[0]))
    data = pd.DataFrame(pd.concat(X))
    Y = [y for g in Y for y in g]
    data.insert(1,'y', np.array(Y), True)
    # insert every sentence into word category list and add to dataframe
    words =[]
    for i, sen in enumerate(data.comment_text):
        words.append(sen.split())
    data['words'] = words

    return data

# Function to build the  Bert model.
def build_model(output_units):
    """
    Build and compile the  Bert model..
    
    Parameters:
    - input_shape: tuple, shape of the input data.
    - output_units: int, number of output units/classes.
    
    Returns:
    - Bert model.
    """
    # remove access token later!!
    access_token = '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'
    model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=output_units, token = access_token)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Function to train the model
def train_model(model, train_dataset, test_dataset, epochs=5, batch_size=20):
    """
    Train the model and save it.
    
    Parameters:
    - model: compiled model.
    - X_train, y_train: training data and labels.
    - X_val, y_val: validation data and labels.
    - epochs: int, number of epochs to train.
    - batch_size: int, batch size for training.
    
    Returns:
    - History object containing training information.
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory for models and checkpoints
        num_train_epochs=epochs,              # Total number of training epochs
        per_device_train_batch_size=batch_size,   # Batch size for training
        per_device_eval_batch_size=batch_size,    # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        eval_steps =0.05,
        save_steps = 0.05,
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for logs
        logging_steps=10,                # Log every X updates steps
        evaluation_strategy="steps",     # Evaluate after every epoch
        save_strategy="steps",# Save model after every epoch
        load_best_model_at_end=True,     # Load the best model when finished training
    )

    # Define the trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    )
    # Train the model
    trainer.train()
    model.save_pretrained('hate_speech_model.h5')  # Save the model
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    - model: trained model.
    - X_test, y_test: test data and labels.
    
    Returns:
    - Evaluation results.
    """
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
    return results

# Function for making predictions
def make_prediction(model, text):
    """
    Make a prediction with the model.
    
    Parameters:
    - model: trained model.
    - text: str, input text for prediction.
    
    Returns:
    - Prediction result.
    """
    # Process text...
    prediction = model.predict(text)
    return prediction

def tokenize_function(prmpt, max_length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(prmpt["text"], padding="max_length", truncation=True, max_length = max_length)

def one_hot_encode_example(example):
    # This function will be applied to each example in the dataset
    one_hot_encoded = np.eye(25)[int(example['labels'])]
    example['labels'] = one_hot_encoded.tolist()  # Convert numpy array to list
    return example

def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    # Convert logits to predictions (assuming logits are outputs from a softmax or similar function)
    predictions = np.argmax(logits, axis=-1)
    # Convert one-hot encoded labels back to integer class labels
    true_labels = np.argmax(labels, axis=-1)
    # Compute accuracy
    return accuracy_metric.compute(predictions=predictions, references=true_labels)

# Main function to run the pipeline
def main():
    # Use jigsaw
    filepath = 'jigsaw-unintended-bias-in-toxicity-classification/train.csv'
    
    # Load and preprocess data
    data = load_and_preprocess_data(filepath)
    # calculate max length of words and transform to a dictionary
    max_length = list(set(w for x in data.words for w in x ))
    dataset_dict = {
    "text": data['comment_text'].tolist(),
    "labels": data['y'].tolist()  # Ensure this is just a list of integers, not one-hot encoded
    }
    dataset  = Dataset.from_dict(dataset_dict)
    
    # tokenize the datasets
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, max_length), batched=True)
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)  # Adjust test_size as needed
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
  
    # Apply one hot encoding to the labels
    train_dataset = train_dataset.map(one_hot_encode_example)
    test_dataset = test_dataset.map(one_hot_encode_example)
    # Build the model
    model = build_model(25)
    
    # Train the model
    train_model(model, train_dataset, test_dataset, epochs=5, batch_size=20)

    # Evaluate the model
    evaluate_model(model, test_dataset)

    # Make predictions
    text = "I hate sushi"
    prediction = make_prediction(model, text)

if __name__ == "__main__":
    main()
