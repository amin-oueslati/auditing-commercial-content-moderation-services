# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, SpatialDropout1D, add, concatenate
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import keras
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import TFBertModel, BertModel
from nltk.tokenize import word_tokenize
import nltk
from datasets import load_dataset, load_metric, Dataset, load_from_disk
nltk.download('punkt')
from tqdm import tqdm
from numpy.random import seed
import os
import platform
from tqdm.keras import TqdmCallback
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
MAX_LEN = 512
tokenizer = None
model = None
groups = None

bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.

    Parameters:
    - filepath: str, path to the dataset file.

    Returns:
    - preprocessed data ready for model input.
    """
    # Load data
    data = pd.read_json(filepath, lines=True)
    # group by identity groups
    data = data[data.hate == True]
    data_g = []
    for g in data.grouping.unique():
      data_g.append(data[data.grouping == g])
    X = []
    Y = []
    for i, group in enumerate(data_g):
        X.append(group.text)#.sample(min))
        Y.append(i*np.ones(group.shape[0]))
    data = pd.DataFrame(pd.concat(X))
    Y = [y for g in Y for y in g]
    data.insert(1,'y', np.array(Y), True)
    # insert every sentence into word category list and add to dataframe
    words =[]
    for i, sen in enumerate(data.text):
        words.append(sen.split())
    data['words'] = words

    return data, groups

def get_bert_embed_matrix():
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat
def build_bert_lstm_model(output_dim, max_seq_length, max_word_length, n_target_groups, embedding_matrix):
    """
    Build and compile a model that uses BERT for embeddings and LSTM for sequence modeling.

    Parameters:
    - max_seq_length: int, the length of the input sequences.
    - n_target_groups: int, the number of target groups/classes.

    Returns:
    - Compiled model with BERT embeddings and a Bi-LSTM layer.
    """
    # Define input layer
    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")

    # Get BERT embeddings
    bert_embeddings = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_ids)#bert_model(input_ids)[0]  # We use the first output (last hidden state)
    dropout = SpatialDropout1D(0.3)(bert_embeddings)
    # LSTM layer
    lstm_layer = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, name="LSTM_Layer"), name="LSTM_Bi_Layer")(dropout)
    lstm_layer2 = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, name="LSTM_Layer_2"), name="LSTM_Bi_Layer_2")(lstm_layer)
   # x = Dropout(rate=0.2)(lstm_layer2)

    hidden = concatenate([GlobalMaxPooling1D()(lstm_layer2),GlobalAveragePooling1D()(lstm_layer2),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    # Output layer
    output = Dense(n_target_groups, activation="softmax", name="Output")(hidden)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)
    # Build and compile model
    model = Model(inputs=[input_ids], outputs=output)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(label_smoothing=0.2), metrics=['accuracy'])
    model.summary()
    return model


# Function to train the model
def train_model(model, train, val, epochs=5, batch_size=32):
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
    loss = list()
    # fit model for one epoch on this sequence
    print("Fitting model...")
    hist = model.fit(train, batch_size=batch_size, verbose=1, epochs=epochs, validation_data=val)
    loss.append(hist.history['loss'][0])

    return loss, hist, model

# Function to evaluate the model
def evaluate_model(model, test):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: trained model.
    - X_test, y_test: test data and labels.

    Returns:
    - Evaluation results.
    """
    results = model.evaluate(test)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
    return results

# Function for making predictions
def make_prediction(model, text, tokenizer, max_seq_length):
    """
    Make a prediction with the model.

    Parameters:
    - model: trained model.
    - text: str, input text for prediction.
    - tokenizer: Tokenizer, the tokenizer used for training.
    - max_seq_length: int, the length to which sequences should be padded.

    Returns:
    - Prediction result.
    """
    # Tokenize and convert to input format
    seq = tokenizer.texts_to_sequences([text])

    # Make prediction
    prediction = model.predict(padded_seq)
    return prediction



def tokenize_function(prmpt, tokenizer):
   # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(prmpt["text"], padding="max_length", truncation=True, max_length = MAX_LEN)

def one_hot_encode_example(example):
    # This function will be applied to each example in the dataset
    one_hot_encoded = np.eye(10)[int(example['labels'])]
    example['labels'] = one_hot_encoded.tolist()  # Convert numpy array to list
    return example

def predict_proba(texts):
    # Ensure input is in a list format
    if isinstance(texts, str):
        texts = [texts]

    # Initialize an empty array for predictions
    predictions = np.zeros((len(texts), len(groups)+1))

    for i, text in enumerate(texts):
        tokens = tokenizer.encode_plus(text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="tf")
        input_ids = tokens["input_ids"]
        pred = model.predict(input_ids)
        predictions[i] = pred

    return predictions



def lime_eval(text, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=10, num_samples=10) # num_features is a max value

    print(exp.as_list())
    predict_proba(text)
    print("---")

# Main function to run the pipeline
def main():

    global tokenizer, model, groups

    filepath = 'identity_hate_corpora.jsonl'
    model_file_path = './trained_model_identity_hate_corpora.keras'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data, groups = load_and_preprocess_data(filepath)

    if os.path.exists(model_file_path):
        print(f"Loading model weights from {model_file_path}")
        #model = keras.models.load_model(model_file_path)
        model = tf.keras.models.load_model(model_file_path)
    else:
        print('Data loaded and preprocessed')
        #embedding_vector = load_embeddings('glove.42B.300d.txt')
        #print('Embeddings loaded')

        data['text'] = data['text'].fillna('')  # D: becausepip install tensorflow==2.15.0.post1 I got an error
        data['text'] = data['text'].astype(str)  # D: because I got an error

        #max_length = list(set(w for x in data.words for w in x ))
        dataset_dict = {
        "text": data['text'].tolist(),
        "labels": data['y'].tolist()  # Ensure this is just a list of integers, not one-hot encoded
        }
        dataset = Dataset.from_dict(dataset_dict)

        # tokenize the datasets
        tokenized_datasets = dataset.map(lambda x: tokenize_function(x ,tokenizer), batched=True)
        train_test_split = tokenized_datasets.train_test_split(test_size=0.2)  # Adjust test_size as needed
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        # Apply one hot encoding to the labels
        train_dataset = train_dataset.map(one_hot_encode_example)
        test_dataset = test_dataset.map(one_hot_encode_example)

        # tranform to tf
        tf_train_ds = train_dataset.to_tf_dataset(
                columns=["input_ids"],
                label_cols=["labels"],
                batch_size=32,
                shuffle=True
                )
        tf_test_ds = test_dataset.to_tf_dataset(
                columns=["input_ids"],
                label_cols=["labels"],
                batch_size=32,
                shuffle=True
                )

        # embedding matrix
        '''vocab_size = len(tokenizer.vocab)
        vocab = tokenizer.get_vocab()
        embedding_matrix = np.zeros((vocab_size, 300)) # changed D: this needs to be 300, because the Glove Embeddings have a size of 300
        for word,i in tqdm(vocab.items()):
            embedding_value = embedding_vector.get(word)
            if embedding_value is not None:
                embedding_matrix[i] = embedding_value

        input_dim = vocab_size
        output_dim = 300'''

        # Build the model
        #model = build_model(input_dim, output_dim, max_seq_length, max_word_length, n_tags)
        #model = build_model(input_dim, output_dim, 512, 100, 25, embedding_matrix)
        model = build_bert_lstm_model(32, MAX_LEN, 100, 10, get_bert_embed_matrix)

        # Train the model
        loss, hist, model = train_model(model, tf_train_ds, val = tf_test_ds, epochs=1, batch_size=32)
        model.save(model_file_path, save_format='tf')

        results = evaluate_model(model, tf_test_ds)
        print(results)

    test_sentence = 'I hate blacks'
    tokens = tokenizer.encode_plus(test_sentence, max_length=512, truncation=True, padding="max_length", return_tensors="tf")
    input_ids = tokens["input_ids"]
    prediction = model.predict(input_ids)
    print(prediction)
    print(groups)
    prediction = groups[np.argmax(prediction)]
    print(prediction)

    lime_eval(text=test_sentence, class_names=groups)


if __name__ == "__main__":
    main()
