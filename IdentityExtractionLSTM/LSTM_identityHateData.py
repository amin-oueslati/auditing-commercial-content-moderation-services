# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import nltk
from datasets import load_dataset, load_metric, Dataset, load_from_disk
nltk.download('punkt')
from tqdm import tqdm
from numpy.random import seed
import os

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

    return data


# Function to build the Bi-LSTM model
def build_model(input_dim, output_dim, max_seq_length, max_word_length, n_target_groups, embedding_matrix):
    """
    Build and compile the Bi-LSTM model.

    Parameters:
    - input_shape: tuple, shape of the input data.
    - output_units: int, number of output units/classes.

    Returns:
    - Compiled Bi-LSTM model.
    """
   #  Word-level input
    word_input = Input(shape=(max_seq_length,))
    #word_embedding = Embedding(input_dim=input_dim,output_dim = output_dim, weights = [embedding_matrix])(word_input)
    word_embedding = Embedding(input_dim=input_dim, output_dim=output_dim, trainable=False)(word_input) # D: I'll set the weight further down. Also I added trainable=False, because we're using pre-trained embeddings

    word_lstm = Bidirectional(LSTM(units=output_dim,dropout=0.2, recurrent_dropout=0.2),merge_mode = 'concat')(word_embedding)
    #lstm2 = LSTM(units=output_dim, dropout=0.5, recurrent_dropout=0.5)(word_lstm)

    #pooled_output = GlobalMaxPooling1D()()
    # TimeDistributed Layer for output
    output = Dense(n_target_groups, activation="softmax")(word_lstm)
    # Define model
    model = Model(inputs=[word_input], outputs=output)

    # D: Set pre-trained weights here, because it was giving me an issue with the latest Keras version
    model.layers[1].set_weights([embedding_matrix])

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    # Compile model
    model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1), optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model
    m

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
    print('Train Model')
    loss = list()
    # fit model for one epoch on this sequence
    hist = model.fit(train, batch_size=batch_size, verbose=1, epochs=epochs, validation_data =val)
    print('2')
    loss.append(hist.history['loss'][0])
    model.save(f'model_240301.keras')
    return loss, hist, model

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

def load_embeddings(file_path = 'glove.42B.300d.txt'):
    embedding_vector = {}
    f = open(file_path, encoding = 'utf-8')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:],dtype = 'float32')
        embedding_vector[word] = coef
    return embedding_vector

def tokenize_words_per_sentence(words):

    tokenizer_word = Tokenizer()
    vocab = list(set(w for x in words for w in x ))
    tokenizer_word.fit_on_texts(vocab)
    tokenized_sentences = []
    maxlen = max([len(s) for s in words])
    for words_list in words:
      w_tok = tokenizer_word.texts_to_sequences([words_list])
      pad_tok = pad_sequences(w_tok, maxlen = maxlen, padding='post', truncating='post')
      tokenized_sentences.append(pad_tok)
    return tokenized_sentences, tokenizer_word

def tokenize(sentences, words, max_seq_length, max_word_length):
    # Tokenize words

    X_word, tokenizer_word = tokenize_words_per_sentence(words)
    # tokenizer BERT
    # Tokenize sentences
    tokenizer_sent = Tokenizer()
    tokenizer_sent.fit_on_texts(sentences)
    sequences_sent = tokenizer_sent.texts_to_sequences(sentences)
    X_sent = pad_sequences(sequences_sent, maxlen=max_seq_length, padding='post', truncating='post')

    return X_word, tokenizer_sent, tokenizer_word

def get_pad_train_test_val(data, max_seq_length, max_word_length):
    # Assuming you have 'comment_text', 'Sent_idx', and 'Label_idx' columns in your DataFrame 'data'

    # Tokenize sentences and words, and get maximum token and tag length
    X_word,_,_ = tokenize(data.comment_text, data.words, max_seq_length, max_word_length)

    X_word = np.array(X_word)
    X_word = X_word.reshape(X_word.shape[0], -1)
    # Pad Tags (y var) and convert it into one hot encoding
    n_tags = int(np.max(data['y']))+1
    tags = np.array(data['y'].tolist())
    tags = np.array([to_categorical(i, num_classes=n_tags) for i in tags])

    # Split train, test, and validation set
    X_train_word, X_test_word, y_train, y_test = train_test_split(X_word, tags, test_size=0.1, train_size=0.9, random_state=2020)
   # X_train_sent, X_val_sent, X_train_word, X_val_word, y_train, y_val = train_test_split(X_train_sent, X_train_word, y_train, test_size=0.25, train_size=0.75, random_state=2020)

    print(
        'Train data:', X_train_word.shape, y_train.shape,
        '\nTest data:', X_test_word.shape, y_test.shape,
    )

    return X_train_word, y_train, X_test_word, y_test

def tokenize_function(prmpt, tokenizer):
   # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(prmpt["text"], padding="max_length", truncation=True, max_length = 512)

def one_hot_encode_example(example):
    # This function will be applied to each example in the dataset
    one_hot_encoded = np.eye(10)[int(example['labels'])]
    example['labels'] = one_hot_encoded.tolist()  # Convert numpy array to list
    return example

# Main function to run the pipeline
def main():

    filepath = 'identity_hate_corpora.jsonl'
    model_file_path = 'trained_model.keras'

    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}")
        model = tf.keras.models.load_model(model_file_path)
    else:
        data = load_and_preprocess_data(filepath)
        print('Data loaded and preprocessed')
        embedding_vector = load_embeddings('glove.42B.300d.txt')
        print('Embeddings loaded')

        data['comment_text'] = data['comment_text'].fillna('')  # D: because I got an error
        data['comment_text'] = data['comment_text'].astype(str)  # D: because I got an error

        #max_length = list(set(w for x in data.words for w in x ))
        dataset_dict = {
        "text": data['comment_text'].tolist(),
        "labels": data['y'].tolist()  # Ensure this is just a list of integers, not one-hot encoded
        }
        dataset  = Dataset.from_dict(dataset_dict)

        # tokenize the datasets
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
                batch_size=20,
                shuffle=True
                )
        tf_test_ds = test_dataset.to_tf_dataset(
                columns=["input_ids"],
                label_cols=["labels"],
                batch_size=20,
                shuffle=True
                )

        # embedding matrix
        vocab_size = len(tokenizer.vocab)
        vocab = tokenizer.get_vocab()
        embedding_matrix = np.zeros((vocab_size, 300)) # changed D: this needs to be 300, because the Glove Embeddings have a size of 300
        for word,i in tqdm(vocab.items()):
            embedding_value = embedding_vector.get(word)
            if embedding_value is not None:
                embedding_matrix[i] = embedding_value

        input_dim = vocab_size
        output_dim = 300
        input_length = 209
        #n_tags = y_train.shape[1]
        print('input_dim: ', input_dim, '\noutput_dim: ', output_dim, '\ninput_length: ', input_length)#, '\nn_tags: ', n_tags)

        # Build the model
        #model = build_model(input_dim, output_dim, max_seq_length, max_word_length, n_tags)
        model = build_model(input_dim, output_dim, 512, 100, 25, embedding_matrix)

        # Train the model
        loss, hist, model = train_model(model, tf_train_ds, val = tf_test_ds, epochs=5, batch_size=20)
        model.save(model_file_path)

    results = evaluate_model(model, tf_test_ds)
    print(results)

    test_sentence = 'I hate sushi'
    prediction = make_prediction(model, test_sentence)
    print(prediction)


if __name__ == "__main__":
    main()
