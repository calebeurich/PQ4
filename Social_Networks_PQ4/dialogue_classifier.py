import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time

'''
This function should open a CSV file and extract its data into a dictionary structure. 
Each dictionary will have two keys: "person", which is paired with the name of the actor
 who is speaking, and "sentence", which is paired with a sentence that person has said.
 All dictionaries will be added to the list_of_dictionaries and that will be returned.
'''
def get_raw_training_data(filename):
    list_of_dictionaries = []
    with open(filename, 'r') as initial_data:
        for line in initial_data:
            line_data = line.split('","')
            line_dict = dict()
            line_dict['person'] = line_data[0][1:]
            line_dict['sentence'] = line_data[1][:-2].lower()
            list_of_dictionaries.append(line_dict)

    return list_of_dictionaries


'''
This method will return a list of all the stems from the list of words
given as a parameter. This method will make sure no 'bad symbols' are in the
word before it is turned into its stem because those 'bas symbols' will
cause the stemmer to work incorrectly.
'''
def preprocess_words(words, stemmer):
    bad_symbols = ['?', ',', '!', '.', ';', ':']
    stem_set = set()
    separated_words = words.split(' ')
    for word in separated_words:
        if any (x in word for x in bad_symbols):
            for letter in word:
                if letter in bad_symbols:
                    word = word.replace(letter, '')
        the_stem = stemmer.stem(word)
        stem_set.add(the_stem)

    return list(stem_set)

'''
This function takes the raw training data and stemmer, and organizes the data into words, classes and documents. 
Stemmed words from preprocessed words are added to words. Documents is appended with token words from the sentences and the person
who said that sentence. Classes is appended with each potential person. Finally words is converted to a set to avoid duplicates in 
the corpus. 
'''
def organize_raw_training_data(raw_training_data, stemmer):
    words = set()
    documents = []
    classes = []
    for data in raw_training_data:
        sentence = data['sentence']
        tokens = nltk.word_tokenize(sentence)
        preprocessed_words = preprocess_words(sentence, stemmer)
        for word in preprocessed_words:
            words.add(word)
        documents.append((tokens, data['person']))
        if data['person'] not in classes:
            classes.append(data['person'])

    return list(words), classes, documents


'''
This function creates a training data and output which is a list representation 
of which class each sentence belongs to. Training data is a list of lists. Each 
internal is built by appending a 1 when a stem word is in the sentence and a “0” 
when not. This produces a list of 1 and 0 that occurrence of stem words in sentences.
The output list is simply built by appending a list with a 1 at the index of the 
sentences class indices. There are three classes, and each list added is of length 
three. 
'''
def create_training_data(words, classes, documents):
    training_data = []
    output = []

    # for each line
    for i in range(len(documents)):
        bag_of_words = []
        sentence = documents[i][0]
        # for stem word in our words
        for word in words:
            # if that word is in our sentence
            if word in sentence:
                # add a 1
                bag_of_words.append(1)
            else:
                # add a 0
                bag_of_words.append(0)
        training_data.append(bag_of_words)



    for line in documents:
        if line[1] == classes[0]:
            output.append([1,0,0])
        if line[1] == classes[1]:
            output.append([0,1,0])
        if line[1] == classes[2]:
            output.append([0,0,1])

    return training_data, output

"""
The funtion represents the sigmoid funtion and returns the f(z) value for a given z value. 
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_output_to_derivative(output):
    """Convert the sigmoid function's output to its derivative."""
    return output * (1-output)


"""* * * TRAINING * * *"""
def init_synapses(X, hidden_neurons, classes):
    """Initializes our synapses (using random values)."""
    # Ensures we have a "consistent" randomness for convenience.
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    return synapse_0, synapse_1


def feedforward(X, synapse_0, synapse_1):
    """Feed forward through layers 0, 1, and 2."""
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    return layer_0, layer_1, layer_2


def get_synapses(epochs, X, y, alpha, synapse_0, synapse_1):
    """Update our weights for each epoch."""
    # Initializations.
    last_mean_error = 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    synapse_0_direction_count = np.zeros_like(synapse_0)

    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    # Make an iterator out of the number of epochs we requested.
    for j in iter(range(epochs+1)):
        layer_0, layer_1, layer_2 = feedforward(X, synapse_0, synapse_1)

        # How much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # If this 10k iteration's error is greater than the last iteration,
            # break out.
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        # In what direction is the target value?  How much is the change for layer_2?
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # How much did each l1 value contribute to the l2 error (according to the weights)?
        # (Note: .T means transpose and can be accessed via numpy!)
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # In what direction is the target l1?  How much is the change for layer_1?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # Manage updates.
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if j > 0:
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    return synapse_0, synapse_1


def save_synapses(filename, words, classes, synapse_0, synapse_1):
    """Save our weights as a JSON file for later use."""
    now = datetime.datetime.now()

    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("Saved synapses to:", synapse_file)


def train(X, y, words, classes, hidden_neurons=10, alpha=1, epochs=50000):
    """Train using specified parameters."""
    print("Training with {0} neurons and alpha = {1}".format(hidden_neurons, alpha))

    synapse_0, synapse_1 = init_synapses(X, hidden_neurons, classes)

    # For each epoch, update our weights
    synapse_0, synapse_1 = get_synapses(epochs, X, y, alpha, synapse_0, synapse_1)

    # Save our work
    save_synapses("synapses.json", words, classes, synapse_0, synapse_1)


def start_training(words, classes, training_data, output):
    """Initialize training process and keep track of processing time."""
    start_time = time.time()
    X = np.array(training_data)
    y = np.array(output)

    train(X, y, words, classes, hidden_neurons=20, alpha=0.1, epochs=100000)

    elapsed_time = time.time() - start_time
    print("Processing time:", elapsed_time, "seconds")



"""* * * CLASSIFICATION * * *"""

def bow(sentence, words):
    """Return bag of words for a sentence."""
    stemmer = LancasterStemmer()

    # Break each sentence into tokens and stem each token.
    sentence_words = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence)]

    # Create the bag of words.
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return (np.array(bag))


def get_output_layer(words, sentence):
    """Open our saved weights from training and use them to predict based on
    our bag of words for the new sentence to classify."""

    # Load calculated weights.
    synapse_file = 'synapses.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])

    # Retrieve our bag of words for the sentence.
    x = bow(sentence.lower(), words)
    # This is our input layer (which is simply our bag of words for the sentence).
    l0 = x
    # Perform matrix multiplication of input and hidden layer.
    l1 = sigmoid(np.dot(l0, synapse_0))
    # Create the output layer.
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def classify(words, classes, sentence):
    """Classifies a sentence by examining known words and classes and loading our calculated weights (synapse values)."""
    error_threshold = 0.2
    results = get_output_layer(words, sentence)
    results = [[i,r] for i,r in enumerate(results) if r>error_threshold ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print("\nSentence to classify: {0}\nClassification: {1}".format(sentence, return_results))
    return return_results




def main():
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    stemmer = LancasterStemmer()
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(words, classes, documents)

    # Comment this out if you have already trained once and don't want to re-train.
    start_training(words, classes, training_data, output)

    # Classify new sentences.
    classify(words, classes, "will you look into the mirror?")
    classify(words, classes, "mithril, as light as a feather, and as hard as dragon scales.")
    classify(words, classes, "the thieves!")


if __name__ == "__main__":
    main()
