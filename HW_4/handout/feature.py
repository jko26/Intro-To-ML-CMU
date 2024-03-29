import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt
GLOVE_MAP = None

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim_dataset(data): #data is an array of pairs
    #trimmed_data = np.zeros(shape=(data.shape[0], 2),dtype='l,O') #trimmed_data has same shape as dataset
    trimmed_data = []
    for i in range(data.shape[0]): #index through data
        review = data[i] #review is (integer label, string)
        trimmed_data.append([review[0]])
        trimmed_review = trim_review(review)
        trimmed_data[i].append(trimmed_review)
    return trimmed_data

def trim_review(review): 
    trimmed = [] #store as list to append items, then convert back to string at the end
    orig_review = review[1].split() #list of original words
    for word in orig_review:
        if word in GLOVE_MAP: #word is contained in the GloVe mapping, so keep it in the trimmed review
            trimmed.append(word)
    return trimmed

def feature_vec(trimmed_review):
    avg = np.zeros(shape=300)
    words = trimmed_review[1]
    for word in words:
        avg = avg + GLOVE_MAP[word]
    return avg/len(words)

def featurize(data, out_file_name):
    out_file = open(out_file_name, 'w')
    for review in data:
        out_file.write(str(review[0]) + '\t')
        for num in feature_vec(review):
            out_file.write(str(num) + '\t')
        out_file.write('\n')
    
if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    GLOVE_MAP = load_feature_dictionary(args.feature_dictionary_in)

    train_data = load_tsv_dataset(args.train_input) #train_data is an array of pairs
    train_trimmed = trim_dataset(train_data)
    val_data = load_tsv_dataset(args.validation_input) #train_data is an array of pairs
    val_trimmed = trim_dataset(val_data)
    test_data = load_tsv_dataset(args.test_input) #train_data is an array of pairs
    test_trimmed = trim_dataset(test_data)

    featurize(train_trimmed, args.train_out)
    featurize(val_trimmed, args.validation_out)
    featurize(test_trimmed, args.test_out)

