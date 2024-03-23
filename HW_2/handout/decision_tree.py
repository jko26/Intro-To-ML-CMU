import argparse
import sys

from inspection import entropy
import pandas as pd

sys.path.insert(0, '/Users/jko26/Intro-To-ML-CMU/HW_1')
from majority_vote import majority_vote, error


possible_splits = None
root = None

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.split_attr = None
        self.vote = None
        self.depth = None
        self.hd_num = None
        self.healthy_num = None
    
def print_tree(Node, out_file_name):
    out_file = open(out_file_name, 'w')
    out_file.write("[" + str(get_zero_counts(train_df, train_df.columns[-1])) + " 0/" + str(get_one_counts(train_df, train_df.columns[-1])) + " 1]\n")
    print_tree_recurse(Node, out_file)

def print_tree_recurse(Node, out_file_name):
    if Node is None or Node.split_attr is None:
        return
    for i in range(Node.depth + 1):
        out_file_name.write("| ")

    out_file_name.write(str(Node.split_attr) + " = 0: ")
    out_file_name.write("[" + str(Node.left.healthy_num) + " 0/" + str(Node.left.hd_num) + " 1]\n")
    print_tree_recurse(Node.left, out_file_name)

    for i in range(Node.depth + 1):
        out_file_name.write("| ")

    out_file_name.write(Node.split_attr + " = 1: ")
    out_file_name.write("[" + str(Node.right.healthy_num) + " 0/" + str(Node.right.hd_num) + " 1]\n")
    print_tree_recurse(Node.right, out_file_name)

#predict a dataframe d of multiple instances
def predict(d, out_file_name):
    predictions = []
    out_file = open(out_file_name, "w")
    for index, row in d.iterrows():
        pred = h(row)
        out_file.write(str(pred) + '\n')
        predictions.append(pred)
    return predictions

#apply our hypothesis tree to a single instance
def h(x):
    cur_node = root
    while True:
        if (cur_node.left is None and cur_node.right is None): #node is a leaf, so simply use majority vote
            return cur_node.label
        if x[cur_node.split_attr] == 0:
            cur_node = cur_node.left
        else:
            cur_node = cur_node.right

def mutual_info(d, attribute):
    h_y = entropy(d)
    h_y_ones = entropy(get_one_rows(d, attribute))
    h_y_zeros = entropy(get_zero_rows(d, attribute))
    if (get_one_counts(d, attribute) == d.shape[0] or get_one_counts(d,attribute) == 0): #mutual info is 0 since all data points have the same value for that attribute
        return 0
    if (attribute == 'thalassemia'):
        print(" h_y is: " + str(h_y))
        print(" h_y_ones is: " + str(h_y_ones))
        print(" h_y_zeros is: " + str(h_y_zeros))
    return h_y - get_one_counts(d, attribute)/d.shape[0]*h_y_ones - get_zero_counts(d, attribute)/d.shape[0]*h_y_zeros

def train(d):
    global root
    root = tree_recurse(d, 0)

def tree_recurse(d, depth):
    q = Node()
    q.depth = depth

    q.hd_num = get_one_counts(d, d.columns[-1])
    q.healthy_num = get_zero_counts(d, d.columns[-1])
    
    #Base case:
    if ((get_one_counts(d, d.columns[-1]) == d.shape[0] or get_one_counts(d, d.columns[-1]) == 0) or 
    len(possible_splits) == 0 or 
    d.shape[0] == 0 or 
    depth >= args.max_depth):
        q.label = majority_vote(d)
        return q

    split_attribute = get_best_split(d)

    #Another base case:
    if (split_attribute is None): #maximum mutual information is 0, so there's no use in splitting
        q.label = majority_vote(d)
        return q
    
    #Recursive case:
    q.split_attr = split_attribute
    q.left = tree_recurse(get_zero_rows(d, split_attribute), depth + 1)
    q.right = tree_recurse(get_one_rows(d, split_attribute), depth + 1)

    return q

def get_best_split(d):
    max_mutual_info = 0
    max_info_attribute = None
    for attribute in possible_splits:
        #print("Chest pain MI: " + str(mutual_info(d, 'chest_pain')) + " Thalassemia MI: " + str(mutual_info(d, 'thalassemia')))
        if (mutual_info(d, attribute) > max_mutual_info):
            max_mutual_info = mutual_info(d, attribute)
            max_info_attribute = attribute
    if max_mutual_info == 0:
        return None
    #print("Best split is " + str(max_info_attribute) + " with mutual info " + str(max_mutual_info))
    return max_info_attribute

def get_one_rows(d, attribute):
    return d.loc[d[attribute] == 1]

def get_zero_rows(d, attribute):
    return d.loc[d[attribute] == 0]

def get_one_counts(d, attribute):
    return d[attribute].sum()

def get_zero_counts(d, attribute):
    return len(d[attribute]) - d[attribute].sum()

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_input, sep='\t')
    test_df = pd.read_csv(args.test_input, sep='\t')

    possible_splits = train_df.columns[:-1]
    train(train_df)
    train_pred = predict(train_df, args.train_out)
    test_pred = predict(test_df, args.test_out)

    train_error = error(train_pred, train_df[train_df.columns[-1]])
    test_error = error(test_pred, test_df[train_df.columns[-1]])
    metrics_out_file = open(args.metrics_out, "w")
    metrics_out_file.write("error(train): " + str(train_error) + '\n')
    metrics_out_file.write("error(test): " + str(test_error))
    print_tree(root, args.print_out)


    
    #Here's an example of how to use argparse
    #print_out = args.print_out

    #Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
