import sys
import pandas as pd
import os

v = -1

def train(d):
    global v 
    v = majority_vote(d)

def majority_vote(d):
    pos_count = 0
    last_col = d.columns[-1]
    labels = d[last_col]
    for label in labels:
        pos_count = pos_count + label
    return int(pos_count >= len(labels) - pos_count)


# h(x) returns the hypothesized function that best explains the data
def h(x): 
    global v 
    return v 

def predict(d, out_file_name):
    predictions = []
    out_file = open(os.path.join('/Users','jko26','Intro-To-ML-CMU', 'HW_1', out_file_name), "w")
    for index, row in d.iterrows():
        predictions.append(h(d.iloc[index]))
        out_file.write(str(h(d.iloc[index])) + '\n')
    return predictions

def predict(d):
    predictions = []
    for index, row in d.iterrows():
        predictions.append(h(d.iloc[index]))
    return predictions

def error(pred, labels):
    error_count = 0
    for i in range(len(labels)):
        error_count = error_count + int(pred[i] != labels[i])
    return error_count / len(labels)

if __name__ == '__main__':
    train_out = sys.argv[3] # path of output .txt 
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    train_input = pd.read_csv(os.path.join('/Users','jko26','Intro-To-ML-CMU', 'HW_1', 'HW_1_Data', sys.argv[1]), sep='\t')
    test_input = pd.read_csv(os.path.join('/Users','jko26','Intro-To-ML-CMU', 'HW_1', 'HW_1_Data', sys.argv[2]), sep='\t')

    train(train_input)
    train_pred = predict(train_input, train_out)
    test_pred = predict(test_input, test_out)

    train_error = error(train_pred, train_input["heart_disease"])
    test_error = error(test_pred, test_input["heart_disease"])
    metrics_file = open(os.path.join('/Users','jko26','Intro-To-ML-CMU', 'HW_1', metrics_out), "w")
    metrics_file.write(str(train_error) + '\n')
    metrics_file.write(str(test_error) + '\n')






