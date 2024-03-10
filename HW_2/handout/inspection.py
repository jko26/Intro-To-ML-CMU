import sys
import os
import pandas as pd
import math

sys.path.insert(0, '/Users/jko26/Intro-To-ML-CMU/HW_1')

from majority_vote import error, train, predict


def entropy(d):
    labels = d['heart_disease']
    one_count = 0
    for label in labels:
        one_count = one_count + int(label == 1)
    zero_count = len(labels) - one_count
    return -one_count/len(labels)*math.log2(one_count/len(labels)) - zero_count/len(labels)*math.log2(zero_count/len(labels))

if __name__ == '__main__':
    input_name = sys.argv[1]
    output_name = sys.argv[2]

    df = pd.read_csv(os.path.join('/Users','jko26','Intro-To-ML-CMU','HW_2', 'handout',input_name),sep='\t')
    entropy = entropy(df)
    
    train(df)
    train_pred = predict(df)
    error = error(train_pred, df['heart_disease'])
    inspection_metrics = open(os.path.join('/Users','jko26','Intro-To-ML-CMU', 'HW_2','handout', output_name), "w")
    inspection_metrics.write("Entropy: " + str(entropy) + '\n')
    inspection_metrics.write("Error: " + str(error) + '\n')



    

