import numpy as np
import argparse
import csv
import os

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    for epoch in range(num_epoch): 
        #skip shuffling step to make code reproducible
        for i in range(X.shape[0]): 
            features = X[i]
            label = y[i]
            features = np.insert(features, 0, 1, axis=0) #prepend 1 to features to account for intercept term
            grad = instance_gradient(theta, features, label) #compute pointwise gradient
            #update theta
            theta = theta - learning_rate*grad

    return theta
        
def instance_gradient( #calculates gradient of the loss with respect to theta for a single training instance
    theta : np.ndarray, 
    features : np.ndarray, 
    label : int): 
    
    #print("instance probability: " + str((instance_probability(theta, features))))
    return features*(instance_probability(theta, features) - label)

def predict(
    theta : np.ndarray,
    X : np.ndarray, #X is 2D array of (N, D) of N instances,
    out_file_name
) -> np.ndarray:
    predictions = np.zeros(X.shape[0],)
    out_file = open(out_file_name, 'w')
    for i in range(X.shape[0]):
        prediction = instance_predict(theta, np.insert(X[i], 0, 1))
        out_file.write(str(prediction) + '\n')
        predictions[i] = prediction
    return predictions

def instance_predict(
    theta : np.ndarray,
    X : np.ndarray,
) -> int:
    return int(instance_probability(theta, X) >= 0.5)

def instance_probability( #return a float probability that the label Y is 1
    theta : np.ndarray,
    X : np.ndarray,
) -> float:
    
    return sigmoid(np.transpose(theta) @ X)


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    return 1 - (np.sum(y_pred == y)/len(y_pred))

def load_data(file_name):
    arr = None
    with open(os.getcwd() + '/' + file_name) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            arr = np.vstack((arr, np.array(row[:-1], dtype=float))) if not arr is None else np.array(row[:-1], dtype=float)
    return arr

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    theta = np.zeros(shape=(301,)) #column vector to hold 1 intercept term and 300 parameters
    train_data = load_data(args.train_input[1:]) #train data is a (N, D + 1) numpy array (D + 1 is due to label)
    test_data = load_data(args.test_input)
    val_data = load_data(args.validation_input[1:])

    train_features = train_data[:,1:]
    train_labels = train_data[:,0]
    test_features = test_data[:,1:]
    test_labels = test_data[:,0]

    theta = train(theta, train_features, train_labels, args.num_epoch, args.learning_rate)
    #print("theta after training: " + str(theta))

    test_predictions = predict(theta, test_features, args.test_out)
    train_predictions = predict(theta, train_features, args.train_out)

    metrics_file = open(args.metrics_out, 'w')
    metrics_file.write("error(test): " + str(compute_error(test_predictions, test_labels)) + '\n')
    metrics_file.write("error(train): " + str(compute_error(train_predictions, train_labels)))






