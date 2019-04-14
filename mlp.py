import numpy as np
import pandas as pd

class MultiLayerPerceptron():
    def __init__(self, in_train, in_test, hidden_num = 10, lr = 0.01, max_epoch = 30, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.training_data = in_train
        self.testing_data = in_test
        self.max_epoch = max_epoch
        self.hidden_num = hidden_num
        self.weights_I_H = np.random.uniform(-0.05, 0.05, (785, self.hidden_num))
        self.weights_H_O = np.random.uniform(-0.05, 0.05, (self.hidden_num + 1, 10))
        self.prev_weights_I_H = np.zeros((785, self.hidden_num))
        self.prev_weights_H_O = np.zeros((self.hidden_num + 1, 10))
        self.hidden_act_with_bias = np.ones(self.hidden_num + 1)
    def sigmoid_func(self, input_val):
        return 1 / (1 + np.exp(-input_val))
    def fit(self):
        col_index = self.training_data.shape[1]
        print('training the network...')
        for epoch_num in range(self.max_epoch):
            print('======================')
            print('Epoch: ', epoch_num + 1)
            for example_index in range(self.training_data.shape[0]):
                # cast the target label to a vector using one-hot encoding
                target_vector = np.ones(10) * 0.1
                target_vector[int(self.training_data[example_index, 0])] = 0.9
                
                # forward propagation
                input_to_hidden = self.sigmoid_func(np.dot(self.training_data[example_index, 1:col_index], self.weights_I_H))
                self.hidden_act_with_bias[:self.hidden_num] = input_to_hidden
                hidden_to_output = self.sigmoid_func(np.dot(self.hidden_act_with_bias, self.weights_H_O))

                # backward propagation
                # compute error terms
                error_output_layer = hidden_to_output * (1 - hidden_to_output) * np.subtract(target_vector, hidden_to_output)
                sum_term = np.dot(self.weights_H_O, error_output_layer)
                error_hidden_layer = self.hidden_act_with_bias * (1 - self.hidden_act_with_bias) * sum_term

                # update the weights
                delta_hidden_output = self.lr * np.outer(self.hidden_act_with_bias, error_output_layer) + self.momentum * self.prev_weights_H_O
                self.weights_H_O += delta_hidden_output
                delta_input_hidden = self.lr * np.outer(error_hidden_layer[:self.hidden_num], self.training_data[example_index, 1:col_index]).T + self.momentum * self.prev_weights_I_H
                self.weights_I_H += delta_input_hidden

                # update the previous delta weights
                self.prev_weights_H_O = delta_hidden_output
                self.prev_weights_I_H = delta_input_hidden
        print('Training is done...')
    def predict(self):
        col_index = self.testing_data.shape[1]
        correct_num = 0
        
        # forward propagation
        out_hidden = np.ones((self.testing_data.shape[0], self.hidden_num + 1))
        tmp_out_hidden = self.sigmoid_func(np.dot(self.testing_data[:, 1: col_index], self.weights_I_H))
        out_hidden[:, :self.hidden_num] = tmp_out_hidden
        final_out = self.sigmoid_func(np.dot(out_hidden, self.weights_H_O))

        # compute testing accuracy
        for test_index in range(self.testing_data.shape[0]):
            pred = np.argmax(final_out[test_index, :])
            if pred == self.testing_data[test_index, 0]:
                correct_num += 1
        return correct_num / float(self.testing_data.shape[0]) * 100.0
    
def data_loading():
    out_data = {}
    training = pd.read_csv('mnist_train.csv', header = None).values
    training = np.asarray(training, dtype = float)
    training[:, 1:785] = training[:, 1:785] / 255.0
    training = np.append(training, np.ones((training.shape[0], 1)), axis = 1)
    out_data['train_data'] = training

    testing = pd.read_csv('mnist_test.csv', header = None).values
    testing = np.asarray(testing, dtype = float)
    testing[:, 1:785] = testing[:, 1:785] / 255.0
    testing = np.append(testing, np.ones((testing.shape[0], 1)), axis = 1)
    out_data['test_data'] = testing

    return out_data

def main():
    # data loading
    data_collect = data_loading()

    mlp = MultiLayerPerceptron(data_collect['train_data'], data_collect['test_data'], max_epoch = 10, hidden_num = 50)

    # train a Multi-layer perceptron
    mlp.fit()

    # prediction computation
    accuracy = mlp.predict()
    print('Testing Accuracy: ', accuracy)
    
if __name__ == "__main__":
    main()
