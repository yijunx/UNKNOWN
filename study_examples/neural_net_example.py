import numpy as np
from study_examples.neural_perceptron_example import sigmoid
from study_examples.neural_perceptron_example import sigmoid_derivative


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)

        # init the weights
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_inputs, training_outputs, 20000)

    print("synaptic weights after training")
    print(neural_network.synaptic_weights)





