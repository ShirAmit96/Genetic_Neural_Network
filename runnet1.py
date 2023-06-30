import numpy as np


# This class represents a neural network.
class NeuralNetwork:
    # This class represents a layer of neural network
    class Layer:
        # This function initialize a neural network layer with random weights and specified activation function.
        def __init__(self, input_size, output_size, activation, weights=None):
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
            self.activation = activation

        # This function returns the weights of the layer
        def get_layers_weights(self):
            return self.weights

        # This function performs forward propagation on a certain layer
        def multiply_and_activate(self, inputs):
            dot_product = np.dot(inputs, self.weights)
            return self.activation(dot_product)

    # Initialization of a neural network.
    def __init__(self, input_size, output_size, weights=None):
        self.layers = []
        self.activate_sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activate_relu = lambda x: np.maximum(0, x)
        self.layers.append(self.Layer(input_size, output_size, activation=self.activate_sigmoid, weights=weights))

    # This function returns all the layers in the neural network.
    def get_all_layers(self):
        return self.layers

    # This function performs forward propagation in the neural network and obtain predictions.
    def forward_propagate(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.multiply_and_activate(result)
        predictions = (result > 0.3).astype(int).flatten()
        return predictions

    # This function computes the accuracy of the neural network predictions.
    def compute_accuracy(self, labels, pred_labels):
        correct_predictions = sum(1 for label, pred_label in zip(labels, pred_labels) if label == pred_label)
        accuracy = correct_predictions / len(labels)
        return accuracy

    # This function computes the fitness of the neural network.
    def fitness(self, inputs, labels):
        pred_labels = self.forward_propagate(inputs)
        return self.compute_accuracy(labels, pred_labels)


def load_test_data(file):
    # Read the data from the file
    with open(file, 'r') as f:
        lines = f.readlines()
    # Split each line into inputs and labels
    inputs = []
    for line in lines:
        sequence= line.strip()  # strip leading/trailing spaces and split by any whitespace
        # Convert the binary sequence to a numpy array of integers
        inputs.append(np.array([int(bit) for bit in sequence]))
    inputs = np.array(inputs)
    return inputs

if __name__ == "__main__":
    # Load the values from the text file into a NumPy array
    layer = np.loadtxt("wnet1.txt")
    best_network = NeuralNetwork(16, 1, layer)
    # Load data
    x_test_inputs = load_test_data("testnet1.txt")

    # Perform inference
    test_predictions = best_network.forward_propagate(x_test_inputs)

    # Write predictions to a file
    with open("result1.txt", "w") as file:
        for label in test_predictions:
            file.write(str(label) + "\n")
