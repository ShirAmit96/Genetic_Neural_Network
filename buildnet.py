import numpy as np
import random
# Init a  population of neural network - aka weights.
# The nn is 2-3 layers - random between 2-4
# The size of each layer also a random parameter - 5-10
# Activation function - sigmoid
# Train the nn
# fitness = evaluation the nn using the loss function
# Use GA to make changes in the weights - instead of back propagation - mutation, crossover, elitism.
# Choose the model weights and structure using the test set.

# This class represent a neural network object.
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.network = self._initialize_network()

    def _initialize_network(self):
        # Initialize the weights for each layer in the network
        network = []
        for i in range(len(self.layer_sizes) - 1):
            # Randomly initialize the weights
            weight_matrix = np.random.uniform(-1, 1, size=(self.layer_sizes[i], self.layer_sizes[i+1]))
            network.append(weight_matrix)
        return network

    # Multiply the weights with the inputs and sum all the products.
    def _weighted_sum(self, weights, inputs):
        return np.dot(inputs, weights)

    def _activate(self, activation):
        # Sigmoid activation function
        return 1.0 / (1.0 + np.exp(-activation))

    # This function does the forward propgation.
    def forward_propagate(self, inputs):
        # Convert inputs list to a 2D array
        if inputs.ndim == 1:
            inputs = np.reshape(inputs, (1, -1))

        for i in range(len(self.network)):
            new_inputs = []
            for j in range(self.network[i].shape[1]):
                # Make sure the shapes of weights and inputs match for dot product
                weights = np.reshape(self.network[i][:, j], (-1,))
                sum = self._weighted_sum(weights, inputs)
                new_inputs.append(self._activate(sum))
            # Transpose for the shapes to fit
            inputs = np.array(new_inputs).T  
        return inputs

    def compute_fitness(self, inputs, labels):
        predictions = self.forward_propagate(inputs)
        # Calc the cross entropy loss
        loss = -np.mean(np.log(predictions) * labels + np.log(1 - predictions) * (1 - labels))
        return loss


# Assuming NeuralNetwork is the class defined earlier
population = []
population_size = 2


def create_population():
    for _ in range(population_size):
        # Random number of hidden layers (between 2 and 4)
        n_hidden_layers = random.randint(2, 4)

        # Random layer sizes (between 5 and 10)
        layer_sizes = [random.randint(5, 10) for _ in range(n_hidden_layers)]

        # Input layer
        layer_sizes.insert(0, 16)
        # Output layer
        layer_sizes.append(1)

        nn = NeuralNetwork(layer_sizes)
        population.append(nn)

def load_data():
    # Read the data from the file
    with open('nn0.txt', 'r') as f:
        lines = f.readlines()

    # Split each line into inputs and labels
    inputs = []
    labels = []
    for line in lines:
        sequence, label = line.strip().split()  # strip leading/trailing spaces and split by any whitespace
        # Convert the binary sequence to a numpy array of integers
        inputs.append(np.array([int(bit) for bit in sequence]))
        # Convert the label to an integer
        labels.append(int(label))

    # Convert lists to numpy arrays
    inputs = np.array(inputs)
    labels = np.array(labels)

    # Calculate the number of training samples (80% of total data)
    n_train = int(0.8 * len(inputs))

    # Split the data into training and test sets
    train_inputs = inputs[:n_train]
    train_labels = labels[:n_train]
    test_inputs = inputs[n_train:]
    test_labels = labels[n_train:]

    return train_inputs,train_labels,test_inputs,test_labels



def flow():
    create_population()
    train_inputs, train_labels, test_inputs, test_labels = load_data()
    fitness_list = []
    gen = 100
    for i in range(gen):
        for nn in population:
            # To shir: I used two fitness measures, we'll choose the one that works the best.
            loss = nn.compute_fitness(train_inputs, train_labels)
            layers = len(nn.layer_sizes)
            print(layers,nn.layer_sizes, loss)
            fitness_list.append((nn,loss))
        # Use GA according to finess scores


        # For my test:
        # for nn in population:
        #     for i in range(len(nn.network)):
        #         nn.network[i] += random.random()


        # Create new population

flow()
