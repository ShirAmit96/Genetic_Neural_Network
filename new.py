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
    train_input = inputs[:n_train]
    train_label = labels[:n_train]
    test_input = inputs[n_train:]
    test_label = labels[n_train:]

    return train_input,train_label,test_input,test_label

train_inputs, train_labels, test_inputs, test_labels = load_data()
num_offsprings = 50
mutation_rate = 20
generations = 100
convergence_limit= 10
class NeuralNetwork:
    def __init__(self, layer_sizes,matrix=[], new=True):
        self.layer_sizes = layer_sizes
        if new:
            self.network = self._initialize_network()
        else:
            self.network = matrix

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

    def compute_accuracy(self, inputs, labels):
        right_count = 0
        predictions = self.forward_propagate(inputs)
        total = len(predictions)
        for label, pred_label in zip(labels,predictions):
            if label == pred_label:
                right_count+=1
        accuracy = right_count/total
        return accuracy


# Assuming NeuralNetwork is the class defined earlier
population = []
population_size = 2


def create_population():
    global population_size, population
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
        population.append((nn,0))

def flatten_list(lst):
    flattened = []
    for sublist1 in lst:
        for sublist2 in sublist1:
            for item in sublist2:
                flattened.append(item)
    return flattened


def reshape_list(flattened, dimensions):
    reshaped = []
    index = 0
    for dim1, dim2 in zip(dimensions[:-1], dimensions[1:]):
        sublist = np.array(flattened[index: index + (dim1 * dim2)]).reshape((dim1, dim2))
        index += dim1 * dim2
        reshaped.append(sublist)
    return reshaped


# This class represent a genetic algorithm object that evolves the nn population.
class Genetic_Algorithm:
    def __init__(self):
        pass

    def crossover(self, num_of_offsprings):
        global population,train_inputs, train_labels, test_inputs, test_labels, num_offsprings
        for i in range(num_of_offsprings):
            # randomly choose 2 parents:
            parent_1, parent_2 = random.sample(population, 2)
            parent_1_nn = np.array(parent_1[0].network)
            parent_2_nn = np.array(parent_2[0].network)
            # Flatten the array
            flattened_par1 = flatten_list(parent_1_nn)
            flattened_par2 = flatten_list(parent_2_nn)
            larger=0
            # Determine the larger and smaller matrices
            if len(flattened_par1) >= len(flattened_par2):
                larger_mat = flattened_par1
                smaller_mat = flattened_par2
                larger=1
            else:
                larger_mat = flattened_par2
                smaller_mat = flattened_par1
                larger=2

            # Choose a random index within the range of the smaller matrix
            chosen_index = np.random.randint(len(smaller_mat)-1)
            # Copy values from the smaller matrix to the larger matrix until the chosen index
            larger_mat[:chosen_index] = smaller_mat[:chosen_index]
            if larger==1:
                # Reshape the larger matrix to its original shape
                child = reshape_list(larger_mat,parent_1[0].layer_sizes)
            else:
                child=reshape_list(larger_mat, parent_2[0].layer_sizes)
            layer_sizes = [layer.shape[0] for layer in child]
            child_nn = NeuralNetwork(layer_sizes, child, False)
            child_fit = child_nn.compute_fitness(train_inputs, train_labels)
            population.append((child_nn, child_fit))

    def mutate(self, mutation_list):
        global population, train_labels,train_inputs
        new_list = []
        for nn, fitness in mutation_list:
            # Create a random mask based on the probability
            mask = np.random.choice([0, 1], size=nn.network.shape, p=[0.9, 0.1])

            # Iterate over the matrix using nditer
            with np.nditer(nn.network, op_flags=['readwrite']) as it:
                for cell in it:
                    # Check if the corresponding mask cell is 1 (True)
                    if mask[it.multi_index]:
                        # Add a random value between -1 and 1 to the current cell
                        cell[...] += np.random.uniform(-1, 1)
            matrix_fit = nn.compute_fitness(train_inputs, train_labels)
            new_list.append(nn, matrix_fit)


    def evolve_pop(self):
        global population, population_size ,mutation_rate, generations,convergence_limit
        best_fit = 1
        count_same_fit = 0
        for i in range(generations):
            print(i)
            # sort the pop by fitness:
            population = sorted(population, key=lambda x: x[1])
            # save the top 10 nn's (elitism):
            elitism_list = population[:10]
            # create 50 offsprings using crossover:
            self.crossover(num_offsprings)
            # mutate 20 randon nn's from the population:
            mutation_list = random.sample(population, k=mutation_rate)
            self.mutate(mutation_list)
            population.extend(mutation_list)
            population.extend(elitism_list)
            # sort the list again:
            population = sorted(population, key=lambda x: x[1])
            # take only the top 100 nn's:
            population = population[:population_size]
            # check convergence:
            if population[0][1]==best_fit:
                count_same_fit+=1
            best_fit = population[0][1]
            if count_same_fit> convergence_limit:
                return population[0]
        return population[0]






def flow():
    global population
    create_population()
    global train_inputs, train_labels, test_inputs, test_labels
    for i in range(len(population)):
                nn = population[i][0]
                loss = nn.compute_fitness(train_inputs, train_labels)
                layers = len(nn.layer_sizes)
                print("creating individual num:", i )
                population[i] = (nn, loss)
    # evolve population using GA:
    GA = Genetic_Algorithm()
    chosen_nn = GA.evolve_pop()
    accuracy = chosen_nn.compute_accuracy(test_inputs,test_labels)
    print("accuracy: ", accuracy)



        # sort the  fitness list so the smallest fitness will be first:
        # save the top nn's (elitisem):
        # create offsprings using crssover:
        # perform mutation on 5 perecnt of


        # For my test:
        # for nn in population:
        #     for i in range(len(nn.network)):
        #         nn.network[i] += random.random()


        # Create new population

flow()
