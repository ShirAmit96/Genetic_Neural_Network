import random
import numpy as np

# global variables:
pop_size = 1000
num_of_gens = 100
mut_rate = 0.5
elite_rate = 0.1
non_mutated_rate = 0.1
convergence_gens = 10
# This function loads the data from the given file and spilts it into train and test.
def load_and_spilt_data(file):
    # Read the data from the file
    with open(file, 'r') as f:
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
    # Combine inputs and labels into a single array for shuffling
    data = list(zip(inputs, labels))
    random.shuffle(data)
    # Convert the shuffled data back to separate inputs and labels
    inputs, labels = zip(*data)
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
    return train_input, train_label, test_input, test_label

class Layer:
    def __init__(self, input_size, output_size, activation=lambda x: sigmoid(x)):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.activation = activation
    def set_weights(self, weights):
        self.weights = weights
    def get_weights(self):
        return self.weights
    # Computes the forward propagation of the layer for given inputs
    def forward(self, inputs):
        # Calculate output as matrix product of inputs and weights
        output = np.dot(inputs, self.weights)
        # Apply the activation function to the output
        output = self.activation(output)
        return output
    # Retrieves the shape of the layer's weights and the activation function.
    def get_shape(self):
        return self.weights.shape, self.activation
class NeuralNetwork:
    def activate_sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def activate_relu(self,x):
        return np.maximum(0, x)
    def __init__(self,input_size,output_size):
        # List to hold all layers of the neural network
        self.layers = []
        self.add_layer(Layer(input_size, output_size, activation=lambda x: self.activate_sigmoid(x)))
    def add_layer(self, layer):
        # Appends a new layer to the network
        self.layers.append(layer)
    def get_layers(self):
        # Gets the list of layers of the model
        return self.layers
    def forward_propagate(self, inputs):
        # Passes the inputs through each layer of the network
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        # Converts the output of the final layer to binary predictions
        binary_predictions = (outputs > 0.6).astype(int)
        return binary_predictions.flatten()
    def compute_accuracy(self, lables, pred_labels):
        correct_predictions = sum(1 for label, pred_label in zip(lables, pred_labels) if label == pred_label)
        accuracy = correct_predictions / len(lables)
        return accuracy
    def fitness(self,inputs,labels):
        pred_lables = self.forward_propagate(inputs)
        return self.compute_accuracy(labels, pred_lables)
    def mutate(self):
        # choose in probablity of 50 -50 the mutation type:
        random_value = random.choice([0, 1])
        if random_value:
            self.mutate_swap()
        else:
            self.mutate_add_value(mut_rate)
    def mutate_swap(self):
        # Iterate through each layer in the network
        for layer in self.layers:
            if random.uniform(0.0, 1.0) < mut_rate:
                # Get the shape of the layer weights
                shape = layer.weights.shape
                # Generate two different random indices
                index1, index2 = np.random.choice(np.prod(shape), size=2, replace=False)
                # Convert the indices to multi-dimensional indices
                index1 = np.unravel_index(index1, shape)
                index2 = np.unravel_index(index2, shape)
                # Perform the weight swap to introduce mutation
                temp = layer.weights[index1]
                layer.weights[index1] = layer.weights[index2]
                layer.weights[index1] = temp

    def mutate_add_value(self, probability):
        # Iterate through each layer in the network
        for layer in self.layers:
            if random.uniform(0.0, 1.0) < probability:
                # Get the shape of the layer weights
                shape = layer.weights.shape
                # Generate a random index
                index = np.random.choice(np.prod(shape))
                # Convert the index to multi-dimensional index
                index = np.unravel_index(index, shape)
                # Generate a random value from the range [-0.01, 0.01]
                random_value = random.uniform(-0.01, 0.01)
                # Add the random value to the selected index
                layer.weights[index] += random_value


#  This function creates biased list of parents based on their fitness scores.
def create_biased_list(population, train_samples, train_labels):
    # Calculate fitness scores for each network in the population
    fitness_scores = [net[0].fitness(train_samples, train_labels) for net in population]
    # Sort the population based on fitness scores in descending order
    sorted_population = [net for _, net in
                         sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    # Get the number of parents to select
    num_parents = len(population)
    # Calculate the sum of all values from 1 to num_parents
    total_sum = num_parents * (num_parents + 1) // 2
    # create a list of indexes based on score:
    biased_indexes = [(num_parents - pos + 1) / total_sum for pos in range(1, num_parents + 1)]
    # Select parents from the sorted population using the biased_indexes
    biased_list = random.choices(sorted_population, weights=biased_indexes, k=num_parents)
    return biased_list

# Performs crossover between parent networks to generate offspring networks.
def crossover(num_offsprings, parents_list, train_samples, train_labels):
    offsprings_list = []
    for i in range(num_offsprings):
        random_parents = random.sample(parents_list, 2)
        parent1 = random_parents[0][0]
        parent2 = random_parents[1][0]
        offspring = NeuralNetwork(16, 1)
        for j in range(len(parent1.layers)):
            num = np.random.uniform(0.0, 1.0, size=parent1.layers[j].weights.shape)
            offspring.layers[j].weights = np.multiply(1 - num, parent1.layers[j].weights) + np.multiply(num,
                                                                                                          parent2.layers[
                                                                                                            j].weights)
        fitness = offspring.fitness(train_samples, train_labels)
        offsprings_list.append((offspring, fitness))
    return offsprings_list

def evolve_population(population, train_samples, train_labels):
    global mut_rate,num_of_gens, elite_rate, convergence_gens
    history_best_fitness = 0
    count_convergence = 0
    for gen in range(num_of_gens):
        print(f"Generation {gen + 1}/{num_of_gens}") # delete
        for i in range(len(population)):
            fitness = population[i][0].fitness(train_samples, train_labels)
            population[i] = (population[i][0], fitness)
        sorted_population = sorted(population, key=lambda x: x[1], reverse=True)
        best_fitness = sorted_population[0][1]
        print(f"Generation {gen + 1} best fitness score: {best_fitness}") # delete

        if best_fitness == history_best_fitness:
            count_convergence += 1
            if count_convergence >= 2 and mut_rate < 0.9:
                mut_rate += 0.05
            if count_convergence >= convergence_gens:
                print("Convergence reached. stuck for ", convergence_gens, " generations")  # delete
                break
        else:
            count_convergence = 0
            mut_rate = 0.5
        history_best_fitness = best_fitness
        # save the top nn's of the population:
        elite_population = sorted_population[:int(len(population) * elite_rate)]
        # select parents in order to create offsprings:
        num_offsprings = len(population) - len(elite_population)
        parents_list = create_biased_list(population, train_samples, train_labels)
        # create offsprings using crossover:
        offsprings_list = crossover(num_offsprings, parents_list, train_samples, train_labels)
        # mutate the offsprings:
        mutated_offsprings = []
        for offspring in offsprings_list:
            offspring[0].mutate()
            fitness = offspring[0].fitness(train_samples, train_labels)
            mutated_offsprings.append((offspring[0], fitness))
        # create new population from elite and off springs
        new_population = elite_population + mutated_offsprings
        population = sorted(new_population, key=lambda x: x[1], reverse=True)
    # return the best network and it's fitness:
    best_network = population[0]
    return best_network


def create_population():
    global pop_size
    population = []
    for i in range(pop_size):
        network = NeuralNetwork(16, 1)
        population.append((network,0))
    return population


# load data and split it into train and test:
train_samples, train_labels, test_samples, test_labels = load_and_spilt_data("nn0.txt")
# initialize the population of the nn's:
population = create_population()
# apply evolution on the population:
# take the nn with the max fitness, save it's weights and measure accuracy on test:
best_network, best_fitness = evolve_population(population, train_samples, train_labels)
layer_1 = best_network.get_layers()[0].get_weights()
np.savez("wnet0", arr1=layer_1)
predictions = best_network.forward_propagate(test_samples)
accuracy = best_network.compute_accuracy(test_labels, predictions)
print(f"Test Accuracy: {accuracy}")