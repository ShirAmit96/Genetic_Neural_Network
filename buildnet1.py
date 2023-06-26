import random
import numpy as np

# global variables:
pop_size = 1000
num_of_gens = 100
mut_rate = 0.5
elite_rate = 0.1
non_mutated_rate = 0.1
convergence_gens = 10


# This function prompts the user to enter paths for test and train data.
def get_data_paths():
    train_data_path = input("Enter the path for the train data: ")
    test_data_path = input("Enter the path for the test data: ")

    return train_data_path,test_data_path,


# This function loads the data from the given file and splits it into train and test.
def load_data(file):
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
    return inputs, labels



class NeuralNetwork:
    class Layer:
        def __init__(self, input_size, output_size, activation):
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
            self.activation = activation

        def get_layers_shape(self):
            return self.weights.shape, self.activation

        def get_layers_weights(self):
            return self.weights

        def multiply_and_activate(self, inputs):
            dot_product = np.dot(inputs, self.weights)
            return self.activation(dot_product)



    def __init__(self, input_size, output_size):
        self.layers = []
        self.activate_sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activate_relu = lambda x: np.maximum(0, x)
        self.layers.append(self.Layer(input_size, output_size, activation=self.activate_sigmoid))

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_all_layers(self):
        return self.layers

    def forward_propagate(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.multiply_and_activate(result)
        binary_predictions = (result > 0.3).astype(int)
        return binary_predictions.flatten()

    def compute_accuracy(self, labels, pred_labels):
        correct_predictions = sum(1 for label, pred_label in zip(labels, pred_labels) if label == pred_label)
        accuracy = correct_predictions / len(labels)
        return accuracy

    def fitness(self, inputs, labels):
        pred_labels = self.forward_propagate(inputs)
        return self.compute_accuracy(labels, pred_labels)



def mutate(nn):
    # choose in probablity of 50 -50 the mutation type:
    random_value = random.choice([0, 1])
    if random_value:
        mutate_swap(nn)
    else:
        mutate_add_value(nn,mut_rate)

def mutate_swap(nn):
    # Iterate through each layer in the network
    for layer in nn.layers:
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

def mutate_add_value(nn, probability):
    # Iterate through each layer in the network
    for layer in nn.layers:
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
        for i in range(len(population)):
            fitness = population[i][0].fitness(train_samples, train_labels)
            population[i] = (population[i][0], fitness)
        sorted_population = sorted(population, key=lambda x: x[1], reverse=True)
        best_fitness = sorted_population[0][1]

        if best_fitness == history_best_fitness:
            count_convergence += 1
            if count_convergence >= 2 and mut_rate < 0.9:
                mut_rate += 0.05
            if count_convergence >= convergence_gens:
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
            mutate(offspring[0])
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
train_file, test_file = get_data_paths()
train_samples, train_labels = load_data(train_file)
test_samples, test_labels = load_data(test_file)
# initialize the population of the nn's:
population = create_population()
# apply evolution on the population:
# take the nn with the max fitness, save it's weights and measure accuracy on test:
best_network, best_fitness = evolve_population(population, train_samples, train_labels)
layer_1 = best_network.get_all_layers()[0].get_layers_weights()
np.savez("wnet1", arr1=layer_1)
predictions = best_network.forward_propagate(test_samples)
accuracy = best_network.compute_accuracy(test_labels, predictions)

