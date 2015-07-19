from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 vector, with values in the range -1 to 1 and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the synaptic weights through this function to normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function (the gradient of the curve).
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error. Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):

            # In each iteration we use the entire training set.
            # Hence the variables are matrices.
            layer0 = training_set_inputs

            # Pass the training set through our neural network (a single neuron).
            layer1 = self.think(layer0)

            # Calculate the error for each example (the difference between the desired output and the actual output).
            layer1_error = training_set_outputs - layer1

            # Multiply the error by the gradient of the sigmoid curve at that point.
            # This means weak weights are adjusted more.
            layer1_delta = layer1_error * self.__sigmoid_derivative(layer1)

            # Multiply the adjustment by the input.
            # This means that inputs which are zero do not cause changes to the weights.
            layer1_delta = dot(layer0.T, layer1_delta)

            # Adjust the weights.
            self.synaptic_weights += layer1_delta

    # The neural network thinks.
    def think(self, layer0):
        # Pass inputs through layer1 (our single neuron).
        return self.__sigmoid(dot(layer0, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new example.
    print "Think: "
    print neural_network.think(array([1, 0, 0]))
