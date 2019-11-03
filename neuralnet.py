import numpy as np
# import yfinance as yf


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 3)) - 1

    # def getcurrentweek(self):
    #     data = yf.download('TSLA', '2019-10-14', '2019-10-21')
    #     closing = data[['Adj Close']]
    #     return [price for price in closing]
    #
    # def getlastweek(self):
    #     data = yf.download('TSLA', '2019-10-07', '2019-10-11')
    #     closing = data[['Adj Close']]
    #     return [price for price in closing]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(- x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_output, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            error = training_output - output
            adjustments = np.dot(training_inputs.T, error *
                                 self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        input = inputs
        output = self.sigmoid(np.dot(input, self.synaptic_weights))

        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[1731, 1728, 1742],
                                [1728, 1731, 1729],
                                [1731, 1745, 1761],
                                ])

    # training_output = neural_network.getlastweek()
    # training_inputs = neural_network.getcurrentweek()

    # print(neural_network.getcurrentweek())
    # print(neural_network.getlastweek())

    training_output = np.array([[1731, 1726, 1762, 1781]]).T
    neural_network.train(training_inputs, training_output, 100000)

    print("synaptic_weights after Training: ")
    print(neural_network.synaptic_weights)

    print("Next day assumption: ")
    print(neural_network.think(training_inputs))
