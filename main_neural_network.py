import numpy as np
import json

np.set_printoptions(suppress=True)


def sigmoid(inputs):
    """
    Oblicza funkcję sigmoidalną dla danego wejścia.

    Funkcja sigmoidalna jest zdefiniowana jako:
    σ(x) = 1 / (1 + e^(-x))

    Parametry:
    inputs (numpy.ndarray lub float): Wartość lub tablica wartości, dla których ma być obliczona funkcja sigmoidalna.

    Zwraca:
    numpy.ndarray lub float: Wartość lub tablica wartości po zastosowaniu funkcji sigmoidalnej.
    """
    return 1/(1+np.exp(np.negative(inputs)))


def d_sigmoid(inputs):
    """
    Oblicza pochodną funkcji sigmoidalnej dla danego wejścia.

    Pochodna funkcji sigmoidalnej jest zdefiniowana jako:
    σ'(x) = σ(x) * (1 - σ(x))
    gdzie σ(x) jest wartością funkcji sigmoidalnej dla x.

    Parametry:
    inputs (numpy.ndarray lub float): Wartość lub tablica wartości, dla których ma być obliczona pochodna funkcji sigmoidalnej.

    Zwraca:
    numpy.ndarray lub float: Wartość lub tablica wartości po zastosowaniu pochodnej funkcji sigmoidalnej.
    """
    sigmoidP = sigmoid(inputs)
    return sigmoidP*(1-sigmoidP)


# Layer
class Layer:
    def __init__(self, input_n, output_n):
        self.weights = [[-2]*output_n]*input_n  # np.random.rand(input_n, output_n)*2-2
        self.biases = [0]*output_n
        self.Y = 0
        self.X = 0

    def forward(self, inputs):
        self.X = inputs
        self.Y = np.dot(self.X, self.weights) + self.biases

# Neural network that chooses dominant color of image from multiple numbers (pseudo-pixels n(X)=20)
class Neural:
    """
    Neural Network Class
    
    Parameters
    ----------
        `layers`
            Lista utworzonych warstw sieci neuronowej, podana w kolejności chronologicznej
    """
    def __init__(self, layers):
        self.layers = layers
        if (type(self.layers) != list):
            self.layers = [self.layers]

    def test(self, input):
        self.layers[0].forward(input)

        for j in range(1, len(self.layers)):
            self.layers[j].forward(sigmoid(self.layers[j-1].Y))

    def get_net_output(self):
        return sigmoid(self.layers[-1].Y)

    def train(self, input, output):
        self.test(input)
        self.Backprop(d_sigmoid, output)

    def Backprop(self, derivative_func, y_true):
        ratio = (2*(y_true-sigmoid(self.layers[-1].Y))).T
        for j in reversed(range(len(self.layers))):
            ratio = derivative_func(self.layers[j].Y)*ratio.T

            self.layers[j].weights += np.dot(np.array(self.layers[j].X).T, ratio)
            self.layers[j].biases += sum(ratio)

            ratio = np.dot(self.layers[j].weights, ratio.T)


# Training Data
X = [[1, 1], [1, 0], [0, 1], [0, 0]]
Y = [[1], [0], [0], [0]]

# Initialize Network
layer1 = Layer(2, 8)
layer2 = Layer(8, 4)
layer3 = Layer(4, 4)
layer4 = Layer(4, 1)
net = Neural([layer1, layer2, layer3, layer4])

# Train Network
for i in range(500):
    net.train(X, Y)


###
# W tym miejscu - zapis layer1.weights (2 zmienne) i layer1.bias (1 zmienna) do pliku json
###
data_to_save = {
    'weights': layer1.weights.tolist(),
    'biases': layer1.biases.tolist()
}
print("Zmienne:\n", layer1.weights, "\n"*2, layer1.biases, "\n"*2)
with open('variables.json', 'w') as json_file:
    json.dump(data_to_save, json_file)
try:
    with open("variables.json", "r") as input_file:
        content = json.load(input_file)
        print(f"Zawartość pliku json: {content}\n")
except FileNotFoundError:
    print("Nie znaleziono pliku!")

try:
    with open("historical-results.json", "r") as input_file:
        content = json.load(input_file)
except FileNotFoundError:
    content = {}

# Ensure 'data' key exists in the content dictionary
if 'data' not in content:
    content['data'] = []

# Test Network
net.test(X)
print(net.get_net_output())

data_results = {
    'results': net.get_net_output().tolist()
}

content["data"].append(data_results)

try:
    with open("historical-results.json", "w") as output_file:
        json.dump(content, output_file)
except FileNotFoundError:
    print("Nie znaleziono pliku!")
