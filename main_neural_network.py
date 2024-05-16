import numpy as np
import json


def sigmoid(inputs):
    return 1/(1+np.exp(np.negative(inputs)))


def d_sigmoid(inputs):
    sigmoidP = sigmoid(inputs)
    return sigmoidP*(1-sigmoidP)


# Layer
class Layer:
    def __init__(self, input_n, output_n):
        self.weights = np.random.rand(input_n, output_n)*2-1
        self.biases = [0]*output_n

    def forward(self, inputs):
        self.Y = np.dot(inputs, self.weights) + self.biases


# Neural network that chooses dominant color of image from multiple numbers (pseudo-pixels n(X)=20)
class Neural:
    def __init__(self):
        pass

    def Backprop(self, derivative_func, dot_sum, y_out, y_true):
        self.error = y_true-np.array(y_out).T
        self.d_x = np.array(derivative_func(dot_sum)).T*2*(self.error)
        return self.d_x


net = Neural()

X = [[1, 1], [1, 0], [0, 1], [0, 0]]
Y = [1, 0, 0, 0]

layer1 = Layer(2, 1)


for i in range(50):
    for j in range(len(X)):
        layer1.forward(X[j])

        net.Backprop(d_sigmoid, layer1.Y, sigmoid(layer1.Y), Y[j])
        layer1.weights += np.dot(np.array([X[j]]).T, [net.d_x])
        layer1.biases += net.d_x

###
# W tym miejscu - zapis layer1.weights (2 zmienne) i layer1.bias (1 zmienna) do pliku json
###
data_to_save = {
    'weights': layer1.weights.tolist(),
    'biases': layer1.biases.tolist()
}
print("Zmienne:\n", layer1.weights, "\n"*2, layer1.biases, "\n"*2)
with open('zmienne.json', 'w') as json_file:
    json.dump(data_to_save, json_file)
try:
    with open("zmienne.json", "r") as input_file:
        content = json.load(input_file)
        print(f"Zawartość pliku json: {content}\n")
except FileNotFoundError:
    print("Nie znaleziono pliku!")

layer1.forward(X)
print(sigmoid(layer1.Y))
