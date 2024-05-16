import numpy as np
import json

# Generate random sets of inputs with outputs assinged
X, Y = [], []
for i in range(50):  # Number of sets of inputs
    image = []
    offset = i % 5
    for j in range(20):  # Number of pixels (inputs)
        image.append(np.random.random() / 5 + (offset / 5))
    R1 = np.random.randint(0, 5)
    R2 = np.random.randint(0, 2)
    for k in range(R1):
        image[np.random.randint(0, 20)] = R2
    X.append(image)

    # One-hot encoding scalars - Classes: (0)Red, (1)Green, (2)Blue, (3)Yellow
    match offset:
        case 1:
            Y.append(3)  # Yellow
        case 2:
            Y.append(1)  # Green
        case 3:
            Y.append(2)  # Blue
        case default:
            Y.append(0)  # Red

    # Expected final input mean to color = [0.14, 0.32, 0.5, 0.68, 0.86] (R-Y-G-B-R)    *color slider*
    # output_product = sum(image)/20
    # print(f"{offset}: {output_product}, R:(n{R1},b{R2}), Y:{y[i]}")


def one_hot(label, class_n=4):
    x = [0] * class_n
    x[label] = 1
    return x


#Activation functions
def ReLU(inputs):
    return np.maximum(0, inputs)


def Softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def sigmoid(inputs):
    return 1 / (1 + np.exp(np.negative(inputs)))


# Derivatives of activation functions
def d_ReLU(input):
    if input > 0:
        return 1
    else:
        return 0


def d_Softmax(inputs):
    return 1


def d_sigmoid(inputs):
    sigmoidP = sigmoid(inputs)
    return sigmoidP * (1 - sigmoidP)


# Layer
class Layer:
    def __init__(self, input_n, output_n):
        self.weights = np.random.rand(input_n, output_n) * 2 - 1
        self.biases = [0] * output_n
        self.error = [0] * output_n

    def forward(self, inputs):
        self.Y = np.dot(inputs, self.weights) + self.biases


# Neural network that chooses dominant color of image from multiple numbers (pseudo-pixels n(X)=20)
class Neural:
    def __init__(self):
        pass

    # Loss
    def Loss(self, y_out, y_true):
        y_out_clip = np.clip(y_out, 1e-7, 1)
        self.losses = -np.log(y_out_clip[range(len(y_out)), y_true])
        return np.mean(self.losses)

    def Backprop(self, derivative_func, dot_sum, y_out, y_true):
        self.error = y_out - y_true
        self.d_x = derivative_func(dot_sum) * 2 * (self.error)
        return self.d_x


layer1 = Layer(20, 8)
layer2 = Layer(8, 4)
net = Neural()

for i in range(len(X)):
    # print(X[i], Y[i], "\n")
    layer1.forward(X[i])
    layer2.forward(sigmoid(layer1.Y))

    net.Backprop(d_sigmoid, layer2.Y, sigmoid(layer2.Y), one_hot(Y[i]))
    layer2.error = np.dot(net.error, layer2.weights.T)
    layer2.weights += layer2.weights * net.d_x
    layer2.biases += net.d_x

    # net.Backprop(d_sigmoid, layer1.Y, sigmoid(layer1.Y), sigmoid(layer2.error))
    # layer1.weights += layer1.weights*net.d_x
    # layer1.biases += net.d_x

print(layer2.Y)

# layer1.forward(X)
# layer2.forward(sigmoid(layer1.Y))

# print(net.Loss(sigmoid(layer2.Y), Y))
