import numpy as np

np.set_printoptions(suppress=True)


def sigmoid(inputs):
    return 1/(1+np.exp(np.negative(inputs)))

def d_sigmoid(inputs):
    sigmoidP = sigmoid(inputs)
    return sigmoidP*(1-sigmoidP)


#Layer
class Layer:
    def __init__(self, input_n, output_n):
        self.weights = [[-2]*output_n]*input_n #np.random.rand(input_n, output_n)*2-2
        self.biases = [0]*output_n
        self.Y = 0
        # print(f"{self.weights}\n{self.biases}\n")

    def forward(self, inputs):
        self.Y = np.dot(inputs, self.weights) + self.biases

#Neural network that chooses dominant color of image from multiple numbers (pseudo-pixels n(X)=20)
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
        self.X = 0
        if (type(self.layers) != list):
            self.layers = [self.layers]


    def test(self, input):
        self.X = [sigmoid(layer.Y) for layer in self.layers if (layer == self.layers[-1])]
        self.X.insert(0, input)

        print(self.X[0])

        for j in range(len(self.layers)):
            self.layers[j].forward(self.X[j])
            print(self.layers[j].Y)
    

    def get_net_output(self):
        return sigmoid(self.layers[-1].Y)
    

    def train(self, input, output):
        # for
        self.test(input)
        self.Backprop(d_sigmoid, output)
        

    def Backprop(self, derivative_func, y_true):
        ratio_el = []
        ratio_el.append(2*(y_true-self.layers[-1].Y))

        for j in reversed(range(len(self.layers))):
            ratio_el.append(derivative_func(self.layers[j].Y))
            ratio = np.dot(np.array(self.layers[j].weights), ratio_el[0].T)

            self.layers[j].weights += np.dot(np.array(sigmoid(self.X[j])).T, ratio)
            self.layers[j].biases += sum(ratio)


X = [[1,1],[1,0],[0,1],[0,0]]
Y = [[1],[0],[0],[0]]

layer1 = Layer(2, 4)
layer2 = Layer(4, 1)

net = Neural([layer1, layer2])
# net.test(X)
# print(net.get_net_output())

for i in range(500):
    net.train(X, Y)
    print(i)

net.test(X)
print(net.get_net_output())
