# import numpy as np

# #Generate random sets of inputs with outputs assinged
# def generate_test_inputs():
#     X, Y = [],[]
#     for i in range(50): #Number of sets of inputs
#         image = []
#         offset = i%5
#         for j in range(20): #Number of pixels (inputs)
#             image.append(np.random.random()/5+(offset/5))
#         R1 = np.random.randint(0,5)
#         R2 = np.random.randint(0,2)
#         for k in range(R1):
#             image[np.random.randint(0,20)] = R2
#         X.append(image)
        
#         #One-hot encoding scalars - Classes: (0)Red, (1)Green, (2)Blue, (3)Yellow
#         match offset:
#             case 1:
#                 Y.append(3) #Yellow
#             case 2:
#                 Y.append(1) #Green
#             case 3:
#                 Y.append(2) #Blue
#             case default:
#                 Y.append(0) #Red
        
#         # # Expected final input mean to color = [0.14, 0.32, 0.5, 0.68, 0.86] (R-Y-G-B-R)    *color slider*
#         # output_product = sum(image)/20
#         # print(f"{offset}: {output_product}, R:(n{R1},b{R2}), Y:{y[i]}")
#     return X, Y

# def one_hot(label, class_n = 4):
#     x=[0]*class_n
#     x[label] = 1
#     return x


# #Activation functions
# def ReLU(inputs):
#     return np.maximum(0, inputs)

# def Softmax(inputs):
#     exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#     return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# def sigmoid(inputs):
#     return 1/(1+np.exp(np.negative(inputs)))

# #Derivatives of activation functions
# def d_ReLU(input):
#     if input > 0:
#         return 1
#     else:
#         return 0

# def d_Softmax(inputs):
#     return 1

# def d_sigmoid(inputs):
#     sigmoidP = sigmoid(inputs)
#     return sigmoidP*(1-sigmoidP)



# #Layer
# class Layer:
#     def __init__(self, input_n, output_n):
#         self.weights = np.random.rand(input_n, output_n)*2-1
#         self.biases = [0]*output_n
#         self.error = [0]*output_n

#     def forward(self, inputs):
#         self.Y = np.dot(inputs, self.weights) + self.biases

# #Neural network that chooses dominant color of image from multiple numbers (pseudo-pixels n(X)=20)
# class Neural:
#     def __init__(self):
#         pass

#     #Loss
#     def Loss(self, y_out, y_true):
#         y_out_clip = np.clip(y_out, 1e-7, 1)
#         self.losses = -np.log(y_out_clip[range(len(y_out)), y_true])
#         return np.mean(self.losses)

#     def Backprop(self, derivative_func, dot_sum, y_out, y_true):
#         self.error = y_true-y_out
#         self.d_x = derivative_func(dot_sum)*2*(self.error)
#         return self.d_x


# layer1 = Layer(20, 8)
# layer2 = Layer(8, 4)
# net = Neural()

# X, Y = generate_test_inputs()

# for i in range(len(X)):
#     #print(X[i], Y[i], "\n")
#     layer1.forward(X[i])
#     layer2.forward(sigmoid(layer1.Y))

#     net.Backprop(d_sigmoid, layer2.Y, sigmoid(layer2.Y), one_hot(Y[i]))
#     layer2.error = np.dot(net.error, layer2.weights.T)
#     layer2.weights += np.dot(np.array(sigmoid([layer1.Y])).T, np.array([net.d_x]))
#     layer2.biases += net.d_x

#     # net.Backprop(d_sigmoid, layer1.Y, sigmoid(layer1.Y), sigmoid(layer2.error))
#     # layer1.weights += layer1.weights*net.d_xw
#     # layer1.biases += net.d_x


# print(layer2.Y)

# # layer1.forward(X)
# # layer2.forward(sigmoid(layer1.Y))

# # print(net.Loss(sigmoid(layer2.Y), Y))






















































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
        self.X = [sigmoid(layer.Y) for layer in self.layers if (layer != self.layers[-1])]
        self.X.insert(0, input)

        for j in range(len(self.layers)):
            self.layers[j].forward(self.X[j])
    

    def get_net_output(self):
        return sigmoid(self.layers[-1].Y)
    

    def train(self, input, output):
        self.test(input)
        self.Backprop(d_sigmoid, output)
        

    def Backprop(self, derivative_func, y_true):
        ratio = (2*(y_true-sigmoid(self.layers[-1].Y))).T
        for j in reversed(range(len(self.layers))):
            ratio = derivative_func(self.layers[j].Y)*ratio.T

            self.layers[j].weights += np.dot(np.array(self.X[j]).T, ratio)
            self.layers[j].biases += sum(ratio)

            ratio = np.dot(self.layers[j].weights, ratio.T)


# Training Data
X = [[1,1],[1,0],[0,1],[0,0]]
Y = [[1],[0],[0],[0]]

# Initialize Network
layer1 = Layer(2, 8)
layer2 = Layer(8, 4)
layer3 = Layer(4, 1)
net = Neural([layer1, layer2, layer3])

# Train Network
for i in range(500):
    net.train(X, Y)

# Test Network
net.test(X)
print(net.get_net_output())









































#######
# OK.py
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
        self.ratio = [0]*output_n
        # print(f"{self.weights}\n{self.biases}\n")

    def forward(self, inputs):
        self.Y = np.dot(inputs, self.weights) + self.biases

#Neural network that chooses dominant color of image from multiple numbers (pseudo-pixels n(X)=20)
class Neural:
    def __init__(self):
        pass

    def Backprop(self, derivative_func, dot_sum, y_out, y_true):
        self.error = y_true-y_out
        self.d_x = derivative_func(dot_sum)*2*(self.error)
        return self.d_x
    
    def d_Backprop(self, derivative_func, dot_sum):
        self.d_y = derivative_func(dot_sum)
        return self.d_y


net = Neural()

X = [[1,1],[1,0],[0,1],[0,0]]
Y = [[1],[0],[0],[0]]

layer1 = Layer(2, 4)
layer2 = Layer(4, 1)


for i in range(500):
    layer1.forward(X)
    layer2.forward(sigmoid(layer1.Y))

    net.Backprop(d_sigmoid, layer2.Y, sigmoid(layer2.Y), Y)
    layer2.ratio = np.dot(np.array(layer2.weights), net.d_x.T)
    layer2.weights += np.dot(np.array(sigmoid(layer1.Y)).T, net.d_x)
    layer2.biases += sum(net.d_x)

    net.d_Backprop(d_sigmoid, layer1.Y)
    layer1.ratio = net.d_y
    layer1.weights += np.dot(np.array(X).T, (layer1.ratio*layer2.ratio.T))
    layer1.biases += sum(layer1.ratio)


###
# W tym miejscu - zapis layer1.weights (2 zmienne) i layer1.bias (1 zmienna) do pliku json
###


layer1.forward(X)
layer2.forward(sigmoid(layer1.Y))
print(sigmoid(layer2.Y))