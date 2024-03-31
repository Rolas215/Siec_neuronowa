import numpy as np

#Neural network (Perceptron) that checks if set of two binary nums is true
class Neural:
    def __init__(self):
        self.bias = 1
        self.weights = [np.random.random(), np.random.random(), np.random.random()]
    
    def activate(self, x):
        # Sigmoid
        return 1/(1+np.exp(-x))

        # # ReLU
        # if x > 0:
        #     return x
        # else:
        #     return 0
    
        # if x > 0:
        #     return 1
        # else:
        #     return 0

    def test(self, input1, input2):
        return self.activate(input1*self.weights[0]+input2*self.weights[1]+self.bias*self.weights[2])
    
    def train(self, input1, input2, output):
        cost = output - self.test(input1, input2)
        self.weights[0] += cost * input1
        self.weights[1] += cost * input2
        self.weights[2] += cost * self.bias

    def getStats(self):
        print(f"W0: {self.weights[0]}, W1: {self.weights[1]}, W2: {self.weights[2]}, B:{self.bias}")


Net = Neural()
Net.getStats()

#Train to bahave like AND operator
for i in range(100):
    Net.train(1,1,1)
    Net.train(1,0,0)
    Net.train(0,1,0)
    Net.train(0,0,0)

Net.getStats()

#Output close to 1.0 is True
print(\
Net.test(1,1),\
Net.test(1,0),\
Net.test(0,1),\
Net.test(0,0)\
)
