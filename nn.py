import random
import numpy as np
'''data is a list in the format [[a, b, c], 
                                 [x, y, z]....]
where a,b or x, y are features while c, z is the label(0 or 1)
of that particular features in the dataset'''

class NN():

    def __innit__(self,data, epochs, learning_rate):
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        return self.train()

    def loss(self, x, y):
        return (x-y)^2

    def sigmoid(self,x):
        return(1/(1+np.exp(-x)))

    def slope_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self):
        weight1, weight2, bias = random.randint(), random.randint(), 1
        epochs = self.epochs
        learning_rate = self.learning_rate

        for i in range(epochs):#training loop
            random_point = random.randint(len(self.data))
            chosen_input1 = self.data[random_point[0]]
            chosen_in2 = self.data[random_point[1]]
            chosen_output = self.data[random_point[2]]
            
            z = weight1 * chosen_input1 + weight2 * chosen_in2 + bias
            pred = self.sigmoid(z)
            cost = self.loss(pred, chosen_output)

            #finding dervatives ie. slope
            slope_cost = 2(pred - chosen_output)
            
            slope_pred_wrt_z = self.slope_sigmoid(z)
            slope_cost_wrt_pred = slope_cost * slope_pred_wrt_z

            slope_cost_wrt_w1 = slope_cost_wrt_pred * chosen_input1
            slope_cost_wrt_w2 = slope_cost_wrt_pred * chosen_in2
            slope_cost_wrt_bias = 1
            
            #optmizing wts and bias to minimize cost
            weight1 -= learning_rate * slope_cost_wrt_w1
            weight2 -= learning_rate * slope_cost_wrt_w2
            bias -= learning_rate * slope_cost_wrt_bias
            
            #returning final and optimized weights and bias
            return weight1, weight2, bias  

