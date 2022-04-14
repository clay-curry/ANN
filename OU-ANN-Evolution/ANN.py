from operator import le, matmul
from re import L, S
from this import d
from typing import List
import numpy as np

def tanh(x, ddx=False): 
    if ddx:
        return 1 - np.square(np.tanh(x))
    return np.tanh(x)

def leaky_ReLU(x, ddx=False):
    if ddx:
        return np.where(x > 0, 1, 0.01)
    return np.where(x > 0, x, x * 0.01)

def sigmoid(x, ddx=False):
    if ddx:
        return np.exp(-x) / np.square(1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))

def half_SSE(x, y, ddx=False):
    if ddx:
        return np.subtract(x,y)
    return np.linalg.norm(x-y)**2 / 2


class ANN:
    def __init__(self, weight_mat_list: List[float], activation: callable, cost: callable, bias = False):
        """Models a simple multi-layer feed-forward ANN with ReLU activation functions and no skip connections.
    
        weight_mat_list:
            * tuple or list of length 'n' of matrices (row-major) whose rows and columns determine neurons per layer and starting weights
            * weight_mat_list[0] connects the input layer to the first hidden layer
            * weight_mat_list[n-1] connects the last hidden layer to the output layer
            * in general, weight_mat_list[k] connects layer '(k)' to layer '(k+1)'
            * artificial neurons in each layer are 0-indexed
            * weight_mat_list[k][i][j] points away from neuron '(j)' in layer '(k)' and enters neuron '(i)' in layer '(k+1)'
            * unless 'bias' is set to true, the bias at each layer is automatically initialized to -1

        Note: linearly transforming '(m)' activations requires a matrix with '(m+1)' columns
        """
        self.last_z = []
        self.last_a = []
        self.activation = activation
        self.cost = cost
        self.num_layers = len(weight_mat_list)
        if not bias:   # then initialize bias  
            for layer in range(self.num_layers):
                dendrites = weight_mat_list[layer]
                weight_mat_list[layer] = np.column_stack(
                        (
                        dendrites,                       # initializing weights
                        np.ones(np.shape(dendrites)[0])  # initializing bias
                        ) 
                    )
        self.weight_mat_list = weight_mat_list

        

    def eval(self, input):         
        L = self.num_layers - 1         # Output should be linear <=> skip ReLU on layer L
        W_list = self.weight_mat_list   # Alias for readability
        z = np.reshape(input, (1,-1))   # 0. store inputs as a column vector
        a = np.reshape(input, (1,-1))   # 0. store inputs as a column vector
        z_hist = [z]                    # 1. store net sum at each summation unit
        a_hist = [a]                    # 1. and store activations for backprop
        for layer in range(L):          # 2. Forward propogation starting at layer 0 (input)
            W = W_list[layer]           # 3. get weights connecting layer L to layer L+1
            z = np.matmul(W, np.append(a, [[-1]]))  # 4. Activations + Bias
            a = self.activation(z)      # 5. Activation = ReLU(Net)
            z_hist.append(z.reshape(1,-1)) # 6. Store net and activations for backprop
            a_hist.append(a.reshape(1,-1)) # 7. Store net and activations for backprop
        
        a = z = np.matmul(W_list[L], np.append(a, [[-1]]))
        z_hist.append(z.reshape(1,-1)) # 7. Output layer is full linear
        a_hist.append(a.reshape(1,-1)) 
        self.last_z = z_hist            # 8. Required for backprop
        self.last_a = a_hist            # 9. Required for backprop
        return a

    def back_prop(self, expected_output: List[float], input=None, learning_rate=0.0001):
        if input!=None:
            self.eval(input)        
        
        N = self.num_layers                             # back prop starts at N, moves toward 0
        weights = self.weight_mat_list                  # weights[j] - maps layer j to j+1
        summation_vector = self.last_z                  # last AN summations  (last_z[0] = input)
        activation_vector = self.last_a                 # last AN activations (last_a[0] = input)
        da = lambda net : self.activation(net,ddx=True) # derivative of activation function

        # gradC enables the backprop loop to dynamically compute dC wrt dW^(L) for
        # L = N to 0 with the relation dC/dW^(L) = dC/dA^(L+1) dA^(L+1)/dW^(L)
        gradC = self.cost(activation_vector[N], expected_output, ddx=True).reshape((-1,1))
        dAdN = np.full_like(gradC, 1)                   # dAdN = 1 <=> no activation on layer N 
        dNdW = np.append(activation_vector[N-1],[[-1]]) # for dW = (gradC dAdN) dNdW
        W = weights[N-1]                                # store for dynamic backprop
        dW = np.matmul(np.multiply(gradC,dAdN),dNdW.reshape(1,-1)) # for dW = (gradC dAdN) dNdW
        weights[N-1] = W - learning_rate * dW
        for n in range(N-1, 0, -1):
            dAdN = da(summation_vector[n+1])
            gradC = np.matmul(W[:,:-1].T,np.multiply(gradC,dAdN.T)) # dynamically computes dC/dA
            dNdW = np.append(activation_vector[n-1],[[-1]])
            dAdN = da(summation_vector[n])
            W = weights[n-1]
            dW = np.matmul(np.multiply(gradC,dAdN.T), dNdW.reshape(1,-1))
            weights[n-1] = W - learning_rate * dW
        



if __name__ == "__main__":
    np.random.seed(seed=None)
    L1 = np.ones(shape=(3,1))
    L2 = np.ones(shape=(20,3))
    L3 = np.ones(shape=(1,20))    
    weight_matrices = [L1,L2, L3]
    ann = ANN(weight_matrices, activation=tanh, cost=half_SSE)


    f1 = lambda x : x**4 - 22*x**2
    x = np.random.uniform(low=(-5.0), high=(5.0), size=(3))
    y = f1(x)

    for i in range(50):
        print(f"Trained {i} times")
        for i in range(len(x)):    
            out = ann.eval(x[i])
            print(f"{i}. x = {x[i]}; y = {out[0]}; expect = {y[i]}; diff = {out[0]-y[i]})")

        for l in range(len(ann.weight_mat_list)):
                print(ann.weight_mat_list[l])    
        
        for epoch in range(1000):
            for i in range(len(x)):    
                ann.back_prop(y[i],x[i], 
                learning_rate=.001)



    
    '''train_input = np.random.uniform(low=(-5.0), high=(5.0), size=(30))
    validate_input = np.random.uniform(low=(-5.0), high=(5.0), size=(10))
    test_input = np.linspace(start=(-5.0), stop=(5.0), num=200)
    
    train_output = f1(train_input)
    validate_output = f1(validate_input)
    test_output = f1(test_input)

    out = []
    for x in test_input:
        out.append(ann.eval(x))
    print(f'before out: {np.mean(out)}')
    print(f'before cost: {ann.cost(out,test_output)}')


    # Stochastic gradient descent
    train_input_size = len(train_input)
    num_epochs = 5
    for i in range(train_input_size*num_epochs):
        ann.back_prop(expected_output=train_output[i%train_input_size],
        input=train_input[i%train_input_size], learning_rate=0.001)

    for x in test_input:
        out.append(ann.eval(x))
    
    print(f'after out: {np.mean(out)}')
    print(f'after cost: {ann.cost(out,test_output)}')
'''