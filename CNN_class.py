#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.special import softmax


# ## Conv layers



class ConvLayer:
    '''
    This class represents a convolution layer. 
    For a matter of simplicity we'll consider stride=1, pad=0, and no bias.
    '''

    def __init__(self, kernel_size, layer_size, input_depth = 1):
        self.kernel_size = kernel_size
        self.layer_size = layer_size
        self.input_depth = input_depth
        sigma = 2/(self.kernel_size[0]+self.kernel_size[1])
        self.filters = sigma * np.random.randn(layer_size, input_depth, kernel_size[0], kernel_size[1])
        self.stride = 1 
        #Let's choose Xavier initialization of the kernel: normal distribution with a mean of 0 and a variance of 1/kernel_size^2

    def input_coef(self, coef):
        '''
        Allows to create a convolution with its own weights.
        It is useful for the backprop method.
        '''
        self.coef = coef
        if self.coef.shape != self.filters.shape:
            print('wrong dimension')
        else:
            self.filters = self.coef
    

    #The convolution layer size will be made of layer_size filters of size kernel_size x kernel_size 
   
    def windows_gen(self, image):
        '''
        This method extract the submatrix from the input image in order to apply a kernel on each one of them 
        to create one "pixel" in the feature map.
        Image might be a 3D array if it is a hidden layer of convolution.
        '''
        if len(image.shape) == 2:
            image = image[np.newaxis,:, :]
        #if it is the first convolution layer
            
        d, h, w = image.shape
        im_region = []
        
        for i in range(h - (self.kernel_size[0] - 1)):#possibility to add padding and stride
            for j in range(w - (self.kernel_size[1] - 1)):
                im_region.append([image[0:d,i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1])]])
        return im_region

    def forward_prop(self, input_fm):
        '''
        This method allows to process an input through a layer of convolution.
        It takes as an input a image/feature map and return a feature map of size 
        (number of kernels in this layer, h - (kernel_size - 1), w - (kernel_size - 1) )
        '''
        if input_fm.shape[0] != self.input_depth:
            print('the input image depth doesn t correspond to the supposed depth on this layer')
        
        self.last_input = input_fm

        d, h, w = input_fm.shape
        output = []

        for i , kernel in enumerate(self.filters):
            for j, im_gen in enumerate(self.windows_gen(input_fm)):
                output.append(np.sum(im_gen * kernel))
        output = np.reshape(output,(self.layer_size,h - (self.kernel_size[0]-1),w - (self.kernel_size[1]-1)))
        self.last_output = output
        return output

    def back_prop(self, next_error, learning_rate = 0.01):
        '''
        In order to update the weights of the convolution we'll use the formula:
        gradient_matrix = convolution product of (last_input by next_error)
        next error being dE/dOutput
        
        The output of this backprop method is error previous layer which stands for 
        dE/dInput. We can get it by calculating:
        
        '''
        #update of the weights by calculating gradient_weight:
        gradient_weight = np.zeros(self.filters.shape)
        for n in range(self.layer_size): 
            for d in range(self.last_input.shape[0]):
                conv_error = ConvLayer([next_error.shape[1],next_error.shape[2]],1,1)
                #print(n,d)
                conv_error.input_coef(np.array([[next_error[d]]]))
                gradient_weight[n,d] = conv_error.forward_prop(np.array([self.last_input[d]]))
        self.filters += learning_rate * gradient_weight
        
        #return the partial derivative of the error for every input : dE/dinput
        error_previous_l = np.zeros(self.last_input.shape)
        
        for n in range(self.last_output.shape[0]): #we proceed by neuron
            for h in range(self.last_output.shape[1]):
                for w in range(self.last_output.shape[2]):
                    #here we treat the coefficient at the position d,h,w in the next_error array
                    #what are the corresponding weights
                    weights = self.filters[n] #weights for the considered neuron
                    error_previous_l[:,h:h+self.kernel_size[0],w:w+self.kernel_size[1]] += next_error[n,h,w] * weights 
                    #because we chose a default stride of 1
        
        return error_previous_l
        



# ## ReLU layers




class relu:
    """
    This class represents a relu layer. It takes a feature map as an input. 
    """
    def __init__(self,feature_map_shape):
        self.shape = feature_map_shape
    
    def forward_prop(self,feature_map):
        output = np.zeros(self.shape)
        if list(feature_map.shape) != list(self.shape):
            print('the feature map shape taken as an input doesnt correspond to the relu layer shape')
        
        for d in range(self.shape[0]):
            for h in range(self.shape[1]):
                for w in range(self.shape[2]): 
                    output[d,h,w] = np.max((feature_map[d,h,w],0))
        self.lastoutput = output
        return output
    
    def back_prop(self, next_error): 
        error_previous_l = np.zeros(next_error.shape)
        for d in range(next_error.shape[0]):
            for h in range(next_error.shape[1]):
                for w in range(next_error.shape[2]):
                    if self.lastoutput[d,h,w] != 0:
                        error_previous_l[d,h,w] = next_error[d,h,w]
        return error_previous_l





# ## Max Pool layers



class maxpool: 
    def __init__ (self, poolsize, stride = 1):
        self.poolsize = poolsize
        self.stride = stride
    
    
    def forward_prop(self, feature_map):
        self.input_shape = feature_map.shape
        output = np.zeros((self.input_shape[0],int(np.floor((self.input_shape[1]-self.poolsize)/self.stride)+1), int(np.floor((self.input_shape[2]-self.poolsize)/self.stride)+1)))
        for map_n in range(self.input_shape[0]):
            for i in np.arange(0,self.input_shape[1]-self.poolsize+1, self.stride):
                for j in np.arange(0,self.input_shape[2]-self.poolsize+1, self.stride):
                    output[map_n,int(i/self.stride),int(j/self.stride)] = np.max(feature_map[map_n,i:i+self.poolsize,j:j+self.poolsize])
        self.lastinput = feature_map
        self.lastoutput = output
        return output
    
    def back_prop(self, next_error):
        '''
        To process the backpropagation of the gradient through the MaxPool layer, we can notice that
        a MaxPool layer works like a Relu except that it is max([x[i] for i in [1:N]]) instead of max(x,0).
        So the gradient is 0 for the 'erased pixels' and 1 for the one corresponding to value'''
        if next_error.shape != self.lastoutput.shape: 
            print('the error array doesnt fit the output shape')
            
        error_previous_l = np.zeros(self.input_shape)
        for map_n in range(self.input_shape[0]):
            for i in np.arange(0,self.input_shape[1]-self.poolsize+1, self.stride):
                for j in np.arange(0,self.input_shape[2]-self.poolsize+1, self.stride):
                    window = self.lastinput[map_n,i:i+self.poolsize,j:j+self.poolsize]
                    for k in range(0,self.poolsize):
                        for l in range(0,self.poolsize):
                            if window[k,l] == self.lastoutput[map_n,int(i/self.stride),int(j/self.stride)]:
                                error_previous_l[map_n,k,l] += next_error[map_n,int(i/self.stride),int(j/self.stride)] # * 1
                                # else + 0
        
        return error_previous_l
    


# ## Flatten Layer




class flatten:
    '''
    This class represents a flatten layer in order to take the output of a conv layer and flatten it in order to 
    input it into a fully connected neural layer (Dense)'''
    def __init__(self,input_shape):
        self.input_shape = input_shape
        
    def forward_prop(self,feature_map):
        if list(feature_map.shape) != list(self.input_shape):
            print ('wrong input shape')
        output = np.ndarray.flatten(feature_map)
        return output
    
    def back_prop(self, next_error):
        error_previous_l = np.reshape(next_error,self.input_shape)
        return error_previous_l



# ## Dense Layers :




class Dense:
    '''
    This class corresponds to a fully connected layer.
    '''
    
    
    def __init__(self, size, input_shape):
        '''
        Initialization of the weights and biases of the neurons of the layer.
        '''
        
        self.size = size #number of neurons in the layer
        self.input_shape = input_shape #size of the input array
        #initialization of the weights and biases:
        sigma = np.sqrt(1/input_shape)
        self.weights = np.array([sigma * np.random.randn(input_shape) for i in range(size)])
        self.biases = sigma * np.random.rand(size)
        
        
    def forward_prop(self, layer_input):
        output = np.zeros(self.size)
        for i in range(self.size):
            output[i] = sum(self.weights[i]*layer_input) + self.biases[i]
            output[i] = np.max((output[i],0)) #RELU
        self.lastinput = layer_input
        self.lastoutput = output
        return output
 

    def back_prop(self, next_error, lr = 0.01): 
        '''
        This method has two objectives: 
        - update the weights by calculating dE/dw[i,j] and the biases dE/db[i]
        - return the error to be input in the gradient method of the previous layer dE/ds[i] l-1
        '''
       
        grad_weight = np.array([np.zeros(self.input_shape) for i in range(self.size)])
        grad_biases = np.zeros(self.size)
        for j in range(self.size): #for every neuron
            if self.lastoutput[j] == 0: #derivation of the relu is null or equal to 1 if self.lastoutput[j] > 0
                for i in range(self.input_shape): #for every weight
                    grad_weight[j,i] = 0 #gradient for the i-th neuron of the j-th
                grad_biases = 0
            else :
                for i in range(self.input_shape):
                    grad_weight[j,i] = next_error[j] * self.lastinput[i]
                grad_biases = next_error[j]
        
        erreur_previous_l = np.zeros(self.input_shape)
        for j in range(self.input_shape):
            if self.lastinput[j] == 0 :
                erreur_previous_l[j] = 0
            else :
                erreur_previous_l[j] = sum([self.weights[i,j]*next_error[i] for i in range(self.size)])
            
        #update of the weights and biases:
        self.weights -= lr * grad_weight
        self.biases -= lr * grad_biases
        
        return erreur_previous_l


# ## Last layer :  Softmax




class softmaxlayer:
    '''
    Last layer of the network.
    '''
    def __init__ (self, input_shape):
        self.input_shape = input_shape 
        #the input shape is the same as the output shape on the last layer. It corresponds to the number of class.
        
    def forward_prop(self, layer_input):
        output = softmax(layer_input)
        self.lastoutput = output
        return output
        
    def cross_entropy(self, pred, target, epsilon = 0):
        '''
        This function compute the cross entropy from a predictions and the actual labels.
        The output value allows to process the backpropagation of the gradient.
        '''
        pred = np.clip(pred, epsilon, 1. - epsilon)
        ce = - np.sum(np.log(pred) * target) 
        return ce
    
    def back_prop(self, target):
        '''
        s[i] represents here the input of the softmax layer; which is the output of the last Dense layer.
        So the back_prop method here is supposed to return :
        dE/ds[i] for every i from 1 to number of classes.
        '''
        #if target.shape[0] != self.input_shape :
            #print('the last layer size doesnt correspond to the number of classes')
        grad = self.lastoutput - target
        return grad


