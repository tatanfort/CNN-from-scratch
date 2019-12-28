# CNN-from-scratch
This repository contains the script of a CNN  coded from scratch using only numpy.
The Back Propagation of the algorithm chosen is a SGD. 
The architecture is :

We define the network with the following architecture:
- A first Convolution layer with __8 kernels of size 3 by 3__ and a __ReLU__ activation function
- A __MaxPool__ layer
- A second Convolution layer with __16 kernels of size 3 by 3__ and a __ReLU__ activation function
- A __Flatten__ Layer 
- A first Dense layer with __64 neurons__ with a __ReLU__ activation function
- A first Dense layer with __32 neurons__ with a __ReLU__ activation function
- A first Dense layer with __10 neurons__ with a __ReLU__ activation function
- A __Softmax__ layer

nb: you can upload pretrained weight to the network (weights.txt)
