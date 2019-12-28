# CNN-from-scratch
This repository contains the script of a CNN  coded from scratch using numpy

nb: you can upload pretrained weight to the network using :

with open("weights.txt", "rb") as fp:
    weights = pickle.load(fp)
    
net.upload_weights(weights)
