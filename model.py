# PROGRAMMER: Laura Dhatt
# DATE CREATED:     March 3, 2019                            
# REVISED DATE:     March 4, 2019
# PURPOSE: File used to build deep neural network model that is built dynamically
#          Includes define for forward function
#          
#       
##
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from torchvision import datasets, transforms
from collections import OrderedDict
import torchvision.models as models


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        ''' Builds a feedforward network with arbitrary hidden layers.
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of layer sizes of the hidden layers
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        
        # Add a variable number of more hidden layers + Relu and dropout.  
        # extend rather than append when building via iterable (e.g.zip())
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])            
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=0.5)
        
    
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x
        