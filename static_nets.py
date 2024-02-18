import torch
import numpy as np
import torch.nn as nn

# Static-architecture neural network with output logit and a
# user-specified number and size of hidden layers.
class BinaryClassifier(nn.Module):
  def __init__(self, input_size, hidden_sizes = [], nonlin = 'relu'):
    super().__init__()
    # Input and hidden layers
    layer_sizes = [input_size, *hidden_sizes]
    layers = []
    self.H = np.sum(hidden_sizes) # Total number of hidden units

    self.nonlin = nn.ReLU() if nonlin == 'relu' else nn.Sigmoid()
    gain = nn.init.calculate_gain(nonlin)

    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
      fcl = nn.Linear(in_size, out_size)
      layers.append(fcl)
      layers.append(self.nonlin)
      # Initialise layer parameters
      nn.init.xavier_uniform_(fcl.weight, gain=gain)
      bias_bound = 1 / np.sqrt(fcl.weight.size(1))
      nn.init.uniform_(fcl.bias, -bias_bound, bias_bound)

    self.fcl_seq = nn.Sequential(*layers)

    # Output layer
    self.out = nn.Linear(layer_sizes[-1], 1)
    sigmoid_gain = nn.init.calculate_gain('sigmoid')
    bias_bound = 1 / np.sqrt(self.out.weight.size(1)) # for bias
    nn.init.xavier_uniform_(self.out.weight, gain=sigmoid_gain)
    nn.init.uniform_(self.out.bias, -bias_bound, bias_bound)
  
  def forward(self, x):
    return self.out(self.fcl_seq(x)) # logit