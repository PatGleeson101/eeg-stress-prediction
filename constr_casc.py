import torch
from torch import zeros # For brevity
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

""" General notes:

This file defines

(1) a generic constructive cascade architecture applicable to
both Casper and Cascor, and 

(2) A Casper subclass.

The (incomplete) Cascor subclass has been removed, as it does
not appear in the submitted paper.

Parameter() automatically sets requires_grad=True

"""

class DangerousColumnAssign(torch.autograd.Function):
  """
  Assign *in-place* to a column of a tensor, but don't
  mark the result as modified (dirty). Allows autograd to
  work when writing cascaded neuron activations to successive
  columns of the *same* storage tensor (i.e. avoid a separate
  [N,1+D+H]-size tensor for each neuron's inputs). However,
  autograd may fail silently if this is used outside of the
  present constructive-cascade architecture.

  For convenience, we also avoid having to flatten the input.

  Credit:
  https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/5

  See also:
  https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
  """

  @staticmethod
  def forward(ctx, cache, new_col_value, col_idx):
    ctx.col_idx = col_idx
    cache.data[:, col_idx:(col_idx+1)] = new_col_value
    # Notably, we don't call ctx.mark_dirty(cache). This is the whole point.
    return cache

  @staticmethod
  def backward(ctx, grad_cache):
    # Return the gradient with respect to each input.
    return grad_cache, grad_cache[:, ctx.col_idx:(ctx.col_idx+1)], None, None


class ConstrCascBinaryClassifier(nn.Module):
  """
  Generic constructive cascade network with single-neuron hidden 'layers'.

  Single output neuron, representing a classification logit.

  By default, all weights are parameters. Subclasses (i.e. Cascor/Casper)
  can override this.

  Weights are 2D column vectors, consistent with Pytorch convention.

  Hidden layers use sigmoid activation.
  """
  def __init__(self, X_train):
    super().__init__()
    self.D_in = X_train.shape[1]
    self.H = 0 # Number of hidden neurons

    # Initialise weights (bias is first element)
    self.out_weight = Parameter(zeros(1 + self.D_in, 1)) # requires_grad
    nn.init.xavier_uniform_(self.out_weight, gain=nn.init.calculate_gain('sigmoid'))

    self.hidden_weights = [] # List of column vectors, each the weights for a hidden neuron

    # Activation cache
    self.activations = zeros( (X_train.shape[0], 1 + self.D_in), 
      requires_grad=False)
    self.activations[:,0] = 1 # Bias neuron
    self.activations[:, 1:(1+self.D_in)] = X_train # Input activations

  def add_neuron(self, w):
    # Add a new hidden neuron to the network, with weights w
    if w.numel() != (1 + self.D_in + self.H):
      raise ValueError("Incorrect number of weights for new neuron.")
    
    # Detach w from its graph, clone it to avoid modifying original,
    # and make it a column vector.
    self.hidden_weights.append(Parameter(w.detach().clone().reshape(-1,1)))

    # Extend activations. Shouldn't happen too often, so ok to reallocate mem.
    self.activations = F.pad(self.activations, (0,1,0,0), mode='constant', value=0) # inherits requires_grad = False from self.activations

    # Initialise connection to output neuron.
    self.extend_out_weight() # Differs for casper/cascor (to deal with L1/L2/L3 in Casper)
    self.H += 1

  def extend_out_weight(self):
    pass
    """
    Casper requires different handling of the weights to Cascor, since
    it applies different learning rates to different parameters.
    extend_out_weight is thus overriden in Casper/Cascor.
    """
  
  def compute_output(self, A=None):
    pass
    """ compute_output distinguishes Casper and Cascor:

    Casper must treat differently the weight connecting the most recent hidden neuron to the output, to apply L1 learning rate. This
    requires separate tensors, which we need to combine explicitly in the forward pass.

    Also, Cascor doesn't have to redo the entire forward pass.
    """

  def forward(self, X=None):
    if X is None: # Use the training data, and cache the result.
      A = self.activations
    else: # New data
      A = zeros((X.shape[0], 1 + self.D_in + self.H), requires_grad=False)
      A[:,1] = 1 # Bias neuron
      A[:, 1:(1+self.D_in)] = X # Input activations
    
    # Iteratively evaluate hidden neurons
    for h in range(0, self.H):
      h_out = F.sigmoid(A[:, :(self.D_in + h + 1)] @ self.hidden_weights[h])
      A = DangerousColumnAssign.apply(A, h_out, self.D_in + h + 1)
      
    # Output neuron (pre-sigmoid)
    out_logit = self.compute_output(A) # Distinguishes casper/cascor

    if X is None:
      self.out_logit = out_logit # Cache if using training data

    return out_logit

  def gen_neuron(self):
    # Generate a new neuron, and return its weights.
    w = zeros(1 + self.D_in + self.H, 1)
    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('sigmoid'))
    return w

class Casper(ConstrCascBinaryClassifier):
  def __init__(self, X_train, L1 = 0.2, L2 = 0.005, L3 = 0.001):
    super().__init__(X_train)
    self.L1, self.L2, self.L3 = L1, L2, L3
    # Default L1,L2,L3 are those from the original casper paper.

    # Redefine out_weight as two tensors, one for most recent hidden neuron
    # (so we can apply separate learning rates)
    self.out_weights_old = self.out_weight
    self.out_weight_new = None

  def get_parameters(self):
    result = []
    if self.H == 0: # Initial training uses L1 on output (as in paper)
      result.append({'params': [self.out_weights_old], 'lr': self.L1})
    else:
      # Most recent hidden neuron.
      result.append({'params': [self.hidden_weights[-1]], 'lr': self.L1})
      # Output connection to most recent hidden neuron
      result.append({'params': [self.out_weight_new], 'lr': self.L2})
      # All other weights
      old = [self.out_weights_old]
      if self.H > 1:
        old += self.hidden_weights[:-1]
      
      result.append({'params': old, 'lr': self.L3})

    return result

  def compute_output(self, A=None):
    # Output is a logit, as usual.
    if A is None: # Use cached activations
      return self.forward() # calls compute_output(A=self.activations)
    else:
      if self.H > 0:
        return A[:,:-1] @ self.out_weights_old + A[:,-1:] * self.out_weight_new
      else:
        return A @ self.out_weights_old

  def extend_out_weight(self):
    if self.H > 0:
      self.out_weights_old = Parameter(
        torch.cat((self.out_weights_old, self.out_weight_new), dim=0))
    # Weight connecting latest hidden neuron to output is initially zero:
    self.out_weight_new = Parameter(zeros(1,1))


#torch.autograd.set_detect_anomaly(True)