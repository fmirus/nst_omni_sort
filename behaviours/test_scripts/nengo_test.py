import nengo
import numpy as np


model = nengo.Network()

with model:
  input = [0.2, 0.4, 0.6, 0.8, 1.0]
  stim = nengo.Node(input)

  ens_diff = nengo.Ensemble(n_neurons=300, dimensions=len(input)-1)

  nengo.Connection(stim[:-1], ens_diff, transform=-1)
  nengo.Connection(stim[1:], ens_diff, transform=1)

  min_neurons = nengo.Ensemble(n_neurons=200, dimensions=1)

  argmin_neurons = nengo.Ensemble(n_neurons=400, dimensions=len(input)-1)

  def min_func(x):
    return min(x)
    
  nengo.Connection(ens_diff, min_neurons, synapse=0.01, function=min_func)

  def worst_func(x):
    return np.eye(len(x))[np.argmin(x)]

  nengo.Connection(ens_diff, argmin_neurons, function=worst_func)

  negative_min = nengo.Ensemble(n_neurons=100, dimensions=1,
                                    encoders=nengo.dists.Choice([[1]]),
                                    intercepts=nengo.dists.Uniform(0.4, 0.9))

  def neg_min_func(x):
    if x < -0.05:
      return 1
    else:
      return 0
  nengo.Connection(min_neurons, negative_min, function=neg_min_func)
