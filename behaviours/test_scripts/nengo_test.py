import nengo
import numpy as np


model = nengo.Network()

with model:
  freqs = [0.2, 0.4, 0.6, 0.8, 1.0]
  stim = nengo.Node(freqs)

  ens_diff = nengo.Ensemble(n_neurons=300, dimensions=len(freqs)-1)

  nengo.Connection(stim[:-1], ens_diff, transform=-1)
  nengo.Connection(stim[1:], ens_diff, transform=1)

  min_neurons = nengo.Ensemble(n_neurons=200, dimensions=1)

  odd = nengo.Ensemble(n_neurons=100*len(freqs), dimensions=len(freqs))
  left = nengo.Ensemble(n_neurons=100*len(freqs), dimensions=len(freqs))
  right = nengo.Ensemble(n_neurons=100*len(freqs), dimensions=len(freqs))

  def min_func(x):
    return min(x)
    
  nengo.Connection(ens_diff, min_neurons, synapse=0.01, function=min_func)

  def worst_func(x):
    ind_min = np.argmin(x)
    ind_max = np.argmax(x)
    ind_result = ind_min

    if ind_max > ind_min:
        ind_result += 1
    return np.eye(len(freqs))[ind_result]

  nengo.Connection(ens_diff, odd, function=worst_func)

  def get_left(x):
    ind = max(np.argmax(x)-1,0)
    return np.eye(len(freqs))[ind]
  nengo.Connection(odd, left, function=get_left)

  def get_right(x):
    ind = min(np.argmax(x)+1, len(freqs)-1)
    return np.eye(len(freqs))[ind]
  nengo.Connection(odd, right, function=get_right)

  negative_min = nengo.Ensemble(n_neurons=100, dimensions=1,
                                    encoders=nengo.dists.Choice([[1]]),
                                    intercepts=nengo.dists.Uniform(0.4, 0.9))

  def neg_min_func(x):
    if x < -0.05:
      return 1
    else:
      return 0
  nengo.Connection(min_neurons, negative_min, function=neg_min_func)
