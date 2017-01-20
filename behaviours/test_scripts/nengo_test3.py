import nengo
import numpy as np

model = nengo.Network()

with model:
  activation = nengo.Node([0])

  active = nengo.Ensemble(n_neurons=100, dimensions=1)
  nengo.Connection(activation, active)

  inactive = nengo.Ensemble(n_neurons=100, dimensions=1)

  def invert_func(x):
    return 1-x
  nengo.Connection(active, inactive, function=invert_func)

  finished = nengo.Ensemble(n_neurons=100, dimensions=1)

  nengo.Connection(finished, finished, synapse=0.5, transform=1)

  nengo.Connection(inactive, finished, transform=-1)

  nengo.Connection(active, finished, transform=1)

  output_act = nengo.Node(None,size_in=1)

  nengo.Connection(inactive, output_act, synapse=None)

  ens_array = nengo.networks.EnsembleArray(n_neurons=400, n_ensembles=3, ens_dimensions=2, radius=1.5)

  ens_array.add_output('scaled', function=lambda x: x[0]*x[1])

  for ens in ens_array.ensembles:
    nengo.Connection(active, ens[0])

  stim = nengo.Node([0,0,0])

  for i, ens in enumerate(ens_array.ensembles):
    nengo.Connection(stim[i], ens[1])

  for ens in ens_array.ensembles:
    nengo.Connection(inactive, ens.neurons, transform=np.ones((ens.n_neurons, 1))*-5)
  
  #nengo.Connection(active, output_act, transform=0.4)




