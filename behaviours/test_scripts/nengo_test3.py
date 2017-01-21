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

  peak_inverted = nengo.Ensemble(n_neurons=100, dimensions=1)#, intercepts=nengo.dists.Uniform(0.1,1.0))

  diff = nengo.Ensemble(
          100, 1,
          intercepts=nengo.dists.Exponential(0.15, 0., 1.),
          encoders=nengo.dists.Choice([[1]]),
          eval_points=nengo.dists.Uniform(0., 1.))
  peak = nengo.Ensemble(100, 1)
      
  tau = 0.1
  timescale = 0.01
  dt = 0.001
  nengo.Connection(active, diff, synapse=tau/2)
  nengo.Connection(diff, peak, synapse=tau/2, transform=dt / timescale / (1 - np.exp(-dt / tau)))
  nengo.Connection(peak, diff, synapse=tau/2, transform=-1)
  nengo.Connection(peak, peak, synapse=tau)
  # def invert_func(x):
  #   return 1-x
  peak_inverted_init = nengo.Node([1])
  nengo.Connection(peak_inverted_init, peak_inverted)
  nengo.Connection(peak, peak_inverted.neurons, transform=np.ones((peak_inverted.n_neurons, 1))*-5)

  reset_peak_node = nengo.Node([0])
  reset_peak = nengo.Ensemble(n_neurons=100, dimensions=1)
  nengo.Connection(reset_peak_node, reset_peak)
  nengo.Connection(reset_peak, peak.neurons, transform=np.ones((peak.n_neurons, 1))*-5)


  nengo.Connection(finished, finished, synapse=0.5, transform=1)

  nengo.Connection(peak_inverted, finished, transform=-1)

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




