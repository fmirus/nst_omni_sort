import nengo
import numpy as np

model = nengo.Network()

with model:
  input_node = nengo.Node([-1,0])

  #input = nengo.Ensemble(n_neurons=200, dimensions=2)

  active = nengo.Ensemble(n_neurons=100, dimensions=1)
  # nengo.Connection(active, active, synapse=0.1, transform=-1)
  spd = nengo.Ensemble(n_neurons=100, dimensions=1)
  peak_inverted = nengo.Ensemble(n_neurons=100, dimensions=1)#, intercepts=nengo.dists.Uniform(0.1,1.0))

  nengo.Connection(input_node[0], active)#, function=invert_func)
  nengo.Connection(input_node[1], spd)

  reached_pos = nengo.Ensemble(n_neurons=100, dimensions=1,
                                              neuron_type=nengo.LIFRate())   

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

  nengo.Connection(reached_pos, reached_pos, synapse=0.1)
  # recurrent connection
  # nengo.Connection(self.reached_pos, self.reached_pos, synapse=0.05)
  def reached_pos_func(x):
    # move side is active and speed is low
    if abs(x) < 0.1:
      return 1
    else:
      return -1
  nengo.Connection(spd, reached_pos, function=reached_pos_func)
  nengo.Connection(peak_inverted, reached_pos.neurons, transform=np.ones((reached_pos.n_neurons, 1))*-5)

