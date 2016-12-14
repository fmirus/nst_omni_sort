import nstbot
import numpy as np

bot = nstbot.OmniArmBot()
bot.connect(nstbot.SocketList({'motors': ['10.162.177.29', 54322],
                               'retina_left': ['10.162.177.29', 54320],
                               'retina_right': ['10.162.177.29', 54321],
                               'retina_arm': ['10.162.177.29', 54323],}))
bot.tracker('retina_left', True, tracking_freqs=[200], streaming_period=10000)
bot.tracker('retina_right', True, tracking_freqs=[200], streaming_period=10000)
bot.tracker('retina_arm', True, tracking_freqs=[200], streaming_period=10000)


import nengo

model = nengo.Network()
with model:

    def bot_base(t, x):
        bot.base_pos(x, msg_period=0.1)
        return x
    base = nengo.Node(bot_base, size_in=3)

    def bot_tracker(t):
        return np.hstack([
            bot.get_tracker_info('retina_left', 0),
            bot.get_tracker_info('retina_right', 0),
            bot.get_tracker_info('retina_arm', 0),
            ])

    tracker = nengo.Node(bot_tracker)


    x_data = nengo.Ensemble(n_neurons=100, dimensions=6, neuron_type=nengo.Direct())
    nengo.Connection(tracker[[0,3,4,7,8,11]], x_data)

    x_pos = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    def compute_pos(x):
        lx, lc, rx, rc, ax, ac = x

        c = (lc + rc + ac)
        if c <= 0.1:
            return 0
        return (lx*lc+rx*rc+ax*ac) / c

    nengo.Connection(x_data, x_pos, function=compute_pos)

    nengo.Connection(x_pos, base[2], transform=-1)


    def x_vals(t, x):
        return x
    x = nengo.Node(x_vals, size_in=3)
    nengo.Connection(tracker[[0,4,8]], x)


    def c_vals(t, x):
        return x
    c = nengo.Node(c_vals, size_in=3)
    nengo.Connection(tracker[[3,7,11]], c)
